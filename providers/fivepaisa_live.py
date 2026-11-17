"""Read-only 5paisa Xstream WebSocket market-data adapter.

This module subscribes only to NSE cash market feed (MarketFeedV3). It does
not import or expose any order-management operation. Callbacks receive
normalized ticks and completed time-based OHLCV candles.
"""
from __future__ import annotations

import json
import threading
from typing import Callable
from urllib.parse import quote

import pandas as pd
import websocket

WEBSOCKET_URL = "wss://openfeed.5paisa.com/feeds/api/chat"


def _pick(mapping: dict, *names, default=None):
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return default


def _tick_timestamp(value) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Tick timestamp is missing.")
    if text.startswith("/Date(") and text.endswith(")/"):
        millis = int(text[6:-2].split("+")[0].split("-")[0])
        return pd.to_datetime(millis, unit="ms", utc=True).tz_convert("Asia/Kolkata")
    stamp = pd.Timestamp(text)
    if stamp.year < 2000:
        raise ValueError("Tick timestamp is an invalid provider sentinel.")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Kolkata")
    else:
        stamp = stamp.tz_convert("Asia/Kolkata")
    return stamp


def _extract_tick_rows(payload):
    if isinstance(payload, list):
        for item in payload:
            yield from _extract_tick_rows(item)
        return
    if not isinstance(payload, dict):
        return
    if _pick(payload, "Token", "ScripCode") is not None and _pick(payload, "LastRate", "LTP") is not None:
        yield payload
        return
    for key in ("Data", "data", "body", "Body", "MarketFeedData"):
        child = _pick(payload, key)
        if isinstance(child, (list, dict)):
            yield from _extract_tick_rows(child)


def parse_live_tick_message(message: str | bytes, code_to_symbol: dict[int, str]) -> list[dict]:
    """Parse 5paisa MarketFeedV3 JSON into normalized tick records."""
    try:
        payload = json.loads(message)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("5paisa live feed sent invalid JSON.") from exc

    output = []
    for row in _extract_tick_rows(payload):
        token = _pick(row, "Token", "ScripCode")
        try:
            code = int(token)
            price = float(_pick(row, "LastRate", "LTP"))
            if price <= 0:
                continue
            timestamp = _tick_timestamp(_pick(row, "TickDt", "TimeStamp", "Timestamp"))
        except (TypeError, ValueError, OverflowError):
            continue
        symbol = code_to_symbol.get(code, str(code))
        total_qty = _pick(row, "TotalQty")
        last_qty = _pick(row, "LastQty")
        try:
            total_qty = int(float(total_qty)) if total_qty is not None else None
        except (TypeError, ValueError):
            total_qty = None
        try:
            last_qty = int(float(last_qty)) if last_qty is not None else 0
        except (TypeError, ValueError):
            last_qty = 0
        output.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "scrip_code": code,
            "last_price": price,
            "last_qty": max(last_qty, 0),
            "total_qty": max(total_qty, 0) if total_qty is not None else None,
            "open": _pick(row, "OpenRate", "Open"),
            "high": _pick(row, "High"),
            "low": _pick(row, "Low"),
            "previous_close": _pick(row, "PClose", "PreviousClose"),
            "exchange": _pick(row, "Exch", default="N"),
            "segment": _pick(row, "ExchType", default="C"),
        })
    return output


class LiveCandleBuilder:
    """Aggregate ticks into completed 1/5/15-minute OHLCV bars per symbol."""

    def __init__(self, interval_minutes: int):
        if isinstance(interval_minutes, bool) or interval_minutes not in (1, 5, 15):
            raise ValueError("Live candle interval must be 1, 5 or 15 minutes.")
        self.interval_minutes = interval_minutes
        self._current: dict[str, dict] = {}
        self._last_total_qty: dict[str, tuple[str, int]] = {}
        self._last_tick_timestamp: dict[str, pd.Timestamp] = {}

    def add_tick(self, tick: dict) -> dict | None:
        symbol = str(tick["symbol"])
        timestamp = pd.Timestamp(tick["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Kolkata")
        else:
            timestamp = timestamp.tz_convert("Asia/Kolkata")
        bucket = timestamp.floor(f"{self.interval_minutes}min")

        current = self._current.get(symbol)
        last_timestamp = self._last_tick_timestamp.get(symbol)
        if last_timestamp is not None and timestamp < last_timestamp:
            return None
        if current is not None and bucket < current["timestamp"]:
            return None
        self._last_tick_timestamp[symbol] = timestamp

        total = tick.get("total_qty")
        last_qty = int(tick.get("last_qty") or 0)
        session = timestamp.strftime("%Y-%m-%d")
        previous = self._last_total_qty.get(symbol)
        if total is not None:
            total = int(total)
            delta = total - previous[1] if previous and previous[0] == session else last_qty
            volume_delta = max(delta, 0)
            self._last_total_qty[symbol] = (session, total)
        else:
            volume_delta = max(last_qty, 0)

        if current is None:
            self._current[symbol] = {
                "timestamp": bucket, "symbol": symbol,
                "open": float(tick["last_price"]), "high": float(tick["last_price"]),
                "low": float(tick["last_price"]), "close": float(tick["last_price"]),
                "volume": volume_delta, "is_complete": False,
            }
            return None

        if bucket == current["timestamp"]:
            price = float(tick["last_price"])
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += volume_delta
            return None

        completed = dict(current)
        completed["is_complete"] = True
        self._current[symbol] = {
            "timestamp": bucket, "symbol": symbol,
            "open": float(tick["last_price"]), "high": float(tick["last_price"]),
            "low": float(tick["last_price"]), "close": float(tick["last_price"]),
            "volume": volume_delta, "is_complete": False,
        }
        return completed


class FivePaisaLiveFeed:
    """Background WebSocket subscriber for read-only NSE cash market ticks."""

    def __init__(
        self,
        access_token: str,
        client_code: str,
        api_key: str,
        instruments: dict[str, int],
        on_tick: Callable[[dict], None],
        on_error: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        if not access_token or not client_code or not api_key:
            raise ValueError("5paisa access token, client code and API key are required.")
        if not instruments:
            raise ValueError("At least one NSE cash instrument is required.")
        if len(instruments) > 50:
            raise ValueError("Subscribe to no more than 50 instruments in one session.")
        normalized = {}
        for symbol, code in instruments.items():
            code = int(code)
            if code <= 0:
                raise ValueError("Every instrument must have a positive 5paisa ScripCode.")
            normalized[str(symbol).upper()] = code
        self._access_token = access_token
        self._client_code = client_code
        self._instruments = normalized
        self._code_to_symbol = {code: symbol for symbol, code in normalized.items()}
        self._on_tick = on_tick
        self._on_error = on_error or (lambda message: None)
        self._on_status = on_status or (lambda message: None)
        self._app = None
        self._thread = None
        self._stop = threading.Event()

    def _subscription(self, operation: str) -> str:
        payload = {
            "Method": "MarketFeedV3",
            "Operation": operation,
            "ClientCode": self._client_code,
            "MarketFeedData": [
                {"Exch": "N", "ExchType": "C", "ScripCode": code}
                for code in self._instruments.values()
            ],
        }
        return json.dumps(payload, separators=(",", ":"))

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        token = quote(self._access_token, safe=".-_~")
        client = quote(self._client_code, safe="")
        url = f"{WEBSOCKET_URL}?Value1={token}|{client}"

        def on_open(ws):
            ws.send(self._subscription("Subscribe"))
            self._on_status("WebSocket connected; subscription sent; waiting for the first market tick.")

        def on_message(ws, message):
            try:
                for tick in parse_live_tick_message(message, self._code_to_symbol):
                    self._on_tick(tick)
            except Exception:
                self._on_error("Could not parse a 5paisa live market message.")

        def on_error(ws, error):
            self._on_error("5paisa live feed connection error: " + type(error).__name__)

        def on_close(ws, status_code, message):
            self._on_status("5paisa live market feed disconnected.")

        def run():
            retry_delay = 1.0
            while not self._stop.is_set():
                self._app = websocket.WebSocketApp(
                    url, on_open=on_open, on_message=on_message,
                    on_error=on_error, on_close=on_close,
                )
                self._app.run_forever(ping_interval=20, ping_timeout=10)
                if self._stop.is_set():
                    break
                self._on_status(
                    f"5paisa feed disconnected; reconnecting in {int(retry_delay)}s."
                )
                if self._stop.wait(retry_delay):
                    break
                retry_delay = min(retry_delay * 2, 30.0)

        self._thread = threading.Thread(target=run, name="fivepaisa-live-feed", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 3.0):
        self._stop.set()
        app = self._app
        if app is not None:
            try:
                if app.sock and app.sock.connected:
                    app.send(self._subscription("Unsubscribe"))
            except Exception:
                pass
            try:
                app.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
