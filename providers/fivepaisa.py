"""Read-only 5paisa Xstream authentication and historical market-data adapter.

No order-management endpoints are implemented here. Access tokens are kept in
Streamlit session state by the caller and are never written to disk or logged.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import gzip
import zipfile
from urllib.parse import urlencode
import re

import pandas as pd
import requests

OAUTH_BASE = "https://dev-openapi.5paisa.com/WebVendorLogin/VLogin/Index"
ACCESS_TOKEN_URL = "https://openapi.5paisa.com/VendorsAPI/Service1.svc/GetAccessToken"
MARKET_FEED_URL = "https://Openapi.5paisa.com/VendorsAPI/Service1.svc/V1/MarketFeed"
MARKET_FEED_API_UID = "ka7SFqAU6SC"
# Retained for legacy parser tests; the app no longer depends on this cache endpoint.
SCRIP_MASTER_URL = "https://openapi.5paisa.com/VendorsAPI/Service1.svc/ScripMaster/segment/All"
HISTORICAL_URL = "https://openapi.5paisa.com/V2/historical/N/C/{scrip_code}/{interval}"


@dataclass(frozen=True)
class ProviderStatus:
    connected: bool
    message: str


def _secret(secrets, *names: str) -> str:
    for name in names:
        try:
            value = secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            pass
    return ""


def get_credentials(secrets) -> dict:
    """Read supported top-level or [fivepaisa] Streamlit secrets."""
    nested = {}
    try:
        nested = secrets.get("fivepaisa", {}) or {}
    except Exception:
        nested = {}
    class SecretView:
        def get(self, key, default=None):
            value = _secret(secrets, key)
            return value if value else _secret(nested, key, key.lower())
    view = SecretView()
    return {
        "api_key": _secret(view, "FIVEPAISA_API_KEY", "FIVEPAISA_APP_KEY", "API_KEY", "APP_KEY", "api_key", "user_key"),
        "encryption_key": _secret(view, "FIVEPAISA_ENCRYPTION_KEY", "ENCRYPTION_KEY", "encryption_key"),
        "user_id": _secret(view, "FIVEPAISA_USER_ID", "USER_ID", "user_id"),
        "redirect_url": _secret(view, "FIVEPAISA_REDIRECT_URL", "REDIRECT_URL", "redirect_url"),
    }


def connection_status(secrets=None, access_token: str | None = None) -> ProviderStatus:
    if access_token:
        return ProviderStatus(True, "5paisa session authenticated for read-only market data.")
    if secrets is None:
        return ProviderStatus(False, "5paisa credentials are not configured.")
    credentials = get_credentials(secrets)
    missing = [key for key in ("api_key", "encryption_key", "user_id", "redirect_url") if not credentials[key]]
    if missing:
        return ProviderStatus(False, "5paisa setup incomplete. Missing secret fields: " + ", ".join(missing))
    return ProviderStatus(False, "Credentials configured. Complete today's secure 5paisa login to fetch historical candles.")


def build_oauth_url(api_key: str, redirect_url: str, state: str = "chart-reader") -> str:
    if not api_key or not redirect_url:
        raise ValueError("API key and registered redirect URL are required.")
    return OAUTH_BASE + "?" + urlencode({
        "VendorKey": api_key,
        "ResponseURL": redirect_url,
        "State": state,
    })


def exchange_request_token(credentials: dict, request_token: str) -> tuple[str, str]:
    """Exchange short-lived OAuth RequestToken for daily AccessToken."""
    if not request_token:
        raise ValueError("5paisa RequestToken is missing.")
    payload = {
        "head": {"Key": credentials["api_key"]},
        "body": {
            "RequestToken": request_token,
            "EncryKey": credentials["encryption_key"],
            "UserId": credentials["user_id"],
        },
    }
    try:
        response = requests.post(
            ACCESS_TOKEN_URL, json=payload,
            headers={"Content-Type": "application/json"}, timeout=25,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("Could not contact 5paisa access-token service. Check deployment/network and retry.") from exc
    except ValueError as exc:
        raise RuntimeError("5paisa returned an unreadable authentication response.") from exc

    body = result.get("body") or result.get("Body") or {}
    token = body.get("AccessToken") or body.get("accessToken")
    status = body.get("Status", body.get("status", result.get("Status")))
    if not token or (status not in (None, 0, "0", "Success", "success")):
        description = body.get("StatusDescription") or body.get("Message") or result.get("message") or "Authentication was rejected."
        raise RuntimeError("5paisa login failed: " + str(description))
    client_code = str(body.get("ClientCode") or body.get("clientCode") or "")
    return str(token), client_code


def _normalise_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper().replace("_EQ", ""))


def _decode_scrip_master_response(content: bytes) -> str:
    """Decode plain CSV and common compressed CSV responses from the provider."""
    if not content:
        raise RuntimeError("5paisa scrip master response was empty.")
    if content.startswith(b"\x1f\x8b"):
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            raise RuntimeError("5paisa scrip master returned invalid gzip data.") from exc
    elif content.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not candidates:
                    raise RuntimeError("5paisa scrip master ZIP did not contain a CSV file.")
                content = archive.read(candidates[0])
        except zipfile.BadZipFile as exc:
            raise RuntimeError("5paisa scrip master returned an invalid ZIP file.") from exc
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252", errors="replace")
    if text.lstrip().startswith("<"):
        raise RuntimeError("5paisa scrip master returned HTML instead of CSV; check provider availability or endpoint access.")
    return text


def parse_scrip_master(csv_text: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise RuntimeError("Unable to parse 5paisa NSE equity scrip master as CSV.") from exc
    frame.columns = [str(col).strip().lstrip("\ufeff") for col in frame.columns]
    required = {"ScripCode", "ScripData"}
    if not required.issubset(frame.columns):
        found = ", ".join(frame.columns.astype(str)[:12])
        raise RuntimeError(
            "5paisa scrip master format changed; expected ScripCode and ScripData columns. "
            f"Received columns: {found or 'none'}."
        )
    if "Exch" in frame.columns:
        frame = frame[frame["Exch"].astype(str).str.upper() == "N"]
    if "ExchType" in frame.columns:
        frame = frame[frame["ExchType"].astype(str).str.upper() == "C"]
    frame = frame.copy()
    frame["symbol_key"] = frame["ScripData"].astype(str).map(_normalise_symbol)
    frame["ScripCode"] = pd.to_numeric(frame["ScripCode"], errors="coerce")
    frame = frame.dropna(subset=["ScripCode"])
    frame = frame[frame["symbol_key"] != ""]
    if frame.empty:
        raise RuntimeError("5paisa scrip master downloaded, but it contained no usable NSE cash-equity rows.")
    conflicts = frame.groupby("symbol_key")["ScripCode"].nunique()
    conflicting = conflicts[conflicts > 1].index.tolist()
    if conflicting:
        raise RuntimeError(
            "5paisa scrip master contains conflicting NSE cash codes for: "
            + ", ".join(conflicting[:10])
        )
    return frame.drop_duplicates("symbol_key")



def resolve_scrip_codes_via_market_feed(
    access_token: str, api_key: str, client_code: str, symbols: list[str],
) -> dict[str, int]:
    """Resolve NSE cash ScripCodes without assuming response-array ordering.

    Prefer the response's Symbol field when supplied. If Symbol is absent from
    any row, verify the returned Token set against the official scrip master.
    Never pair anonymous response rows with requested symbols by array index.
    """
    cleaned = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    if not cleaned:
        return {}
    if not access_token or not api_key or not client_code:
        raise ValueError("5paisa session, API key and client code are required to resolve stock codes.")
    payload = {
        "head": {"key": api_key},
        "body": {
            "ClientCode": client_code,
            "MarketFeedData": [
                {"Exch": "N", "ExchType": "C", "ScripCode": "0", "ScripData": f"{symbol}_EQ"}
                for symbol in cleaned
            ],
            "LastRequestTime": "/Date(0)/",
            "RefreshRate": "H",
        },
    }
    try:
        response = requests.post(
            MARKET_FEED_URL, json=payload,
            headers={
                "Authorization": "Bearer " + access_token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("Could not resolve 5paisa stock codes through the authenticated MarketFeed API.") from exc
    except ValueError as exc:
        raise RuntimeError("5paisa MarketFeed returned an unreadable symbol-resolution response.") from exc

    body = result.get("body") or result.get("Body") or {}
    head = result.get("head") or result.get("Head") or {}
    head_status = head.get("status", head.get("Status"))
    status = body.get("Status", body.get("status", head_status))
    message = body.get("Message") or body.get("message") or head.get("statusDescription") or head.get("StatusDescription")
    if status not in (None, 0, "0", "Success", "success"):
        raise RuntimeError("5paisa MarketFeed could not resolve stock codes: " + str(message or status))
    rows = body.get("Data") or body.get("data") or []
    if not isinstance(rows, list) or len(rows) != len(cleaned):
        raise RuntimeError(
            "5paisa MarketFeed returned an unexpected number of instruments "
            f"(requested {len(cleaned)}, received {len(rows) if isinstance(rows, list) else 'invalid'}). "
            "No partial subscription was started."
        )

    parsed_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("5paisa MarketFeed returned an invalid instrument row.")
        exch = row.get("Exch", row.get("exch"))
        exch_type = row.get("ExchType", row.get("exchType"))
        if exch is not None and str(exch).upper() != "N":
            raise RuntimeError("5paisa MarketFeed returned a non-NSE instrument.")
        if exch_type is not None and str(exch_type).upper() != "C":
            raise RuntimeError("5paisa MarketFeed returned a non-cash instrument.")
        token = row.get("Token") or row.get("token") or row.get("ScripCode") or row.get("scripCode")
        try:
            code = int(token)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("5paisa MarketFeed did not return a valid NSE cash ScripCode.") from None
        if code <= 0:
            raise RuntimeError("5paisa MarketFeed returned a non-positive NSE cash ScripCode.")
        symbol_value = row.get("Symbol") or row.get("symbol")
        parsed_rows.append((str(symbol_value).strip() if symbol_value is not None else "", code))

    returned_codes = [code for _, code in parsed_rows]
    if len(set(returned_codes)) != len(returned_codes):
        raise RuntimeError("5paisa MarketFeed returned duplicate ScripCodes; refusing ambiguous symbol mapping.")

    if all(symbol for symbol, _ in parsed_rows):
        requested_by_key = {_normalise_symbol(symbol): symbol for symbol in cleaned}
        resolved = {}
        for symbol, code in parsed_rows:
            key = _normalise_symbol(symbol)
            if key not in requested_by_key:
                raise RuntimeError(f"5paisa MarketFeed returned an unexpected symbol: {symbol}.")
            requested_symbol = requested_by_key[key]
            if requested_symbol in resolved:
                raise RuntimeError(f"5paisa MarketFeed returned duplicate rows for {requested_symbol}.")
            resolved[requested_symbol] = code
        missing = [symbol for symbol in cleaned if symbol not in resolved]
        if missing:
            raise RuntimeError("5paisa MarketFeed did not resolve: " + ", ".join(missing))
        return {symbol: resolved[symbol] for symbol in cleaned}

    # Token-only or mixed-identity responses cannot be mapped by row position.
    # Verify the complete token set against the official symbol-keyed master.
    master = fetch_scrip_master()
    expected = {symbol: resolve_scrip_code(master, symbol) for symbol in cleaned}
    expected_codes = list(expected.values())
    if len(set(expected_codes)) != len(expected_codes):
        raise RuntimeError("5paisa scrip master maps requested symbols to duplicate codes; refusing ambiguous mapping.")
    requested_by_key = {_normalise_symbol(symbol): symbol for symbol in cleaned}
    if len(requested_by_key) != len(cleaned):
        raise RuntimeError("Requested symbols collide after normalization; refusing ambiguous mapping.")
    for returned_symbol, code in parsed_rows:
        if returned_symbol:
            key = _normalise_symbol(returned_symbol)
            requested_symbol = requested_by_key.get(key)
            if requested_symbol is None or expected[requested_symbol] != code:
                raise RuntimeError(
                    f"5paisa MarketFeed symbol/token identity mismatch for {returned_symbol}."
                )
    if set(returned_codes) != set(expected_codes):
        raise RuntimeError(
            "5paisa MarketFeed token set does not match the official NSE cash scrip master. "
            "No symbol mapping was accepted."
        )
    return expected

def fetch_scrip_master() -> pd.DataFrame:
    try:
        response = requests.get(
            SCRIP_MASTER_URL, timeout=30,
            headers={"Accept": "text/csv, application/zip, application/gzip, */*"},
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the 5paisa scrip-master service ({type(exc).__name__}). "
            "Check Streamlit deployment network access and retry."
        ) from exc
    if not response.ok:
        content_type = response.headers.get("Content-Type", "unknown")
        detail = (getattr(response, "text", "") or "").strip().replace("\n", " ")[:240]
        raise RuntimeError(
            f"5paisa scrip-master service returned HTTP {response.status_code} "
            f"(content type: {content_type}). {detail or 'No response details provided.'}"
        )
    text = _decode_scrip_master_response(response.content)
    return parse_scrip_master(text)


def resolve_scrip_code(master: pd.DataFrame, symbol: str) -> int:
    key = _normalise_symbol(symbol)
    matches = master.loc[master["symbol_key"] == key, "ScripCode"]
    if matches.empty:
        raise ValueError(f"No NSE cash scrip code found in 5paisa master for {symbol}.")
    return int(matches.iloc[0])


def parse_historical_candles(payload: dict) -> pd.DataFrame:
    body = payload.get("data") or payload.get("Data") or payload.get("body") or {}
    candles = body.get("candles") or body.get("Candles") or []
    if not candles:
        head = payload.get("head") or {}
        description = head.get("Status_description") or head.get("StatusDescription") or payload.get("message")
        raise RuntimeError("5paisa returned no candles. " + str(description or "Check token, scrip code, date range and market-data permissions."))
    frame = pd.DataFrame(candles)
    if frame.shape[1] < 6:
        raise RuntimeError("5paisa candle response did not contain timestamp, OHLC and volume.")
    frame = frame.iloc[:, :6]
    frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def fetch_historical_candles(
    access_token: str, scrip_code: int, interval: str, start_date, end_date,
    client_code: str = "",
) -> pd.DataFrame:
    """Fetch historical candles using the Xstream historical-candles endpoint.

    The official 5paisa Python SDK calls this endpoint with GET and a Bearer
    token. Never include credentials or tokens in raised error messages.
    """
    if interval not in {"5m", "15m"}:
        raise ValueError("Only 5m and 15m intervals are enabled in this build.")
    if not access_token:
        raise ValueError("5paisa session is missing. Authenticate again from the sidebar.")
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("Start date must be on or before end date.")
    if (end - start).days > 183:
        raise ValueError("5paisa intraday historical requests are limited to six months per request.")
    url = HISTORICAL_URL.format(scrip_code=int(scrip_code), interval=interval)
    try:
        # The provider documentation does not clearly state whether `end` is
        # inclusive. Request one extra calendar day, then enforce the user's
        # inclusive date range locally. This works with either provider
        # interpretation and prevents next-day candles leaking into the export.
        request_end = end + pd.Timedelta(days=1)
        response = requests.get(
            url,
            params={"from": start.isoformat(), "end": request_end.isoformat()},
            headers={
                "Authorization": "Bearer " + access_token,
                "Accept": "application/json",
            },
            timeout=30,
        )
        if not response.ok:
            # Keep provider diagnostics useful without ever echoing request headers.
            detail = (response.text or "").strip().replace("\n", " ")[:400]
            raise RuntimeError(
                f"5paisa historical API returned HTTP {response.status_code} "
                f"for {interval}. {detail or 'No response details provided.'}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("5paisa historical API returned a non-JSON response.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the 5paisa historical API for scrip {scrip_code} ({interval}). "
            "Check Streamlit network access and retry."
        ) from exc
    frame = parse_historical_candles(payload)
    lower_bound = pd.Timestamp(start)
    upper_bound = pd.Timestamp(end) + pd.Timedelta(days=1)
    return frame.loc[
        (frame["timestamp"] >= lower_bound) & (frame["timestamp"] < upper_bound)
    ].reset_index(drop=True)
