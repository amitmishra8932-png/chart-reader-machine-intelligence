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
SCRIP_MASTER_URL = "https://openapi.5paisa.com/VendorsAPI/Service1.svc/ScripMaster/segment/nse_eq"
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
    if content.startswith(b"\\x1f\\x8b"):
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            raise RuntimeError("5paisa scrip master returned invalid gzip data.") from exc
    elif content.startswith(b"PK\\x03\\x04"):
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
        frame = pd.read_csv(StringIO(csv_text), sep=None, engine="python")
    except Exception as exc:
        raise RuntimeError("Unable to parse 5paisa NSE equity scrip master as CSV.") from exc
    frame.columns = [str(col).strip().lstrip("\\ufeff") for col in frame.columns]
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
    return frame.drop_duplicates("symbol_key")


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
        detail = (getattr(response, "text", "") or "").strip().replace("\\n", " ")[:240]
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
        response = requests.get(
            url,
            params={"from": start.isoformat(), "end": end.isoformat()},
            headers={
                "Authorization": "Bearer " + access_token,
                "Accept": "application/json",
            },
            timeout=30,
        )
        if not response.ok:
            # Keep provider diagnostics useful without ever echoing request headers.
            detail = (response.text or "").strip().replace("\\n", " ")[:400]
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
    return parse_historical_candles(payload)
