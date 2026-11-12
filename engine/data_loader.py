"""CSV ingestion and OHLCV validation for Chart Reader."""
from __future__ import annotations

from io import BytesIO
import pandas as pd

ALIASES = {
    "date": "timestamp", "datetime": "timestamp", "time": "timestamp",
    "ts": "timestamp", "timestamp": "timestamp",
    "o": "open", "open": "open", "h": "high", "high": "high",
    "l": "low", "low": "low", "c": "close", "close": "close",
    "v": "volume", "vol": "volume", "volume": "volume",
}


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, chronologically sorted OHLCV frame or raise ValueError."""
    if frame is None or frame.empty:
        raise ValueError("The uploaded file contains no rows.")

    df = frame.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {c: ALIASES[c] for c in df.columns if c in ALIASES}
    df = df.rename(columns=rename)

    # If a broker export uses a separate date and time column, combine them.
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # Keep data usable for analysis, but make the missing time axis explicit.
        df["timestamp"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    for col in required + (["volume"] if "volume" in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.dropna(subset=required)
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        df = df.sort_index()

    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    invalid = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[["open", "high", "low", "close"]].le(0).any(axis=1))
        | (df["volume"] < 0)
    )
    df = df.loc[~invalid].reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Fewer than 3 valid candles remain after data cleaning.")
    return df


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit UploadedFile without assuming broker-specific headers."""
    try:
        raw = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    return normalize_ohlcv(raw)
