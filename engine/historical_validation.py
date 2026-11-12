"""Historical OHLCV quality checks for leakage-safe research datasets."""
from __future__ import annotations

import pandas as pd


def audit_historical_ohlcv(frame: pd.DataFrame, interval: str = "5m") -> dict:
    """Summarise data integrity without silently modifying the supplied candles.

    Missing-bar estimates are limited to gaps within the same calendar date;
    overnight, weekend and holiday closures are not counted as missing candles.
    """
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    if interval not in interval_minutes:
        raise ValueError("Unsupported interval. Use 1m, 3m, 5m, 15m, 30m or 1h.")
    if frame is None or frame.empty:
        raise ValueError("Historical dataset is empty.")

    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = frame.copy()
    original_count = len(df)
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    timestamp_invalid = int(timestamps.isna().sum())
    valid_ts = timestamps.dropna()
    duplicate_timestamps = int(valid_ts.duplicated(keep="last").sum())
    out_of_order = int((valid_ts.diff().dropna() < pd.Timedelta(0)).sum())

    numeric = {}
    for col in ("open", "high", "low", "close"):
        numeric[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_numeric = pd.Series(False, index=df.index)
    for values in numeric.values():
        invalid_numeric |= values.isna()
    invalid_ohlc = (
        (numeric["open"] <= 0) | (numeric["high"] <= 0)
        | (numeric["low"] <= 0) | (numeric["close"] <= 0)
        | (numeric["high"] < pd.concat([numeric["open"], numeric["close"], numeric["low"]], axis=1).max(axis=1))
        | (numeric["low"] > pd.concat([numeric["open"], numeric["close"], numeric["high"]], axis=1).min(axis=1))
    ).fillna(True)
    invalid_ohlc_count = int((invalid_numeric | invalid_ohlc).sum())

    negative_volume = 0
    zero_volume = None
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        negative_volume = int((volume < 0).sum())
        zero_volume = int((volume.fillna(0) == 0).sum())

    ordered = pd.DataFrame({"timestamp": timestamps}).dropna().drop_duplicates("timestamp").sort_values("timestamp")
    same_day = ordered["timestamp"].dt.date.eq(ordered["timestamp"].shift().dt.date)
    deltas = ordered["timestamp"].diff()
    expected = pd.Timedelta(minutes=interval_minutes[interval])
    gap_deltas = deltas[same_day & (deltas > expected)]
    estimated_missing_bars = int(sum(max(int(delta // expected) - 1, 1) for delta in gap_deltas))

    clean_rows = original_count - invalid_ohlc_count - timestamp_invalid - duplicate_timestamps
    warnings = []
    if timestamp_invalid:
        warnings.append("Some timestamps could not be parsed.")
    if duplicate_timestamps:
        warnings.append("Duplicate timestamps are present.")
    if out_of_order:
        warnings.append("Input rows are not in chronological order.")
    if invalid_ohlc_count:
        warnings.append("Some OHLC rows contain invalid or non-positive prices.")
    if negative_volume:
        warnings.append("Negative volume values are present.")
    if estimated_missing_bars:
        warnings.append("Possible missing intraday candles detected; verify against exchange sessions.")

    return {
        "rows": original_count,
        "valid_timestamp_rows": int(valid_ts.size),
        "invalid_timestamps": timestamp_invalid,
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order_steps": out_of_order,
        "invalid_ohlc_rows": invalid_ohlc_count,
        "negative_volume_rows": negative_volume,
        "zero_volume_rows": zero_volume,
        "estimated_missing_intraday_bars": estimated_missing_bars,
        "first_timestamp": valid_ts.min().isoformat() if not valid_ts.empty else None,
        "last_timestamp": valid_ts.max().isoformat() if not valid_ts.empty else None,
        "distinct_trading_dates": int(valid_ts.dt.date.nunique()),
        "interval": interval,
        "warnings": warnings,
        "quality_status": "Review required" if warnings else "No basic integrity issues detected",
    }


def audit_nse_session_integrity(
    frame: pd.DataFrame,
    interval: str = "5m",
    holidays: set | None = None,
) -> dict:
    """Flag NSE cash-session timestamp anomalies without changing source rows.

    Assumes candle timestamps represent bar start times: 09:15 is the first
    regular-session candle and the last permitted start is 15:25 for 5m or
    15:15 for 15m. Holiday dates may be supplied by the caller; no static
    holiday calendar is embedded because exchange holidays vary by year.
    """
    interval_minutes = {"5m": 5, "15m": 15}
    if interval not in interval_minutes:
        raise ValueError("Session audit supports 5m and 15m intervals only.")
    if frame is None or frame.empty:
        raise ValueError("Historical dataset is empty.")
    if "timestamp" not in frame.columns:
        raise ValueError("Missing required columns: timestamp")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    valid = timestamps.dropna()
    holiday_set = set(holidays or set())
    weekend_mask = valid.dt.weekday >= 5
    holiday_mask = valid.dt.date.isin(holiday_set)
    minutes = valid.dt.hour * 60 + valid.dt.minute
    open_minutes = 9 * 60 + 15
    close_minutes = 15 * 60 + 30
    interval_size = interval_minutes[interval]
    outside_session = (minutes < open_minutes) | (minutes >= close_minutes)
    misaligned = ((minutes - open_minutes) % interval_size) != 0
    # A start-time candle must finish by the regular 15:30 close.
    too_late_start = minutes + interval_size > close_minutes
    return {
        "rows_checked": int(len(frame)),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "weekend_timestamps": int(weekend_mask.sum()),
        "known_holiday_timestamps": int(holiday_mask.sum()),
        "outside_session_timestamps": int(outside_session.sum()),
        "misaligned_timestamps": int(misaligned.sum()),
        "too_late_candle_starts": int(too_late_start.sum()),
        "session_anomaly_rows": int((weekend_mask | holiday_mask | outside_session | misaligned | too_late_start).sum()),
        "timestamp_convention": "Assumed candle start time; confirm with provider documentation.",
        "warnings": (
            ["Weekend-dated candles found; regular NSE cash session is closed on weekends."]
            if weekend_mask.any() else []
        ) + (
            ["Candles fall on supplied NSE holiday dates."]
            if holiday_mask.any() else []
        ) + (
            ["Candles fall outside the regular NSE cash session (09:15–15:30)."]
            if outside_session.any() else []
        ) + (
            ["Candle timestamps do not align to the selected interval from 09:15."]
            if misaligned.any() else []
        ) + (
            ["Candle start time would extend beyond the 15:30 regular-session close."]
            if too_late_start.any() else []
        ),
    }
