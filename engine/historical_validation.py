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


# Curated NSE special cash-market sessions. Keep this list explicit and review it
# against official NSE circulars when extending the supported calendar.
DEFAULT_SPECIAL_SESSIONS = {
    "2026-11-08": {
        "start": "14:20",
        "end": "15:40",
        "label": "Diwali Muhurat Trading",
        # The 15-minute provider series begins with a 10-minute opening stub
        # (14:20), then follows the 15-minute grid from 14:30.
        "alignment_starts": {"5m": "14:20", "15m": "14:30"},
        "opening_partial_start": "14:20",
    },
}


def audit_nse_session_integrity(
    frame: pd.DataFrame,
    interval: str = "5m",
    holidays: set | None = None,
    special_sessions: dict | None = None,
    cas_eligible: bool = False,
) -> dict:
    """Classify NSE cash timestamps by session without modifying source rows.

    Session rules are selected by calendar date. For CAS-eligible securities,
    the November 2026 schedule is continuous trading 09:15-15:15, CAS
    15:15-15:35, transition 15:35-15:50, and post-close 15:50-16:00.
    From 2026-12-07, continuous trading runs until 15:30, CAS 15:30-15:45,
    transition 15:45-15:50, and post-close 15:50-16:00. Non-CAS securities
    trade continuously until 15:30, then transition until 15:50 and post-close
    until 16:00. Eligibility is supplied by the caller; this function does
    not guess.

    A dated special session overrides normal weekday/weekend rules. Special
    sessions may define interval-specific alignment starts and an opening
    partial candle. Timestamps are assumed to be candle starts; provider
    timestamp semantics are unverified.
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
    holiday_set = {pd.Timestamp(day).date() for day in (holidays or set())}
    sessions = DEFAULT_SPECIAL_SESSIONS if special_sessions is None else special_sessions
    interval_size = interval_minutes[interval]
    open_minutes = 9 * 60 + 15
    cas2_effective_date = pd.Timestamp("2026-12-07").date()
    closing_session_start = 15 * 60 + 50
    exchange_close = 16 * 60

    weekend_count = holiday_count = special_count = 0
    cas_count = transition_count = closing_count = outside_count = misaligned_count = 0
    anomaly_indices = set()
    classifications = {}

    for index, timestamp in valid.items():
        day = timestamp.date()
        minute = timestamp.hour * 60 + timestamp.minute
        day_key = day.isoformat()
        is_weekend = timestamp.weekday() >= 5
        is_holiday = day in holiday_set
        weekend_count += int(is_weekend)
        holiday_count += int(is_holiday)

        special = sessions.get(day_key)
        if special:
            start_h, start_m = map(int, special["start"].split(":"))
            end_h, end_m = map(int, special["end"].split(":"))
            special_start = start_h * 60 + start_m
            special_end = end_h * 60 + end_m
            if special_start <= minute < special_end:
                classifications[index] = "special_continuous"
                special_count += 1
                alignment_text = special.get("alignment_starts", {}).get(
                    interval, special["start"]
                )
                align_h, align_m = map(int, alignment_text.split(":"))
                alignment_start = align_h * 60 + align_m
                partial_text = special.get("opening_partial_start")
                is_opening_partial = (
                    partial_text is not None
                    and minute == sum(int(part) * factor for part, factor in zip(
                        partial_text.split(":"), (60, 1)
                    ))
                )
                if not is_opening_partial and (minute - alignment_start) % interval_size:
                    misaligned_count += 1
                    anomaly_indices.add(index)
                continue

        if is_weekend or is_holiday:
            classifications[index] = "closed_day"
            anomaly_indices.add(index)
            continue

        cas2_effective = day >= cas2_effective_date
        if cas_eligible and cas2_effective:
            continuous_close = 15 * 60 + 30
            cas_start = 15 * 60 + 30
            cas_end = 15 * 60 + 45
            transition_start = cas_end
        elif cas_eligible:
            continuous_close = 15 * 60 + 15
            cas_start = 15 * 60 + 15
            cas_end = 15 * 60 + 35
            transition_start = cas_end
        else:
            continuous_close = 15 * 60 + 30
            cas_start = cas_end = None
            transition_start = 15 * 60 + 30

        if open_minutes <= minute < continuous_close:
            classifications[index] = "regular"
            if (minute - open_minutes) % interval_size:
                misaligned_count += 1
                anomaly_indices.add(index)
        elif cas_eligible and cas_start <= minute < cas_end:
            classifications[index] = "closing_auction_session"
            cas_count += 1
            if (minute - cas_start) % interval_size:
                misaligned_count += 1
                anomaly_indices.add(index)
        elif transition_start <= minute < closing_session_start:
            classifications[index] = "closing_transition"
            transition_count += 1
        elif closing_session_start <= minute < exchange_close:
            classifications[index] = "closing_session"
            closing_count += 1
            if (minute - closing_session_start) % interval_size:
                misaligned_count += 1
                anomaly_indices.add(index)
        else:
            classifications[index] = "outside_exchange_sessions"
            outside_count += 1
            anomaly_indices.add(index)

    closed_day_count = sum(value == "closed_day" for value in classifications.values())
    return {
        "rows_checked": int(len(frame)),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "weekend_timestamps": weekend_count,
        "known_holiday_timestamps": holiday_count,
        "special_session_timestamps": special_count,
        "cas_eligible": bool(cas_eligible),
        "closing_auction_session_timestamps": cas_count,
        "closing_transition_timestamps": transition_count,
        "closing_auction_transition_timestamps": cas_count + transition_count,
        "closing_session_timestamps": closing_count,
        "outside_session_timestamps": outside_count,
        "misaligned_timestamps": misaligned_count,
        "session_anomaly_rows": len(anomaly_indices),
        "non_regular_session_rows": cas_count + transition_count + closing_count,
        "regular_session_rows": sum(value == "regular" for value in classifications.values()),
        "timestamp_convention": "Assumed candle start time; confirm with provider documentation.",
        "session_classifications": classifications,
        "session_anomaly_indices": sorted(anomaly_indices),
        "session_eligible_indices": sorted(
            index for index, label in classifications.items()
            if label == "regular" and index not in anomaly_indices
        ),
        "warnings": (
            ["Weekend/holiday candles outside a configured special session were found."]
            if closed_day_count else []
        ) + (
            ["Some candles fall in the Closing Auction Session (CAS); keep them separate from continuous intraday signal calculations."]
            if cas_count else []
        ) + (
            ["Some candles fall in the CAS transition period; keep them separate from continuous intraday signal calculations."]
            if transition_count else []
        ) + (
            ["Some candles fall in the 15:50-16:00 closing session; treat them separately from continuous intraday trading."]
            if closing_count else []
        ) + (
            ["Some candles fall outside recognised NSE equity sessions."]
            if outside_count else []
        ) + (
            ["Some candle timestamps do not align to the relevant session start."]
            if misaligned_count else []
        ),
    }
