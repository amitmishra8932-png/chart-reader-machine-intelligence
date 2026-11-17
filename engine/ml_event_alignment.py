"""Phase 8.11: reproducible, leakage-safe 5m/15m event alignment.

Timestamps are treated as candle start times (inferred from supplied exports).
Confirm provider semantics before using live data. No model fitting or signals.
"""
from __future__ import annotations

from typing import Iterable
import pandas as pd

KEY = ["symbol", "timeframe", "timestamp"]
OHLCV = ["open", "high", "low", "close", "volume"]


def _normalise(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{label} dataset is empty.")
    required = set(KEY + OHLCV)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} dataset is missing columns: {', '.join(missing)}")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["timeframe"] = out["timeframe"].astype(str).str.strip().str.lower()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[KEY].isna().any().any():
        raise ValueError(f"{label} dataset has missing/invalid key values.")
    if out.duplicated(KEY).any():
        raise ValueError(f"{label} dataset contains duplicate symbol/timeframe/timestamp keys.")
    if not out["timeframe"].isin(["5m", "15m"]).all():
        raise ValueError(f"{label} dataset contains unsupported timeframes.")
    if out[OHLCV].isna().any().any():
        raise ValueError(f"{label} dataset contains missing/non-numeric OHLCV values.")
    bad = (
        (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (out["volume"] < 0)
        | (out["high"] < out[["open", "close", "low"]].max(axis=1))
        | (out["low"] > out[["open", "close", "high"]].min(axis=1))
    )
    if bad.any():
        raise ValueError(f"{label} dataset contains {int(bad.sum())} invalid OHLCV rows.")
    return out.sort_values(KEY).reset_index(drop=True)


def prepare_phase_8_11(
    historical: pd.DataFrame,
    regular: pd.DataFrame,
    *,
    exclude_dates: Iterable[str] = ("2026-11-17",),
    max_context_age_minutes: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return event rows, historical-only quarantine and an audit manifest.

    The regular export is authoritative for eligible signal candles. Historical
    rows absent from that export are quarantined. For each 5m bar, only the
    latest same-symbol/same-session 15m bar whose end time is <= the 5m bar's
    start time is joined. Missing bars are never synthesized.
    """
    hist = _normalise(historical, "Historical")
    reg = _normalise(regular, "Regular-signal")
    if not {"5m", "15m"}.issubset(set(reg["timeframe"])):
        raise ValueError("Regular-signal export must contain both 5m and 15m rows.")
    reg_key = reg[KEY].copy()
    hist_key = hist[KEY].copy()
    reg_idx = pd.MultiIndex.from_frame(reg_key)
    hist_idx = pd.MultiIndex.from_frame(hist_key)
    only_mask = ~hist_idx.isin(reg_idx)
    historical_only = hist.loc[only_mask].copy()
    historical_only["quarantine_reason"] = "not_present_in_regular_signal_export"

    eligible = reg.copy()
    if "regular_signal_eligible" in eligible:
        eligible = eligible[eligible["regular_signal_eligible"].astype(str).str.lower().isin(["true", "1"])]
    excluded_dates = {pd.Timestamp(d).date() for d in exclude_dates}
    eligible = eligible[~eligible["timestamp"].dt.date.isin(excluded_dates)].copy()
    eligible["session_date"] = eligible["timestamp"].dt.date.astype(str)

    five = eligible[eligible["timeframe"] == "5m"].copy()
    fifteen = eligible[eligible["timeframe"] == "15m"].copy()
    if five.empty or fifteen.empty:
        raise ValueError("Eligible regular-session data must contain both 5m and 15m rows.")

    five = five.sort_values(["symbol", "session_date", "timestamp"]).reset_index(drop=True)
    fifteen = fifteen.sort_values(["symbol", "session_date", "timestamp"]).copy()
    fifteen["htf_available_at"] = fifteen["timestamp"] + pd.Timedelta(minutes=15)
    # merge_asof by symbol and date prevents cross-symbol/session look-ahead.
    right = fifteen[["symbol", "session_date", "timestamp", "htf_available_at"] + OHLCV].rename(
        columns={"timestamp": "htf_bar_start", **{c: f"htf_{c}" for c in OHLCV}}
    ).sort_values(["htf_available_at", "symbol", "session_date"])
    left = five.sort_values(["timestamp", "symbol", "session_date"])
    events = pd.merge_asof(
        left, right,
        left_on="timestamp", right_on="htf_available_at",
        by=["symbol", "session_date"], direction="backward",
        allow_exact_matches=True,
    )
    events["htf_age_minutes"] = (
        (events["timestamp"] - events["htf_available_at"]).dt.total_seconds() / 60
    )
    events["htf_context_available"] = events["htf_bar_start"].notna()
    events["htf_context_fresh"] = (
        events["htf_context_available"]
        & events["htf_age_minutes"].between(0, max_context_age_minutes)
    )
    # The 15m candle must be closed at the event timestamp.
    unsafe = events["htf_context_available"] & (events["htf_available_at"] > events["timestamp"])
    if unsafe.any():
        raise AssertionError("Future 15m context detected in event alignment.")
    events = events.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    manifest = {
        "phase": "8.11_event_level_alignment",
        "timestamp_convention": "candle_start_inferred_provider_confirmation_required",
        "historical_rows": int(len(hist)),
        "regular_signal_rows": int(len(reg)),
        "historical_only_rows_quarantined": int(len(historical_only)),
        "eligible_5m_event_rows": int(len(events)),
        "events_with_closed_15m_context": int(events["htf_context_available"].sum()),
        "events_with_fresh_15m_context": int(events["htf_context_fresh"].sum()),
        "symbols": int(events["symbol"].nunique()),
        "date_range": [
            events["timestamp"].min().isoformat(),
            events["timestamp"].max().isoformat(),
        ],
        "excluded_dates": sorted(str(d) for d in excluded_dates),
        "max_context_age_minutes": int(max_context_age_minutes),
        "guardrails": [
            "Only regular-signal export rows marked eligible are used as events.",
            "Historical-only rows are quarantined and never silently merged.",
            "15m context is same-symbol and same-session; available_at = start + 15 minutes.",
            "Only 15m bars closed at or before the 5m event start are joined.",
            "Stale context is flagged; no synthetic candles are inserted.",
            "No model fitting, test-label access, trading signals or order placement.",
            "Timestamp start-time convention is inferred and requires provider confirmation.",
        ],
    }
    return events, historical_only, manifest
