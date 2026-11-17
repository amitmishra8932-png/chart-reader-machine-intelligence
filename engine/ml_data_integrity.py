"""Phase 8.12: explicit session coverage and quarantine classification.

The expected NSE regular-session grids are configuration assumptions, not proof
of provider timestamp semantics. This module flags data; it never repairs it.
"""
from __future__ import annotations

from typing import Iterable
import pandas as pd

EXPECTED = {
    "5m": pd.date_range("09:15", "15:25", freq="5min").strftime("%H:%M").tolist(),
    "15m": pd.date_range("09:15", "15:15", freq="15min").strftime("%H:%M").tolist(),
}
EXPECTED_SETS = {timeframe: set(times) for timeframe, times in EXPECTED.items()}
KEY = ["symbol", "timeframe", "timestamp"]
OHLC = ["open", "high", "low", "close"]


def _validate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = set(KEY + OHLC + ["volume"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} dataset is missing columns: {', '.join(missing)}")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["timeframe"] = out["timeframe"].astype(str).str.strip().str.lower()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out[KEY].isna().any().any():
        raise ValueError(f"{label} dataset has invalid symbol/timeframe/timestamp keys.")
    if not out["timeframe"].isin(EXPECTED).all():
        raise ValueError(f"{label} dataset contains unsupported timeframes.")
    if out.duplicated(KEY).any():
        raise ValueError(f"{label} dataset contains duplicate keys.")
    return out


def audit_session_coverage(
    regular: pd.DataFrame,
    *,
    exclude_dates: Iterable[str] = ("2026-11-17",),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return per-symbol/session coverage, row-level candidate data and manifest.

    Candidate rows are retained; quality labels are joined rather than used to
    silently discard partial sessions. Off-grid rows are flagged separately.
    """
    data = _validate(regular, "Regular-signal")
    if "regular_signal_eligible" in data:
        data = data[data["regular_signal_eligible"].astype(str).str.lower().isin(["true", "1"])].copy()
    excluded = {pd.Timestamp(d).date() for d in exclude_dates}
    data = data[~data["timestamp"].dt.date.isin(excluded)].copy()
    data["session_date"] = data["timestamp"].dt.strftime("%Y-%m-%d")
    data["bar_time"] = data["timestamp"].dt.strftime("%H:%M")
    data["is_weekend"] = data["timestamp"].dt.dayofweek >= 5
    # Avoid DataFrame.apply(axis=1) overhead on large candle datasets.
    data["on_expected_grid"] = [
        bar_time in EXPECTED_SETS[timeframe]
        for bar_time, timeframe in zip(data["bar_time"], data["timeframe"])
    ]

    summaries = []
    for (symbol, timeframe, session_date), group in data.groupby(
        ["symbol", "timeframe", "session_date"], sort=True
    ):
        expected = EXPECTED[timeframe]
        observed = set(group.loc[group["on_expected_grid"], "bar_time"])
        missing = [t for t in expected if t not in observed]
        off_grid = sorted(set(group.loc[~group["on_expected_grid"], "bar_time"]))
        missing_closing = [t for t in missing if t >= ("15:15" if timeframe == "5m" else "15:15")]
        missing_intraday = [t for t in missing if t not in missing_closing]
        weekend = bool(group["is_weekend"].any())
        if weekend:
            quality = "weekend_session_date"
        elif off_grid:
            quality = "off_grid_timestamps_present"
        elif not missing:
            quality = "complete"
        elif missing_intraday and missing_closing:
            quality = "intraday_and_closing_gaps"
        elif missing_intraday:
            quality = "intraday_gaps"
        else:
            quality = "closing_bars_missing_only"
        summaries.append({
            "symbol": symbol, "timeframe": timeframe, "session_date": session_date,
            "observed_rows": int(len(group)), "expected_bars": len(expected),
            "observed_expected_grid_bars": len(observed),
            "missing_bar_count": len(missing), "missing_times": ";".join(missing),
            "off_grid_times": ";".join(off_grid), "is_weekend": weekend,
            "session_quality": quality,
        })
    sessions = pd.DataFrame(summaries)
    if sessions.empty:
        raise ValueError("No eligible rows remain after date exclusions.")
    candidates = data.merge(
        sessions[["symbol", "timeframe", "session_date", "session_quality",
                  "missing_bar_count", "missing_times", "off_grid_times"]],
        on=["symbol", "timeframe", "session_date"], how="left", validate="many_to_one"
    )
    manifest = {
        "phase": "8.12_data_integrity_and_coverage",
        "timestamp_convention": "unverified_provider_confirmation_required",
        "expected_session_grid_assumption": {
            "5m": {"first": "09:15", "last": "15:25", "expected_bars": 75},
            "15m": {"first": "09:15", "last": "15:15", "expected_bars": 25},
        },
        "excluded_dates": sorted(str(d) for d in excluded),
        "candidate_rows": int(len(candidates)),
        "symbols": int(candidates["symbol"].nunique()),
        "session_groups": int(len(sessions)),
        "session_quality_counts": {
            f"{tf}:{quality}": int(count)
            for (tf, quality), count in sessions.groupby(["timeframe", "session_quality"]).size().items()
        },
        "off_grid_rows": int((~candidates["on_expected_grid"]).sum()),
        "weekend_rows": int(candidates["is_weekend"].sum()),
        "guardrails": [
            "No missing candles are synthesized.",
            "No timestamps are shifted or relabelled.",
            "Partial sessions are retained with explicit quality labels.",
            "Off-grid and weekend records are flagged, not automatically repaired.",
            "Expected grids are assumptions pending provider timestamp documentation.",
            "This is a data audit, not model training or evidence of trading edge.",
        ],
    }
    return sessions, candidates, manifest


def classify_quarantine(quarantine: pd.DataFrame) -> pd.DataFrame:
    """Add evidence-based flags to historical-only rows without promoting them."""
    if quarantine.empty:
        return quarantine.copy()
    data = _validate(quarantine, "Quarantine")
    data["bar_time"] = data["timestamp"].dt.strftime("%H:%M")
    data["is_weekend"] = data["timestamp"].dt.dayofweek >= 5
    data["is_1530_timestamp"] = data["bar_time"].eq("15:30")
    data["flat_ohlc"] = (
        data[OHLC].nunique(axis=1).eq(1)
    )
    data["off_expected_grid"] = [
        bar_time not in EXPECTED_SETS[timeframe]
        for bar_time, timeframe in zip(data["bar_time"], data["timeframe"])
    ]
    def reasons(row):
        tags = []
        if row["is_weekend"]:
            tags.append("weekend_date")
        if row["is_1530_timestamp"]:
            tags.append("15:30_timestamp_unverified")
        if row["flat_ohlc"]:
            tags.append("flat_ohlc")
        if row["off_expected_grid"]:
            tags.append("off_expected_session_grid")
        return ";".join(tags) if tags else "historical_only_unclassified"
    data["quarantine_flags"] = data.apply(reasons, axis=1)
    data["promotion_status"] = "remain_quarantined"
    return data
