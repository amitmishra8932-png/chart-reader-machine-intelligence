"""Phase 8.13: strict clean-candle selection and chronological ML dataset prep.

This stage prepares an auditable research dataset only. It does not fit models,
impute feature values, tune thresholds, or create trading signals.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.ml_dataset import build_ml_dataset
from engine.ml_data_integrity import EXPECTED


def prepare_phase_8_13(
    candidates: pd.DataFrame,
    *,
    neutral_bands_pct: dict[int, float],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare only complete, eligible 5m sessions and assign date splits.

    The input is the Phase 8.12 row-level candidate export. 15m context,
    incomplete sessions, off-grid rows, weekend rows and quarantined history
    are intentionally excluded. Feature warm-up NaNs and unlabeled session
    tails are retained and reported, not imputed or silently dropped.
    """
    required = {
        "timestamp", "symbol", "timeframe", "open", "high", "low", "close",
        "volume", "session_quality", "on_expected_grid", "is_weekend",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Phase 8.12 candidate data missing columns: {missing}")
    if candidates.empty:
        raise ValueError("Phase 8.12 candidate dataset is empty.")
    if not neutral_bands_pct:
        raise ValueError("Explicit neutral bands are required for every target horizon.")

    data = candidates.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Candidate dataset contains invalid timestamps.")
    if data[["symbol", "timeframe"]].isna().any().any():
        raise ValueError("Candidate symbol/timeframe keys cannot be missing.")
    if data.duplicated(["symbol", "timeframe", "timestamp"]).any():
        raise ValueError("Duplicate candidate candle keys must be resolved first.")

    # Strict selection: no partial sessions or off-grid candles enter features.
    selected = data[
        data["timeframe"].astype(str).str.lower().eq("5m")
        & data["session_quality"].astype(str).eq("complete")
        & data["on_expected_grid"].astype(str).str.lower().isin(["true", "1"])
        & ~data["is_weekend"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    if selected.empty:
        raise ValueError("No complete eligible 5m sessions remain after strict selection.")

    selected["session_date"] = selected["timestamp"].dt.strftime("%Y-%m-%d")
    selected["bar_time"] = selected["timestamp"].dt.strftime("%H:%M")
    expected = set(EXPECTED["5m"])
    session_rows = selected.groupby(["symbol", "session_date"], sort=False)
    bad_sessions = []
    for (symbol, session_date), group in session_rows:
        observed = set(group["bar_time"])
        if len(group) != len(expected) or observed != expected:
            bad_sessions.append(f"{symbol}:{session_date}")
    if bad_sessions:
        raise ValueError(
            "Rows labelled complete do not match the full 5m grid for sessions: "
            + ", ".join(bad_sessions[:10])
        )

    selected = selected.sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    prepared, summary = build_ml_dataset(
        selected,
        neutral_bands_pct=neutral_bands_pct,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    split_dates: dict[str, set[str]] = {}
    for name, group in prepared.groupby("validation_split", sort=False):
        split_dates[str(name)] = set(
            group["timestamp"].dt.strftime("%Y-%m-%d").unique()
        )
    names = ("train", "validation", "test")
    overlap = {
        f"{left}_{right}": sorted(split_dates.get(left, set()) & split_dates.get(right, set()))
        for i, left in enumerate(names)
        for right in names[i + 1:]
    }
    if any(overlap.values()):
        raise AssertionError(f"Calendar-date leakage across splits: {overlap}")

    target_columns = [
        c for c in prepared.columns
        if c.startswith("target_")
    ]
    feature_columns = summary["feature_columns"]
    summary.update({
        "phase": "8.13_chronological_research_dataset",
        "input_candidate_rows": int(len(data)),
        "selected_complete_5m_rows": int(len(selected)),
        "excluded_rows": int(len(data) - len(selected)),
        "selected_symbols": sorted(selected["symbol"].astype(str).unique().tolist()),
        "selected_sessions": int(selected[["symbol", "session_date"]].drop_duplicates().shape[0]),
        "expected_bars_per_selected_session": len(expected),
        "split_date_overlap": overlap,
        "feature_missing_or_nonfinite": {
            column: int((~pd.to_numeric(prepared[column], errors="coerce").map(
                lambda value: pd.notna(value) and value not in (float("inf"), float("-inf"))
            )).sum())
            for column in feature_columns
        },
        "unlabelled_target_rows": {
            column: int(prepared[column].isna().sum()) for column in target_columns
        },
        "guardrails": [
            "Only complete, eligible, on-grid 5m sessions enter this dataset.",
            "15m rows are not mixed into the 5m event training table.",
            "Incomplete, weekend, off-grid and quarantined rows remain excluded.",
            "Feature calculations are causal; no future rows are used to construct features.",
            "Forward targets are session-bounded and use explicit supplied neutral bands.",
            "Whole calendar dates are assigned to exactly one chronological split.",
            "Feature warm-up NaNs and unlabelled session tails are retained; no imputation is performed.",
            "No model is fitted and no trading signals or orders are produced.",
            "Timestamp convention remains inferred pending provider confirmation.",
        ],
        "limitations": summary.get("limitations", []) + [
            "Complete-session selection can reduce sample size and may introduce selection bias.",
            "Neutral bands are research parameters, not calibrated costs or proven decision thresholds.",
            "This output is not yet model-ready: preprocessing and fitting must be learned on training data only.",
        ],
    })
    return prepared, summary
