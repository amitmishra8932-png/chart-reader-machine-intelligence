"""Leakage-aware historical data and engineered-feature audits."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OHLC = ("open", "high", "low", "close")


def audit_ohlcv_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return reproducible structural checks without silently repairing input."""
    required = {"timestamp", *OHLC, "volume", "symbol", "timeframe"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")

    data = frame.copy()
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    numeric = data.loc[:, list(OHLC)].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")

    invalid_ohlc = (
        numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
    ) | (
        numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    ) | numeric[list(OHLC)].le(0).any(axis=1)

    normalized_keys = data.loc[:, ["symbol", "timeframe"]].copy()
    normalized_keys["timestamp"] = timestamps
    duplicate_keys = normalized_keys.duplicated(
        ["symbol", "timeframe", "timestamp"], keep=False
    )
    rows_by_group = (
        data.groupby(["symbol", "timeframe"], dropna=False)
        .size().rename("rows").reset_index()
    )
    rows_by_group["rows"] = rows_by_group["rows"].astype(int)

    per_group = []
    work = data.assign(__parsed_timestamp=timestamps)
    for (symbol, timeframe), group in work.groupby(
        ["symbol", "timeframe"], dropna=False, sort=True
    ):
        valid_times = group["__parsed_timestamp"].dropna().sort_values()
        per_group.append({
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "rows": int(len(group)),
            "unique_sessions": int(valid_times.dt.date.nunique()),
            "invalid_timestamps": int(group["__parsed_timestamp"].isna().sum()),
            "duplicate_timestamps": int(group.duplicated(
                ["__parsed_timestamp"], keep=False
            ).sum()),
            "first_timestamp": (
                valid_times.iloc[0].isoformat() if len(valid_times) else None
            ),
            "last_timestamp": (
                valid_times.iloc[-1].isoformat() if len(valid_times) else None
            ),
        })

    result: dict[str, Any] = {
        "row_count": int(len(data)),
        "symbol_count": int(data["symbol"].nunique(dropna=True)),
        "symbols": sorted(data["symbol"].dropna().astype(str).unique().tolist()),
        "timeframes": sorted(data["timeframe"].dropna().astype(str).unique().tolist()),
        "first_timestamp": (
            timestamps.min().isoformat() if timestamps.notna().any() else None
        ),
        "last_timestamp": (
            timestamps.max().isoformat() if timestamps.notna().any() else None
        ),
        "missing_values": {
            str(key): int(value) for key, value in data.isna().sum().items()
        },
        "duplicate_symbol_timeframe_timestamps": int(duplicate_keys.sum()),
        "invalid_timestamp_rows": int(timestamps.isna().sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "negative_volume_rows": int((volume < 0).sum()),
        "non_numeric_volume_rows": int(volume.isna().sum()),
        "rows_by_symbol_timeframe": per_group,
    }

    # Keep supplied audit flags visible; do not infer regular-session eligibility
    # when the raw source does not contain these fields.
    for flag in ("regular_signal_eligible", "session_eligible", "research_only"):
        if flag in data.columns:
            result[f"{flag}_counts"] = {
                str(key): int(value)
                for key, value in data[flag].fillna("MISSING").astype(str).value_counts().items()
            }
    if "session_classification" in data.columns:
        result["session_classification_counts"] = {
            str(key): int(value)
            for key, value in data["session_classification"].fillna("MISSING").astype(str).value_counts().items()
        }
    return result


def feature_distribution_report(
    frame: pd.DataFrame, feature_columns: list[str] | None = None
) -> dict[str, dict[str, float | int]]:
    """Summarise numeric feature availability and robust distribution quantiles."""
    columns = feature_columns or [
        column for column in frame.columns
        if column.startswith("feature_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    report: dict[str, dict[str, float | int]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        report[column] = {
            "rows": int(len(values)),
            "valid": int(len(finite)),
            "missing_or_nonfinite": int(len(values) - len(finite)),
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std()) if len(finite) > 1 else None,
            "min": float(finite.min()) if len(finite) else None,
            "p01": float(finite.quantile(0.01)) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "p99": float(finite.quantile(0.99)) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    return report


def chronological_date_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Assign whole calendar dates to chronological train/validation/test sets."""
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp column is required for chronological splitting")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must each be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must be below 1")

    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("Invalid timestamps must be resolved before date splitting")
    dates = pd.Index(parsed.dt.strftime("%Y-%m-%d").unique()).sort_values()
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required")

    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(
        len(dates) * (train_fraction + validation_fraction)
    ))
    validation_end = min(validation_end, len(dates) - 1)

    assignments = {
        **{date: "train" for date in dates[:train_end]},
        **{date: "validation" for date in dates[train_end:validation_end]},
        **{date: "test" for date in dates[validation_end:]},
    }
    result = frame.copy()
    result["validation_split"] = parsed.dt.strftime("%Y-%m-%d").map(assignments)
    return result


def build_feature_validation_report(
    frame: pd.DataFrame, feature_frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Combine source audit, date split coverage and feature distributions."""
    audited = audit_ohlcv_dataset(frame)
    if feature_frame is None:
        from engine.features import add_features
        feature_frame = add_features(frame)

    split = chronological_date_split(frame)
    dates = pd.to_datetime(split["timestamp"]).dt.strftime("%Y-%m-%d")
    split_summary = {}
    for name, group in split.groupby("validation_split", sort=False):
        group_dates = pd.to_datetime(group["timestamp"]).dt.strftime("%Y-%m-%d")
        split_summary[str(name)] = {
            "rows": int(len(group)),
            "sessions": int(group_dates.nunique()),
            "first_date": str(group_dates.min()),
            "last_date": str(group_dates.max()),
        }
    return {
        "dataset_audit": audited,
        "chronological_split": split_summary,
        "feature_distributions": feature_distribution_report(feature_frame),
        "limitations": [
            "This report describes data and feature distributions; it is not a predictive-performance result.",
            "No session eligibility is inferred when the source does not provide session flags.",
            "No multi-timeframe context is joined until candle timestamp-label semantics are explicitly supplied.",
        ],
    }
