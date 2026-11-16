"""Reproducible feature/target dataset assembly for Phase 8.5."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.feature_validation import chronological_date_split
from engine.features import add_features
from engine.ml_targets import add_directional_targets
from engine.patterns import detect_patterns


def _validate_ml_input(frame: pd.DataFrame) -> None:
    """Fail closed on malformed OHLCV rows before feature/label generation."""
    required = {"timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required ML dataset columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot prepare an empty dataset.")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Every timestamp must be parseable before ML preparation.")
    for column in ("symbol", "timeframe"):
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot be missing or blank.")

    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must all be finite numeric values.")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (numeric["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    invalid_ohlc = (
        numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
    ) | (
        numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    )
    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC candle geometry must be resolved first.")

    keys = frame[["symbol", "timeframe"]].astype(str).copy()
    keys["timestamp"] = timestamps
    if keys.duplicated(["symbol", "timeframe", "timestamp"]).any():
        raise ValueError("Duplicate symbol/timeframe/timestamp rows must be resolved first.")


def build_ml_dataset(
    frame: pd.DataFrame,
    *,
    neutral_bands_pct: dict[int, float],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build causal features, explicit directional targets and date partitions.

    This prepares a research dataset only. It does not fit a model or claim
    predictive performance. All rows from a calendar date remain in one split.
    """
    _validate_ml_input(frame)

    # Reset only the working index so independently generated columns align
    # positionally even if the uploaded CSV contains duplicate index labels.
    source = frame.reset_index(drop=True).copy()
    features = add_features(source)

    # Feed the existing causal detector outputs into ML as explicit binary
    # inputs. Keep the human-readable detected_patterns string out of training.
    detected = detect_patterns(source)
    pattern_columns = sorted(
        column for column in detected.columns
        if column.startswith("pattern_")
        and pd.api.types.is_bool_dtype(detected[column])
    )
    for column in pattern_columns:
        features[f"feature_{column}"] = detected[column].astype("int8").to_numpy()

    labelled = add_directional_targets(
        features, neutral_bands_pct=neutral_bands_pct
    )
    split = chronological_date_split(
        labelled,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    feature_columns = [
        column for column in split.columns
        if column.startswith("feature_")
        and pd.api.types.is_numeric_dtype(split[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric engineered feature columns were produced.")

    summary: dict[str, Any] = {
        "rows": int(len(split)),
        "symbols": sorted(split["symbol"].astype(str).unique().tolist()),
        "timeframes": sorted(split["timeframe"].astype(str).unique().tolist()),
        "feature_columns": feature_columns,
        "pattern_feature_columns": [
            f"feature_{column}" for column in pattern_columns
        ],
        "pattern_feature_count": int(len(pattern_columns)),
        "neutral_bands_pct": {
            str(horizon): float(band)
            for horizon, band in sorted(neutral_bands_pct.items())
        },
        "splits": {},
        "target_distributions": {},
        "limitations": [
            "Dataset preparation only; no model has been fitted.",
            "Neutral bands are explicit research parameters, not calibrated trading costs.",
            "Forward labels use close-to-close returns and exclude the unobservable session tail.",
            "No execution, slippage, liquidity or profitability claim is modelled here.",
            "Source-provided session eligibility and timestamp semantics must be audited separately.",
        ],
    }

    for split_name, group in split.groupby("validation_split", sort=False):
        dates = pd.to_datetime(group["timestamp"]).dt.strftime("%Y-%m-%d")
        summary["splits"][str(split_name)] = {
            "rows": int(len(group)),
            "sessions": int(dates.nunique()),
            "first_date": str(dates.min()),
            "last_date": str(dates.max()),
        }
        for horizon in sorted(neutral_bands_pct):
            target_column = f"target_{int(horizon)}"
            counts = group[target_column].value_counts(dropna=False)
            summary["target_distributions"].setdefault(target_column, {})[
                str(split_name)
            ] = {
                str(label): int(count)
                for label, count in counts.items()
            }

    return split, summary
