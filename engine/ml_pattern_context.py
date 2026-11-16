"""Validation-only candle-pattern contribution and context diagnostics.

This module reports descriptive associations, not causal effects or trading
signals. It deliberately excludes train/test rows from every outcome statistic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CLASS_LABELS = (-1, 0, 1)


def _class_rates(values: pd.Series) -> dict[str, float]:
    counts = values.value_counts(normalize=True)
    return {str(label): float(counts.get(label, 0.0)) for label in CLASS_LABELS}


def _comparison(
    frame: pd.DataFrame, pattern_column: str, target_column: str, min_support: int
) -> dict[str, Any]:
    present = frame[frame[pattern_column]]
    absent = frame[~frame[pattern_column]]
    result: dict[str, Any] = {
        "pattern_rows": int(len(present)),
        "comparison_rows": int(len(absent)),
        "eligible": len(present) >= min_support and len(absent) >= min_support,
    }
    if result["eligible"]:
        present_rates = _class_rates(present[target_column].astype(int))
        absent_rates = _class_rates(absent[target_column].astype(int))
        result["pattern_class_rates"] = present_rates
        result["comparison_class_rates"] = absent_rates
        result["rate_difference_pattern_minus_comparison"] = {
            label: float(present_rates[label] - absent_rates[label])
            for label in present_rates
        }
    return result


def analyze_pattern_context(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (3, 6, 12),
    min_support: int = 30,
    trend_feature: str = "feature_sma_20_distance_pct",
    volume_feature: str = "feature_relative_volume_20",
    elevated_volume_threshold: float = 1.5,
) -> dict[str, Any]:
    """Summarise per-pattern outcome associations on validation rows only.

    Trend regime is close relative to SMA20 (above/below). Elevated volume is
    current volume divided by the prior-20-candle mean. Missing context rows
    are excluded from interaction tables, never imputed.
    """
    required = {"validation_split", "symbol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if min_support < 1:
        raise ValueError("min_support must be a positive integer.")
    if not np.isfinite(elevated_volume_threshold) or elevated_volume_threshold <= 0:
        raise ValueError("elevated_volume_threshold must be finite and positive.")

    validation = frame[frame["validation_split"].astype(str) == "validation"].copy()
    if validation.empty:
        raise ValueError("No validation rows are available.")
    pattern_columns = sorted(
        column for column in validation.columns
        if column.startswith("feature_pattern_")
        and pd.api.types.is_numeric_dtype(validation[column])
    )
    if not pattern_columns:
        raise ValueError("No numeric feature_pattern_ columns are available.")

    report: dict[str, Any] = {
        "experiment": "validation_only_pattern_contribution_and_context",
        "validation_rows": int(len(validation)),
        "symbols": sorted(validation["symbol"].astype(str).unique().tolist()),
        "pattern_columns": pattern_columns,
        "minimum_support_per_comparison_group": int(min_support),
        "context_definitions": {
            "trend": f"{trend_feature} > 0 means above SMA20; <= 0 means below/on SMA20",
            "volume": f"{volume_feature} >= {elevated_volume_threshold} means elevated",
        },
        "horizons": {},
        "guardrails": [
            "Only rows explicitly assigned to validation are used for all outcome summaries.",
            "Pattern-present versus pattern-absent differences are descriptive associations, not causal effects.",
            "Correlated pattern flags and repeated candles mean comparisons are not independent.",
            "Subgroups below minimum support are reported as ineligible, not ranked.",
            "No feature selection, threshold tuning, profitability or trading-signal claim is made.",
            "Held-out test rows are not read for outcome analysis.",
        ],
    }

    for horizon in horizons:
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0:
            raise ValueError("Horizons must be positive integers.")
        target = f"target_{int(horizon)}"
        if target not in validation.columns:
            raise ValueError(f"Missing target column {target!r}.")
        usable = validation[validation[target].notna()].copy()
        labels = pd.to_numeric(usable[target], errors="coerce")
        if labels.isna().any() or not set(labels.astype(int).unique()).issubset(set(CLASS_LABELS)):
            raise ValueError(f"{target} must contain only -1, 0 and +1 labels.")
        usable[target] = labels.astype(int)

        horizon_report: dict[str, Any] = {
            "labelled_validation_rows": int(len(usable)),
            "patterns": {},
        }
        for column in pattern_columns:
            flags = pd.to_numeric(usable[column], errors="coerce")
            if flags.isna().any() or not set(flags.unique()).issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f"{column} must contain only binary 0/1 values.")
            usable[column] = flags.astype(bool)
            item: dict[str, Any] = {
                "occurrences": int(usable[column].sum()),
                "occurrence_rate": float(usable[column].mean()),
                "overall": _comparison(usable, column, target, min_support),
                "by_symbol": {},
                "by_context": {},
            }
            for symbol, group in usable.groupby("symbol", sort=True):
                item["by_symbol"][str(symbol)] = _comparison(
                    group, column, target, min_support
                )

            if trend_feature in usable.columns and volume_feature in usable.columns:
                context = usable[[trend_feature, volume_feature]].apply(
                    pd.to_numeric, errors="coerce"
                )
                context_frame = usable.copy()
                trend_values = context[trend_feature]
                volume_values = context[volume_feature]
                trend_regime = pd.Series(pd.NA, index=context.index, dtype="object")
                volume_regime = pd.Series(index=context.index, dtype="object")
                trend_valid = trend_values.notna()
                volume_valid = volume_values.notna()
                trend_regime.loc[trend_valid & (trend_values > 0)] = "above_sma20"
                trend_regime.loc[trend_valid & (trend_values <= 0)] = "below_or_at_sma20"
                volume_regime.loc[volume_valid & (volume_values >= elevated_volume_threshold)] = "elevated_volume"
                volume_regime.loc[volume_valid & (volume_values < elevated_volume_threshold)] = "normal_or_low_volume"
                context_frame["__trend_regime"] = trend_regime
                context_frame["__volume_regime"] = volume_regime
                for trend_name in ("above_sma20", "below_or_at_sma20"):
                    for volume_name in ("elevated_volume", "normal_or_low_volume"):
                        subset = context_frame[
                            (context_frame["__trend_regime"] == trend_name)
                            & (context_frame["__volume_regime"] == volume_name)
                        ]
                        key = f"{trend_name}__{volume_name}"
                        item["by_context"][key] = _comparison(
                            subset, column, target, min_support
                        )
            horizon_report["patterns"][column] = item
        report["horizons"][str(horizon)] = horizon_report

    return report
