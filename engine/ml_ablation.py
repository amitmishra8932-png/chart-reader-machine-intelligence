"""Controlled feature-family ablations for Phase 8.7.

All ablations reuse the same chronological partitions, targets, estimator and
train-only preprocessing. This is a classification robustness experiment,
not a strategy backtest or a claim of trading profitability.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.ml_evaluation import evaluate_baseline_models


def evaluate_feature_ablations(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (3, 6, 12),
) -> dict[str, Any]:
    """Compare all, non-pattern and pattern-only feature families.

    Pattern features are identified strictly by the feature_pattern_ prefix.
    All other numeric feature_ columns form the non-pattern group. Every
    group is evaluated independently with identical date splits and model
    settings. The held-out test partition is reported but must not be used to
    choose a feature family or tune the model.
    """
    feature_columns = [
        column for column in frame.columns
        if column.startswith("feature_") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    pattern_columns = [column for column in feature_columns if column.startswith("feature_pattern_")]
    non_pattern_columns = [column for column in feature_columns if column not in pattern_columns]

    if not feature_columns:
        raise ValueError("No numeric feature_ columns are available for ablation.")
    if not non_pattern_columns:
        raise ValueError("No non-pattern feature family is available for comparison.")
    if not pattern_columns:
        raise ValueError("No feature_pattern_ columns are available for pattern ablation.")

    groups = {
        "all_features": feature_columns,
        "non_pattern_features": non_pattern_columns,
        "pattern_only": pattern_columns,
    }
    results: dict[str, Any] = {}
    for group_name, columns in groups.items():
        results[group_name] = evaluate_baseline_models(
            frame, horizons=horizons, feature_columns=columns
        )

    deltas: dict[str, Any] = {}
    for horizon in map(str, horizons):
        deltas[horizon] = {}
        for group_name in ("all_features", "non_pattern_features", "pattern_only"):
            deltas[horizon][group_name] = {}
            for split_name in ("validation", "test"):
                model_metrics = results[group_name]["horizons"][horizon]["models"]["logistic_regression"][split_name]
                baseline_metrics = results[group_name]["horizons"][horizon]["models"]["majority_baseline"][split_name]
                deltas[horizon][group_name][split_name] = {
                    metric: float(model_metrics[metric] - baseline_metrics[metric])
                    for metric in ("balanced_accuracy", "macro_f1")
                }

    return {
        "experiment": "controlled_feature_family_ablation",
        "feature_groups": {
            name: {"count": len(columns), "columns": columns}
            for name, columns in groups.items()
        },
        "same_chronological_splits": True,
        "same_model_and_preprocessing": True,
        "test_set_role": "held-out reporting only; not for feature selection or tuning",
        "results": results,
        "logistic_minus_majority_deltas": deltas,
        "interpretation_guardrails": [
            "Compare feature families on validation first; treat test as a final held-out report.",
            "Ablation differences are descriptive evidence, not proof of causal predictive value.",
            "Classification metrics do not account for fees, slippage, liquidity, turnover or execution.",
            "Pattern-only performance can be unstable when patterns are rare or absent in a small sample.",
            "Results on a five-symbol pilot dataset are not representative of the NIFTY 50 universe.",
            "Do not select a production model or claim profitability from this experiment alone.",
        ],
    }
