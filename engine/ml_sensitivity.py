"""Neutral-band sensitivity analysis for leakage-aware ML research.

Each candidate percentage is applied uniformly to the requested horizons.
This is a label-sensitivity experiment, not a search for a profitable setting.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from engine.ml_ablation import evaluate_feature_ablations
from engine.ml_dataset import build_ml_dataset

DEFAULT_NEUTRAL_BANDS_PCT = (0.10, 0.15, 0.20, 0.30)


def _validate_grid(bands: Sequence[float], horizons: Sequence[int]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if not bands:
        raise ValueError("Provide at least one neutral-band percentage.")
    normalized_bands = tuple(float(value) for value in bands)
    if any(not np.isfinite(value) or value <= 0 for value in normalized_bands):
        raise ValueError("Neutral-band percentages must be finite and positive.")
    if len(set(normalized_bands)) != len(normalized_bands):
        raise ValueError("Neutral-band percentages must be unique.")
    if not horizons or any(isinstance(value, bool) or int(value) != value or int(value) <= 0 for value in horizons):
        raise ValueError("Horizons must be a non-empty sequence of positive integers.")
    normalized_horizons = tuple(int(value) for value in horizons)
    if len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError("Horizons must be unique.")
    return normalized_bands, normalized_horizons


def run_neutral_band_sensitivity(
    source: pd.DataFrame,
    *,
    bands_pct: Sequence[float] = DEFAULT_NEUTRAL_BANDS_PCT,
    horizons: Sequence[int] = (3, 6, 12),
) -> dict[str, Any]:
    """Rebuild labels and run Phase 8.7 ablations for each neutral band.

    The chronological split is rebuilt from the same source data for each
    candidate. The function verifies that split metadata is identical across
    candidates, preventing comparisons across different date partitions.
    """
    bands, horizon_values = _validate_grid(bands_pct, horizons)
    results: dict[str, Any] = {}
    reference_splits: dict[str, Any] | None = None

    for band in bands:
        neutral_bands = {horizon: band for horizon in horizon_values}
        dataset, dataset_summary = build_ml_dataset(
            source,
            neutral_bands_pct=neutral_bands,
        )
        ablation = evaluate_feature_ablations(
            dataset,
            horizons=tuple(horizon_values),
        )
        split_metadata = ablation["results"]["all_features"]["splits"]
        if reference_splits is None:
            reference_splits = split_metadata
        elif split_metadata != reference_splits:
            raise ValueError("Chronological split metadata changed across neutral-band candidates.")

        results[f"{band:.2f}"] = {
            "neutral_band_pct": band,
            "dataset_summary": dataset_summary,
            "feature_ablation": ablation,
        }

    return {
        "experiment": "neutral_band_label_sensitivity",
        "candidate_neutral_bands_pct": list(bands),
        "horizons_candles": list(horizon_values),
        "band_application": "same percentage applied independently to each horizon",
        "same_chronological_splits_across_candidates": True,
        "split_metadata": reference_splits,
        "selection_policy": (
            "Exploratory sensitivity only. Compare validation results descriptively; "
            "do not select a band from held-out test metrics. No winning band is declared."
        ),
        "results_by_band_pct": results,
        "limitations": [
            "Neutral bands are research label parameters, not calibrated transaction costs.",
            "A band that improves classification metrics is not necessarily more tradable.",
            "Test metrics are held-out reporting only and must not drive band selection.",
            "The underlying historical pilot contains only five symbols, not the full NIFTY 50 universe.",
            "No fees, slippage, liquidity, turnover, execution or profitability are modelled.",
        ],
    }
