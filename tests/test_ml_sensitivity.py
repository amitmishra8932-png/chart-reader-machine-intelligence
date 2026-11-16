"""Tests for Phase 8.8 neutral-band sensitivity orchestration."""
import unittest
from unittest.mock import patch

import pandas as pd

from engine.ml_sensitivity import run_neutral_band_sensitivity


def fake_dataset(source, *, neutral_bands_pct):
    split = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "validation_split": ["train", "validation", "test"],
    })
    summary = {"rows": 3, "neutral_bands_pct": {str(k): v for k, v in neutral_bands_pct.items()}}
    return split, summary


def fake_ablation(frame, *, horizons):
    return {
        "results": {
            "all_features": {"splits": {
                "train": {"rows": 1, "first_date": "2026-01-05", "last_date": "2026-01-05"},
                "validation": {"rows": 1, "first_date": "2026-01-06", "last_date": "2026-01-06"},
                "test": {"rows": 1, "first_date": "2026-01-07", "last_date": "2026-01-07"},
            }}
        },
        "logistic_minus_majority_deltas": {},
    }


class NeutralBandSensitivityTests(unittest.TestCase):
    @patch("engine.ml_sensitivity.evaluate_feature_ablations", side_effect=fake_ablation)
    @patch("engine.ml_sensitivity.build_ml_dataset", side_effect=fake_dataset)
    def test_runs_confirmed_four_band_grid_on_same_splits(self, build, evaluate):
        report = run_neutral_band_sensitivity(
            pd.DataFrame({"raw": [1]}),
            bands_pct=(0.10, 0.15, 0.20, 0.30),
            horizons=(3, 6, 12),
        )
        self.assertEqual(
            report["candidate_neutral_bands_pct"],
            [0.10, 0.15, 0.20, 0.30],
        )
        self.assertEqual(
            list(report["results_by_band_pct"]),
            ["0.10", "0.15", "0.20", "0.30"],
        )
        self.assertEqual(build.call_count, 4)
        self.assertEqual(evaluate.call_count, 4)
        self.assertTrue(report["same_chronological_splits_across_candidates"])
        for call in build.call_args_list:
            self.assertEqual(
                call.kwargs["neutral_bands_pct"],
                {3: call.kwargs["neutral_bands_pct"][3],
                 6: call.kwargs["neutral_bands_pct"][3],
                 12: call.kwargs["neutral_bands_pct"][3]},
            )

    def test_rejects_invalid_or_duplicate_band_grid(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            run_neutral_band_sensitivity(pd.DataFrame(), bands_pct=(0.1, 0.0))
        with self.assertRaisesRegex(ValueError, "unique"):
            run_neutral_band_sensitivity(pd.DataFrame(), bands_pct=(0.1, 0.1))

    def test_rejects_split_drift_between_candidates(self):
        with patch("engine.ml_sensitivity.build_ml_dataset", side_effect=fake_dataset), \
             patch("engine.ml_sensitivity.evaluate_feature_ablations") as evaluate:
            evaluate.side_effect = [
                fake_ablation(None),
                {
                    "results": {"all_features": {"splits": {"train": {"rows": 99}}}},
                    "logistic_minus_majority_deltas": {},
                },
            ]
            with self.assertRaisesRegex(ValueError, "split metadata changed"):
                run_neutral_band_sensitivity(pd.DataFrame(), bands_pct=(0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
