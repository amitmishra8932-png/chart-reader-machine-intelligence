"""Tests for controlled Phase 8.7 feature-family ablations."""
import unittest

import pandas as pd

from engine.ml_ablation import evaluate_feature_ablations


def make_dataset():
    rows = []
    dates = pd.date_range("2026-01-05", periods=8, freq="B")
    for day_index, day in enumerate(dates):
        for bar in range(12):
            signal = (bar + day_index) % 3
            rows.append({
                "timestamp": day + pd.Timedelta(hours=9, minutes=15 + bar * 5),
                "validation_split": (
                    "train" if day_index < 5
                    else "validation" if day_index == 5
                    else "test"
                ),
                "feature_body": float(signal + day_index * 0.01),
                "feature_volume": float(100 + bar + day_index),
                "feature_pattern_doji": int(bar % 4 == 0),
                "feature_pattern_engulfing": int(bar % 5 == 0),
                "target_3": (-1, 0, 1)[signal],
                "target_6": (-1, 0, 1)[signal],
                "target_12": (-1, 0, 1)[signal],
            })
    return pd.DataFrame(rows)


class FeatureAblationTests(unittest.TestCase):
    def test_compares_three_feature_families_on_same_splits(self):
        report = evaluate_feature_ablations(make_dataset(), horizons=(3,))
        self.assertEqual(
            set(report["feature_groups"]),
            {"all_features", "non_pattern_features", "pattern_only"},
        )
        self.assertEqual(report["feature_groups"]["all_features"]["count"], 4)
        self.assertEqual(report["feature_groups"]["non_pattern_features"]["count"], 2)
        self.assertEqual(report["feature_groups"]["pattern_only"]["count"], 2)
        self.assertTrue(report["same_chronological_splits"])
        self.assertTrue(report["same_model_and_preprocessing"])
        self.assertEqual(
            report["results"]["all_features"]["splits"],
            report["results"]["pattern_only"]["splits"],
        )

    def test_each_group_reports_validation_and_held_out_test(self):
        report = evaluate_feature_ablations(make_dataset(), horizons=(3,))
        for group in report["results"].values():
            for model in group["horizons"]["3"]["models"].values():
                self.assertIn("validation", model)
                self.assertIn("test", model)
        self.assertIn("test_set_role", report)

    def test_rejects_missing_pattern_family(self):
        frame = make_dataset().drop(columns=["feature_pattern_doji", "feature_pattern_engulfing"])
        with self.assertRaisesRegex(ValueError, "No feature_pattern_"):
            evaluate_feature_ablations(frame, horizons=(3,))

    def test_rejects_missing_non_pattern_family(self):
        frame = make_dataset().drop(columns=["feature_body", "feature_volume"])
        with self.assertRaisesRegex(ValueError, "No non-pattern"):
            evaluate_feature_ablations(frame, horizons=(3,))


if __name__ == "__main__":
    unittest.main()
