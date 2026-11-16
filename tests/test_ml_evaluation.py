"""Tests for Phase 8.6 leakage-aware baseline ML evaluation."""
import unittest

import numpy as np
import pandas as pd

from engine.ml_evaluation import evaluate_baseline_models


def make_dataset():
    rows = []
    # Eight distinct dates: five train, one validation, two test.
    dates = pd.date_range("2026-01-05", periods=8, freq="B")
    for day_index, day in enumerate(dates):
        for bar in range(12):
            signal = (bar + day_index) % 3
            target = (-1, 0, 1)[signal]
            rows.append({
                "timestamp": day + pd.Timedelta(hours=9, minutes=15 + bar * 5),
                "validation_split": (
                    "train" if day_index < 5
                    else "validation" if day_index == 5
                    else "test"
                ),
                "feature_body": float(signal + day_index * 0.01),
                "feature_volume": float(100 + bar + day_index),
                "target_3": target,
                "target_6": target,
                "target_12": target,
            })
    return pd.DataFrame(rows)


class BaselineEvaluationTests(unittest.TestCase):
    def test_evaluates_majority_and_logistic_with_fixed_class_order(self):
        report = evaluate_baseline_models(make_dataset())
        horizon = report["horizons"]["3"]
        self.assertIn("majority_baseline", horizon["models"])
        self.assertIn("logistic_regression", horizon["models"])
        metrics = horizon["models"]["logistic_regression"]["test"]
        self.assertEqual(metrics["confusion_matrix_labels"], [-1, 0, 1])
        self.assertEqual(len(metrics["confusion_matrix"]), 3)
        self.assertEqual(metrics["rows"], 24)
        self.assertEqual(report["fit_scope"], "training dates only")

    def test_preprocessing_ignores_test_only_feature_values_when_fitting(self):
        original = make_dataset()
        altered = original.copy()
        altered.loc[altered["validation_split"] == "test", "feature_body"] = 1e12
        report_a = evaluate_baseline_models(original, horizons=(3,))
        report_b = evaluate_baseline_models(altered, horizons=(3,))
        # Changing held-out feature values can change predictions, but never
        # changes the training-fitted feature set or train class distribution.
        self.assertEqual(
            report_a["horizons"]["3"]["training_classes"],
            report_b["horizons"]["3"]["training_classes"],
        )
        self.assertEqual(report_a["feature_columns"], report_b["feature_columns"])

    def test_rejects_calendar_date_shared_across_splits(self):
        frame = make_dataset()
        frame.loc[frame.index[0], "validation_split"] = "test"
        with self.assertRaisesRegex(ValueError, "Calendar dates must not be shared"):
            evaluate_baseline_models(frame, horizons=(3,))

    def test_rejects_missing_split_or_unlabelled_partition(self):
        frame = make_dataset()
        frame.loc[frame["validation_split"] == "test", "validation_split"] = "train"
        with self.assertRaisesRegex(ValueError, "All train/validation/test splits"):
            evaluate_baseline_models(frame, horizons=(3,))

    def test_rejects_single_class_training_partition(self):
        frame = make_dataset()
        frame.loc[frame["validation_split"] == "train", "target_3"] = 0
        with self.assertRaisesRegex(ValueError, "fewer than two classes"):
            evaluate_baseline_models(frame, horizons=(3,))

    def test_rejects_non_feature_or_missing_target(self):
        frame = make_dataset().drop(columns=["target_6"])
        with self.assertRaisesRegex(ValueError, "Missing target column"):
            evaluate_baseline_models(frame, horizons=(6,))


if __name__ == "__main__":
    unittest.main()
