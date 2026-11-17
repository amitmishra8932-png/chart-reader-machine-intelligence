import unittest

import numpy as np
import pandas as pd

from scripts.phase_8_20_cost_aware_backtest import (
    HORIZON_FEATURES,
    fit_frozen_test_predictions,
    run_experiment,
)


def synthetic_chronological_dataset():
    rows = []
    dates = [
        ("2026-01-05", "train"),
        ("2026-01-06", "validation"),
        ("2026-01-07", "test"),
    ]
    all_features = sorted({
        feature for columns in HORIZON_FEATURES.values() for feature in columns
    })
    for day_index, (date, split) in enumerate(dates):
        times = pd.date_range(f"{date} 09:15", periods=75, freq="5min")
        for i, timestamp in enumerate(times):
            row = {
                "timestamp": timestamp,
                "symbol": "ABC",
                "timeframe": "5m",
                "open": 100.0 + i * 0.01 + day_index,
                "close": 100.1 + i * 0.01 + day_index,
                "validation_split": split,
            }
            for feature_index, feature in enumerate(all_features):
                row[feature] = np.sin((i + 1) * (feature_index + 1) / 13.0)
            row["target_3"] = [-1, 0, 1][i % 3]
            row["target_6"] = [-1, 0, 1][(i + 1) % 3]
            row["target_12"] = [-1, 0, 1][(i + 2) % 3]
            rows.append(row)
    return pd.DataFrame(rows)


class Phase820RunnerTests(unittest.TestCase):
    def test_predictions_are_test_only_and_all_horizons_run(self):
        predictions = fit_frozen_test_predictions(
            synthetic_chronological_dataset()
        )
        self.assertEqual(set(predictions), {3, 6, 12})
        for horizon, frame in predictions.items():
            self.assertEqual(len(frame), 75)
            self.assertTrue(frame["prediction"].isin([-1, 0, 1]).all())
            self.assertEqual(frame["horizon_bars"].unique().tolist(), [horizon])
            self.assertEqual(frame["session_date"].nunique(), 1)

    def test_cost_sensitivity_has_all_frozen_scenarios(self):
        _, report = run_experiment(synthetic_chronological_dataset())
        self.assertEqual(
            set(report["horizons"]["3"]["cost_sensitivity"]),
            {"0", "5", "10", "20", "30", "50"},
        )
        self.assertIn("by_symbol", report["horizons"]["3"]["cost_sensitivity"]["5"])

    def test_rejects_fractional_targets(self):
        frame = synthetic_chronological_dataset()
        frame.loc[0, "target_3"] = 0.5
        with self.assertRaises(ValueError):
            fit_frozen_test_predictions(frame)

    def test_rejects_calendar_date_split_leakage(self):
        frame = synthetic_chronological_dataset()
        frame.loc[0, "validation_split"] = "validation"
        with self.assertRaises(ValueError):
            fit_frozen_test_predictions(frame)


if __name__ == "__main__":
    unittest.main()
