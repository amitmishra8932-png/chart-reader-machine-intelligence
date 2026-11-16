"""Tests for leakage-safe Phase 8.5 target and dataset preparation."""
import unittest

import numpy as np
import pandas as pd

from engine.ml_dataset import build_ml_dataset
from engine.ml_targets import add_directional_targets


def sample_frame():
    dates = []
    for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"):
        dates.extend(pd.date_range(f"{day} 09:15", periods=8, freq="5min"))
    n = len(dates)
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAA"] * n,
        "timeframe": ["5m"] * n,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": [1000.0] * n,
    })


class DirectionalTargetTests(unittest.TestCase):
    def test_targets_use_explicit_bands_and_leave_session_tails_unlabelled(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-05 09:15", periods=4, freq="5min"),
            "symbol": ["AAA"] * 4,
            "timeframe": ["5m"] * 4,
            "close": [100.0, 101.0, 100.0, 102.0],
        })
        result = add_directional_targets(frame, neutral_bands_pct={1: 0.5})
        self.assertEqual(int(result.loc[0, "target_1"]), 1)
        self.assertEqual(int(result.loc[1, "target_1"]), -1)
        self.assertTrue(result["target_1"].iloc[-1:].isna().all())

    def test_forward_labels_never_cross_calendar_sessions(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-01-05 15:25", "2026-01-06 09:15"]),
            "symbol": ["AAA", "AAA"],
            "timeframe": ["5m", "5m"],
            "close": [100.0, 110.0],
        })
        result = add_directional_targets(frame, neutral_bands_pct={1: 0.1})
        self.assertTrue(result["target_1"].isna().all())

    def test_interleaved_symbols_are_isolated_and_order_is_preserved(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-05 09:15", "2026-01-05 09:15",
                "2026-01-05 09:20", "2026-01-05 09:20",
            ]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "timeframe": ["5m"] * 4,
            "close": [100.0, 200.0, 110.0, 180.0],
        }, index=[9, 3, 8, 2])
        result = add_directional_targets(frame, neutral_bands_pct={1: 1.0})
        self.assertEqual(result.index.tolist(), [9, 3, 8, 2])
        self.assertEqual(int(result.loc[9, "target_1"]), 1)
        self.assertEqual(int(result.loc[3, "target_1"]), -1)

    def test_missing_band_duplicate_key_and_invalid_timestamp_are_rejected(self):
        frame = sample_frame().iloc[:4].copy()
        with self.assertRaisesRegex(ValueError, "explicit neutral band"):
            add_directional_targets(frame, neutral_bands_pct={})
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            add_directional_targets(duplicate, neutral_bands_pct={1: 0.2})
        invalid = frame.copy()
        invalid["timestamp"] = invalid["timestamp"].astype(object)
        invalid.loc[0, "timestamp"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "parseable"):
            add_directional_targets(invalid, neutral_bands_pct={1: 0.2})

    def test_invalid_bands_and_fractional_horizons_are_rejected(self):
        frame = sample_frame().iloc[:4].copy()
        with self.assertRaisesRegex(ValueError, "positive percentages"):
            add_directional_targets(frame, neutral_bands_pct={1: 0})
        with self.assertRaisesRegex(ValueError, "positive integers"):
            add_directional_targets(frame, neutral_bands_pct={1.5: 0.2})


class MlDatasetPreparationTests(unittest.TestCase):
    def test_builds_causal_features_targets_and_date_splits(self):
        result, report = build_ml_dataset(
            sample_frame(), neutral_bands_pct={3: 0.1, 6: 0.2, 12: 0.3}
        )
        self.assertTrue({"target_3", "target_6", "target_12"}.issubset(result.columns))
        self.assertTrue({"train", "validation", "test"}.issubset(set(result["validation_split"])))
        self.assertEqual(report["rows"], len(result))
        self.assertTrue(report["feature_columns"])
        self.assertIn("no model has been fitted", " ".join(report["limitations"]))


if __name__ == "__main__":
    unittest.main()
