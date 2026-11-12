"""Regression and edge-case tests for the research engine.

These tests use synthetic data only. They do not call broker APIs or place orders.
"""
import unittest

import numpy as np
import pandas as pd

from engine.data_loader import normalize_ohlcv
from engine.indicators import add_indicators
from engine.market_structure import add_market_structure
from engine.outcomes import label_forward_returns, outcome_summary
from engine.patterns import detect_patterns


def candles(rows, timestamps=True):
    frame = pd.DataFrame(rows)
    if timestamps:
        frame.insert(
            0,
            "timestamp",
            pd.date_range("2026-01-01 09:15", periods=len(frame), freq="5min"),
        )
    return frame


class DataLoaderEdgeCaseTests(unittest.TestCase):
    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            normalize_ohlcv(pd.DataFrame())

    def test_missing_ohlc_columns_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            normalize_ohlcv(pd.DataFrame({"timestamp": ["2026-01-01"], "open": [10]}))

    def test_aliases_missing_volume_and_synthetic_time_axis(self):
        raw = pd.DataFrame({
            " O ": [10, 11, 12],
            "H": [11, 12, 13],
            "L": [9, 10, 11],
            "C": [10.5, 11.5, 12.5],
        })
        result = normalize_ohlcv(raw)
        self.assertEqual(result["volume"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(result["timestamp"].tolist(), [0, 1, 2])

    def test_duplicate_timestamps_keep_last_observation(self):
        raw = candles({
            "open": [10, 11, 12, 13],
            "high": [11, 12, 13, 14],
            "low": [9, 10, 11, 12],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [100, 110, 120, 130],
        })
        raw.loc[3, "timestamp"] = raw.loc[2, "timestamp"]
        result = normalize_ohlcv(raw)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[-1]["close"], 13.5)

    def test_invalid_prices_and_negative_volume_are_removed(self):
        raw = candles({
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "volume": [100, 110, 120, 130, -1, 150],
        })
        raw.loc[1, "high"] = 1  # high below open/close
        raw.loc[3, "close"] = 0  # non-positive OHLC
        result = normalize_ohlcv(raw)
        self.assertEqual(len(result), 3)
        self.assertTrue((result["volume"] >= 0).all())
        self.assertTrue((result["high"] >= result[["open", "close", "low"]].max(axis=1)).all())

    def test_fewer_than_three_valid_rows_is_rejected(self):
        raw = candles({
            "open": [10, 11],
            "high": [11, 12],
            "low": [9, 10],
            "close": [10.5, 11.5],
        })
        with self.assertRaisesRegex(ValueError, "Fewer than 3"):
            normalize_ohlcv(raw)


class IndicatorEdgeCaseTests(unittest.TestCase):
    def test_vwap_resets_at_new_trading_session(self):
        timestamps = list(pd.date_range("2026-01-01 09:15", periods=15, freq="5min"))
        timestamps += list(pd.date_range("2026-01-02 09:15", periods=15, freq="5min"))
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * 30,
            "high": [102.0] * 30,
            "low": [99.0] * 30,
            "close": [101.0] * 30,
            "volume": [10.0] * 30,
        })
        result = add_indicators(frame)
        self.assertAlmostEqual(result.loc[15, "vwap"], 100.6666666667, places=6)
        self.assertTrue(np.isfinite(result.loc[15, "vwap"]))

    def test_zero_volume_produces_missing_vwap_not_infinity(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="5min"),
            "open": [10.0] * 20,
            "high": [11.0] * 20,
            "low": [9.0] * 20,
            "close": [10.0] * 20,
            "volume": [0.0] * 20,
        })
        result = add_indicators(frame)
        self.assertTrue(result["vwap"].isna().all())
        self.assertFalse(np.isinf(result["vwap"].to_numpy()).any())


class MarketStructureEdgeCaseTests(unittest.TestCase):
    def test_invalid_pivot_window_is_rejected(self):
        frame = pd.DataFrame({
            "high": [2, 3, 4],
            "low": [1, 2, 3],
            "close": [1.5, 2.5, 3.5],
            "ema_20": [2, 2, 2],
            "ema_50": [2, 2, 2],
        })
        with self.assertRaisesRegex(ValueError, "pivot_window"):
            add_market_structure(frame, pivot_window=0)

    def test_missing_required_columns_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            add_market_structure(pd.DataFrame({"high": [1], "low": [0], "close": [0.5]}))

    def test_swing_high_is_stamped_only_after_confirmation(self):
        frame = pd.DataFrame({
            "high": [1.0, 2.0, 5.0, 3.0, 2.0, 2.0, 2.0],
            "low": [0.5, 1.0, 2.0, 1.0, 0.8, 0.8, 0.8],
            "close": [1.0, 1.5, 3.0, 2.0, 2.0, 2.0, 2.0],
            "ema_20": [2.0] * 7,
            "ema_50": [2.0] * 7,
        })
        result = add_market_structure(frame, pivot_window=2)
        self.assertTrue(result["swing_high_confirmed"].iloc[:4].isna().all())
        self.assertEqual(result["swing_high_confirmed"].iloc[4], 5.0)


class PatternAndOutcomeEdgeCaseTests(unittest.TestCase):
    def test_zero_range_candles_do_not_create_nan_pattern_flags(self):
        frame = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [100.0] * 5,
            "low": [100.0] * 5,
            "close": [100.0] * 5,
        })
        result = detect_patterns(frame)
        pattern_columns = [column for column in result if column.startswith("pattern_")]
        self.assertTrue(pattern_columns)
        for column in pattern_columns:
            self.assertFalse(result[column].isna().any())
            self.assertTrue(result[column].map(lambda value: isinstance(value, (bool, np.bool_))).all())

    def test_forward_returns_leave_unobservable_tail_unlabelled(self):
        frame = pd.DataFrame({"close": [100.0, 110.0, 105.0, 120.0]})
        result = label_forward_returns(frame, horizons=(2,))
        self.assertAlmostEqual(result.loc[0, "forward_return_2"], 5.0)
        self.assertTrue(result["forward_return_2"].iloc[-2:].isna().all())

    def test_non_positive_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            label_forward_returns(pd.DataFrame({"close": [1.0, 2.0]}), horizons=(0,))

    def test_empty_pattern_sample_returns_zero_observations(self):
        frame = pd.DataFrame({
            "pattern_doji": [False, False, False],
            "forward_return_3": [1.0, -1.0, 2.0],
        })
        summary = outcome_summary(frame, "pattern_doji", 3)
        self.assertEqual(summary["observations"], 0)
        self.assertIsNone(summary["mean_return_pct"])
        self.assertIsNone(summary["positive_rate_pct"])

    def test_missing_outcome_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            outcome_summary(pd.DataFrame({"pattern_doji": [True]}), "pattern_doji", 3)


if __name__ == "__main__":
    unittest.main()
