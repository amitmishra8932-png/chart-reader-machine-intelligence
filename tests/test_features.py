import unittest

import numpy as np
import pandas as pd

from engine.features import add_features


class FeatureEngineeringTests(unittest.TestCase):
    @staticmethod
    def candles(count=70):
        close = 100 + np.arange(count, dtype=float) * 0.2
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-01-05 09:15", periods=count, freq="5min"),
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.arange(count, dtype=float) + 100,
        })

    def test_adds_documented_causal_feature_families(self):
        result = add_features(self.candles())
        expected = {
            "feature_body_to_range", "feature_upper_wick_to_range",
            "feature_lower_wick_to_range", "feature_close_location",
            "feature_return_3", "feature_return_6", "feature_return_12",
            "feature_atr_14", "feature_range_atr",
            "feature_realized_volatility_20", "feature_relative_volume_20",
            "feature_volume_zscore_20", "feature_sma_20_distance_pct",
            "feature_ema_50_distance_pct",
        }
        self.assertTrue(expected.issubset(result.columns))
        self.assertEqual(len(result), 70)

    def test_future_candle_changes_do_not_change_prior_features(self):
        original = self.candles()
        changed = original.copy()
        changed.loc[50:, ["open", "high", "low", "close", "volume"]] *= 1.8
        left = add_features(original)
        right = add_features(changed)
        columns = [c for c in left.columns if c.startswith("feature_")]
        pd.testing.assert_frame_equal(
            left.loc[:49, columns], right.loc[:49, columns],
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )

    def test_relative_volume_baseline_excludes_current_candle(self):
        frame = self.candles(8)
        frame["volume"] = [10, 20, 30, 40, 50, 60, 70, 80]
        result = add_features(frame)
        self.assertTrue(np.isnan(result.loc[4, "feature_relative_volume_20"]))
        self.assertAlmostEqual(result.loc[5, "feature_relative_volume_20"], 60 / 30)

    def test_symbol_timeframe_and_session_boundaries_are_isolated(self):
        frame = self.candles(6)
        frame = pd.concat([frame.iloc[:3], frame.iloc[:3]], ignore_index=True)
        frame["symbol"] = ["AAA"] * 3 + ["BBB"] * 3
        frame["timeframe"] = ["5m"] * 6
        frame["timestamp"] = list(pd.date_range("2026-01-05 09:15", periods=3, freq="5min")) * 2
        result = add_features(frame)
        self.assertTrue(np.isnan(result.loc[0, "feature_gap_pct"]))
        self.assertTrue(np.isnan(result.loc[3, "feature_gap_pct"]))
        self.assertTrue(np.isnan(result.loc[3, "feature_return_1"]))

        next_day = frame.iloc[[2, 3]].copy()
        next_day["symbol"] = "AAA"
        next_day["timestamp"] = pd.to_datetime(["2026-01-05 15:25", "2026-01-06 09:15"])
        isolated = add_features(next_day)
        self.assertTrue(np.isnan(isolated.iloc[1]["feature_return_1"]))

    def test_unsorted_rows_restore_original_order_and_index(self):
        frame = self.candles(5).iloc[[3, 0, 4, 1, 2]].copy()
        frame.index = [30, 10, 40, 20, 25]
        result = add_features(frame)
        self.assertEqual(result.index.tolist(), [30, 10, 40, 20, 25])
        chronological = add_features(frame.sort_values("timestamp"))
        expected = chronological.set_index("timestamp")["feature_return_1"]
        for timestamp, value in zip(result["timestamp"], result["feature_return_1"]):
            if pd.isna(expected.loc[timestamp]):
                self.assertTrue(pd.isna(value))
            else:
                self.assertAlmostEqual(value, expected.loc[timestamp])

    def test_zero_range_and_missing_volume_are_safe(self):
        frame = self.candles(20).drop(columns=["volume"])
        frame.loc[5, ["open", "high", "low", "close"]] = 100.0
        result = add_features(frame)
        self.assertTrue(np.isnan(result.loc[5, "feature_body_to_range"]))
        self.assertTrue(result["feature_candle_direction"].isin([-1, 0, 1]).all())

    def test_missing_ohlc_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Missing OHLC columns"):
            add_features(pd.DataFrame({"open": [1], "close": [1]}))


if __name__ == "__main__":
    unittest.main()
