import unittest

import numpy as np
import pandas as pd

from engine.features import add_features
from engine.market_structure import add_market_structure


class MarketStructureCausalityTests(unittest.TestCase):
    @staticmethod
    def candles():
        close = np.array([10, 11, 14, 12, 13, 15, 14, 13], dtype=float)
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-11-02 09:15", periods=len(close), freq="5min"),
            "open": close - 0.2,
            "high": [10, 12, 15, 13, 14, 16, 15, 14],
            "low": [9, 10, 11, 10, 11, 12, 11, 10],
            "close": close,
            "ema_20": close - 0.1,
            "ema_50": close - 0.5,
        })

    def test_swing_is_emitted_on_confirmation_bar_not_pivot_bar(self):
        result = add_market_structure(self.candles(), pivot_window=1)
        self.assertTrue(np.isnan(result.loc[2, "swing_high_confirmed"]))
        self.assertEqual(result.loc[3, "swing_high_confirmed"], 15.0)

    def test_future_candles_do_not_change_already_confirmed_structure(self):
        original = self.candles()
        changed = original.copy()
        changed.loc[4:, ["open", "high", "low", "close", "ema_20", "ema_50"]] *= 2.0

        left = add_market_structure(original, pivot_window=1)
        right = add_market_structure(changed, pivot_window=1)
        columns = [
            "trend_state", "swing_high_confirmed", "swing_low_confirmed",
            "swing_high_label", "swing_low_label", "structure_event",
            "structure_break_level", "structure_bias",
        ]
        pd.testing.assert_frame_equal(
            left.loc[:3, columns],
            right.loc[:3, columns],
            check_dtype=False,
            check_exact=True,
        )

    def test_confirmed_swing_feature_only_uses_level_from_confirmation_onward(self):
        frame = self.candles().drop(columns=["ema_20", "ema_50"]).copy()
        frame["swing_high_confirmed"] = [np.nan, np.nan, np.nan, 15.0, np.nan, 16.0, np.nan, np.nan]
        result = add_features(frame)

        self.assertTrue(result.loc[:2, "feature_distance_to_confirmed_swing_high_pct"].isna().all())
        self.assertAlmostEqual(result.loc[3, "feature_distance_to_confirmed_swing_high_pct"], 12.0 / 15.0 - 1.0)
        self.assertAlmostEqual(result.loc[4, "feature_distance_to_confirmed_swing_high_pct"], 13.0 / 15.0 - 1.0)
        self.assertAlmostEqual(result.loc[5, "feature_distance_to_confirmed_swing_high_pct"], 15.0 / 16.0 - 1.0)


if __name__ == "__main__":
    unittest.main()
