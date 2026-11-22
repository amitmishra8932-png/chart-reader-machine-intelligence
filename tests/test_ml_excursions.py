import unittest
import numpy as np
import pandas as pd
from engine.ml_excursions import add_forward_excursions

class ForwardExcursionTests(unittest.TestCase):
    @staticmethod
    def candles():
        close = np.array([100., 101., 99., 102., 100., 103.])
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-01-05 09:15", periods=6, freq="5min"),
            "symbol": "AAA", "timeframe": "5m", "close": close,
            "high": close + np.array([1., 2., 1., 3., 1., 1.]),
            "low": close - np.array([1., 1., 2., 1., 3., 1.]),
        })

    def test_excursions_use_only_next_horizon_bars(self):
        result = add_forward_excursions(self.candles(), horizons=(2,))
        self.assertAlmostEqual(result.loc[0, "forward_mfe_long_2"], 3.0)
        self.assertAlmostEqual(result.loc[0, "forward_mae_long_2"], -3.0)
        self.assertAlmostEqual(result.loc[0, "forward_mfe_short_2"], (100 / 97 - 1) * 100)
        self.assertAlmostEqual(result.loc[0, "forward_mae_short_2"], (100 / 103 - 1) * 100)
        self.assertTrue(result.loc[4:, "forward_mfe_long_2"].isna().all())

    def test_missing_bar_invalidates_window_instead_of_jumping_gap(self):
        frame = self.candles().drop(index=2).reset_index(drop=True)
        result = add_forward_excursions(frame, horizons=(2,))
        self.assertTrue(np.isnan(result.loc[0, "forward_mfe_long_2"]))

    def test_session_boundary_is_never_crossed(self):
        frame = self.candles().iloc[:3].copy()
        frame.loc[2, "timestamp"] = pd.Timestamp("2026-01-06 09:15")
        result = add_forward_excursions(frame, horizons=(2,))
        self.assertTrue(np.isnan(result.loc[0, "forward_mfe_long_2"]))

    def test_unsorted_input_restores_original_row_order(self):
        frame = self.candles().iloc[[3, 0, 1, 2, 4, 5]].copy()
        result = add_forward_excursions(frame, horizons=(2,))
        self.assertAlmostEqual(result.iloc[1]["forward_mfe_long_2"], 3.0)

    def test_duplicate_keys_and_invalid_prices_fail_closed(self):
        frame = self.candles()
        duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            add_forward_excursions(duplicated, horizons=(2,))
        frame.loc[0, "high"] = 90
        with self.assertRaisesRegex(ValueError, "high cannot be below low"):
            add_forward_excursions(frame, horizons=(2,))

if __name__ == "__main__":
    unittest.main()
