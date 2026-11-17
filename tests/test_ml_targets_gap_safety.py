import unittest

import pandas as pd

from engine.ml_targets import add_directional_targets


class GapSafeTargetTests(unittest.TestCase):
    def test_missing_intraday_bar_invalidates_horizon_that_crosses_gap(self):
        frame = pd.DataFrame({
            "timestamp": [
                "2026-06-01 09:15", "2026-06-01 09:20",
                "2026-06-01 09:30", "2026-06-01 09:35",
            ],
            "symbol": ["ABC"] * 4,
            "timeframe": ["5m"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        })
        result = add_directional_targets(
            frame, neutral_bands_pct={1: 0.20, 2: 0.20}
        )
        self.assertEqual(int(result.loc[0, "target_1"]), 1)
        self.assertTrue(pd.isna(result.loc[1, "forward_return_1"]))
        self.assertTrue(pd.isna(result.loc[1, "target_1"]))
        self.assertTrue(pd.isna(result.loc[0, "forward_return_2"]))
        self.assertEqual(int(result.loc[2, "target_1"]), 1)

    def test_regular_15m_sequence_is_labelled(self):
        frame = pd.DataFrame({
            "timestamp": [
                "2026-06-01 09:15", "2026-06-01 09:30",
                "2026-06-01 09:45",
            ],
            "symbol": ["ABC"] * 3,
            "timeframe": ["15m"] * 3,
            "close": [100.0, 101.0, 102.0],
        })
        result = add_directional_targets(frame, neutral_bands_pct={1: 0.20})
        self.assertEqual(int(result.loc[0, "target_1"]), 1)
        self.assertEqual(int(result.loc[1, "target_1"]), 1)
        self.assertTrue(pd.isna(result.loc[2, "target_1"]))

    def test_targets_never_cross_calendar_session(self):
        frame = pd.DataFrame({
            "timestamp": ["2026-06-01 15:25", "2026-06-02 09:15"],
            "symbol": ["ABC", "ABC"],
            "timeframe": ["5m", "5m"],
            "close": [100.0, 110.0],
        })
        result = add_directional_targets(frame, neutral_bands_pct={1: 0.20})
        self.assertTrue(result["target_1"].isna().all())

    def test_unsupported_timeframe_fails_closed(self):
        frame = pd.DataFrame({
            "timestamp": ["2026-06-01 09:15"],
            "symbol": ["ABC"],
            "timeframe": ["1h"],
            "close": [100.0],
        })
        with self.assertRaisesRegex(ValueError, "Unsupported target timeframe"):
            add_directional_targets(frame, neutral_bands_pct={1: 0.20})


if __name__ == "__main__":
    unittest.main()
