"""Tests for leakage-aware forward ML labels."""
import unittest

import pandas as pd

from engine.ml_labels import build_forward_labels


class ForwardLabelTests(unittest.TestCase):
    def make_frame(self):
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-11-02 09:15", "2026-11-02 09:20",
                "2026-11-02 09:25", "2026-11-02 09:30",
                "2026-11-02 09:35", "2026-11-02 09:40",
            ]),
            "symbol": ["RELIANCE"] * 6,
            "timeframe": ["5m"] * 6,
            "close": [100, 101, 102, 104, 105, 106],
            "atr_14": [1.0] * 6,
        })

    def test_forward_return_and_up_class(self):
        result = build_forward_labels(self.make_frame(), horizons=(2,))
        self.assertAlmostEqual(result.loc[0, "forward_return_2"], 2.0)
        self.assertEqual(int(result.loc[0, "target_direction_2"]), 1)

    def test_last_horizon_rows_are_unlabelled(self):
        result = build_forward_labels(self.make_frame(), horizons=(2,))
        self.assertTrue(pd.isna(result.loc[4, "forward_return_2"]))
        self.assertTrue(pd.isna(result.loc[5, "target_direction_2"]))

    def test_does_not_cross_trading_date(self):
        frame = self.make_frame().iloc[[0, 1, 2]].copy()
        next_day = self.make_frame().iloc[[3, 4, 5]].copy()
        next_day["timestamp"] += pd.Timedelta(days=1)
        frame = pd.concat([frame, next_day], ignore_index=True)
        result = build_forward_labels(frame, horizons=(2,))
        self.assertTrue(pd.isna(result.loc[1, "forward_return_2"]))
        self.assertTrue(pd.isna(result.loc[2, "forward_return_2"]))
        self.assertTrue(pd.isna(result.loc[4, "forward_return_2"]))
        self.assertTrue(pd.isna(result.loc[5, "forward_return_2"]))

    def test_symbols_are_isolated(self):
        frame = self.make_frame()
        other = frame.copy()
        other["symbol"] = "TCS"
        other["close"] = [200, 199, 198, 197, 196, 195]
        result = build_forward_labels(
            pd.concat([frame, other], ignore_index=True), horizons=(1,)
        )
        self.assertGreater(result.loc[0, "forward_return_1"], 0)
        self.assertLess(result.loc[6, "forward_return_1"], 0)

    def test_rejects_duplicate_session_timestamp(self):
        frame = pd.concat([self.make_frame(), self.make_frame().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "Duplicate timestamp"):
            build_forward_labels(frame, horizons=(1,))

    def test_rejects_missing_group_keys(self):
        frame = self.make_frame().drop(columns=["symbol"])
        with self.assertRaisesRegex(ValueError, "symbol"):
            build_forward_labels(frame, horizons=(1,))


if __name__ == "__main__":
    unittest.main()
