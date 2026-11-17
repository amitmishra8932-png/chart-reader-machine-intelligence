import unittest

import pandas as pd

from engine.ml_cost_aware_backtest import (
    simulate_fixed_horizon_trades,
    summarize_trades,
)


def sample_rows():
    times = pd.date_range("2026-01-05 09:15", periods=7, freq="5min")
    return pd.DataFrame({
        "timestamp": times,
        "symbol": ["ABC"] * 7,
        "session_date": ["2026-01-05"] * 7,
        "open": [100, 101, 102, 103, 104, 105, 106],
        "close": [100, 102, 103, 104, 105, 106, 107],
        "prediction": [1, 1, 1, -1, 0, 0, 0],
    })


class CostAwareBacktestTests(unittest.TestCase):
    def test_entry_next_open_exit_after_exact_full_horizon_and_cost(self):
        frame = sample_rows()
        frame.loc[3, "prediction"] = 0
        trades = simulate_fixed_horizon_trades(
            frame, horizon_bars=2, round_trip_cost_bps=10
        )
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["entry_time"], pd.Timestamp("2026-01-05 09:20"))
        self.assertEqual(trade["exit_time"], pd.Timestamp("2026-01-05 09:25"))
        self.assertEqual(trade["side"], "LONG")
        self.assertEqual(trade["entry_price"], 101)
        self.assertEqual(trade["exit_price"], 103)
        self.assertAlmostEqual(
            trade["net_return"], (103 / 101 - 1) - 0.001
        )

    def test_one_bar_horizon_exits_at_entry_candle_close(self):
        trades = simulate_fixed_horizon_trades(sample_rows(), horizon_bars=1)
        self.assertEqual(len(trades), 2)
        first = trades.iloc[0]
        self.assertEqual(first["entry_time"], pd.Timestamp("2026-01-05 09:20"))
        self.assertEqual(first["exit_time"], pd.Timestamp("2026-01-05 09:20"))
        self.assertEqual(first["exit_price"], 102)

    def test_does_not_cross_timestamp_gap_or_session(self):
        frame = sample_rows()
        frame.loc[2, "timestamp"] = pd.Timestamp("2026-01-05 09:26")
        frame.loc[3, "prediction"] = 0
        trades = simulate_fixed_horizon_trades(frame, horizon_bars=2)
        self.assertTrue(trades.empty)

    def test_non_overlapping_and_neutral_signals_skipped(self):
        trades = simulate_fixed_horizon_trades(sample_rows(), horizon_bars=2)
        self.assertEqual(len(trades), 2)
        self.assertEqual(
            trades.iloc[0]["signal_time"], pd.Timestamp("2026-01-05 09:15")
        )
        self.assertEqual(
            trades.iloc[1]["signal_time"], pd.Timestamp("2026-01-05 09:30")
        )

    def test_cost_and_prediction_validation(self):
        frame = sample_rows()
        with self.assertRaises(ValueError):
            simulate_fixed_horizon_trades(
                frame, horizon_bars=2, round_trip_cost_bps=-1
            )
        frame.loc[0, "prediction"] = 2
        with self.assertRaises(ValueError):
            simulate_fixed_horizon_trades(frame, horizon_bars=2)

    def test_summary_reports_trade_level_metrics(self):
        trades = simulate_fixed_horizon_trades(sample_rows(), horizon_bars=2)
        summary = summarize_trades(trades)
        self.assertEqual(summary["trade_count"], 2)
        self.assertEqual(summary["net_win_rate_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
