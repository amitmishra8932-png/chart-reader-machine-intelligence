import unittest
from unittest.mock import patch

import pandas as pd

from engine.ai_replay import replay_ai_decisions


def session_bars(freq, periods, start="2026-11-02 09:15"):
    ts = pd.date_range(start, periods=periods, freq=freq, tz="Asia/Kolkata")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * periods,
        "high": [101.0] * periods,
        "low": [99.0] * periods,
        "close": [100.0] * periods,
        "volume": [10000] * periods,
    })


class AIReplayTests(unittest.TestCase):
    def test_next_bar_fill_and_conservative_stop_first_resolution(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 26)
        # The first eligible entry occurs after 21 completed 15m bars.
        # The first signal waits for 21 completed 15m context bars.
        five.loc[64:, "high"] = 105.0
        five.loc[64:, "low"] = 97.0
        decision = {
            "decision": "LONG", "score": 9, "stop_loss": 98.0, "target": 104.0,
        }
        with patch("engine.ai_replay.analyse_trade_setup", return_value=decision):
            report = replay_ai_decisions(
                five, fifteen, symbol="TEST",
                transaction_cost_bps=10, slippage_bps=5,
            )
        self.assertGreater(report["trade_count"], 0)
        first = report["trades"][0]
        self.assertEqual(first["exit_reason"], "STOP")
        self.assertLess(first["exit_price"], 100.0)
        self.assertEqual(first["bars_held"], 1)

    def test_rejects_timezone_naive_replay_data(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 26)
        five["timestamp"] = five["timestamp"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_invalid_cost_assumptions(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 26)
        with self.assertRaisesRegex(ValueError, "slippage_bps"):
            replay_ai_decisions(five, fifteen, symbol="TEST", slippage_bps=-1)


if __name__ == "__main__":
    unittest.main()
