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
        fifteen = session_bars("15min", 25)
        # The first eligible entry occurs after 21 completed 15m bars.
        # The first signal waits for 21 completed 15m context bars.
        five.loc[63:, "high"] = 105.0
        five.loc[63:, "low"] = 97.0
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


    def test_prior_sessions_warm_up_indicators_without_using_stale_context(self):
        five = pd.concat([
            session_bars("5min", 75, start="2026-11-02 09:15"),
            session_bars("5min", 75, start="2026-11-03 09:15"),
        ], ignore_index=True)
        fifteen = pd.concat([
            session_bars("15min", 25, start="2026-11-02 09:15"),
            session_bars("15min", 25, start="2026-11-03 09:15"),
        ], ignore_index=True)
        observed = []

        def no_trade_with_observation(five_history, fifteen_history, **kwargs):
            observed.append((
                five_history["timestamp"].iloc[-1].tz_convert("Asia/Kolkata"),
                len(five_history),
                len(fifteen_history),
                fifteen_history["timestamp"].iloc[-1].tz_convert("Asia/Kolkata"),
            ))
            return {"decision": "NO TRADE"}

        with patch("engine.ai_replay.analyse_trade_setup", side_effect=no_trade_with_observation):
            report = replay_ai_decisions(five, fifteen, symbol="TEST")

        next_day_opening_context = [
            item for item in observed
            if item[0].date() == pd.Timestamp("2026-11-03").date()
        ]
        self.assertTrue(next_day_opening_context)
        first = next_day_opening_context[0]
        # First two 5m bars cannot use yesterday's stale 15m context.
        self.assertEqual(first[0].strftime("%H:%M"), "09:25")
        self.assertEqual(first[3].strftime("%H:%M"), "09:15")
        self.assertGreaterEqual(first[1], 21)
        self.assertGreaterEqual(first[2], 21)
        self.assertEqual(report["trade_count"], 0)

    def test_gap_through_long_stop_fills_at_worse_open(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five.loc[64, ["open", "high", "low", "close"]] = [95.0, 97.0, 94.0, 96.0]
        decision = {
            "decision": "LONG", "score": 9, "stop_loss": 98.0, "target": 104.0,
        }
        with patch("engine.ai_replay.analyse_trade_setup", return_value=decision):
            report = replay_ai_decisions(five, fifteen, symbol="TEST", slippage_bps=5)
        first = report["trades"][0]
        self.assertEqual(first["exit_reason"], "STOP")
        self.assertLess(first["exit_price"], 95.0)

    def test_rejects_invalid_ohlc_geometry(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five.loc[0, "high"] = 98.0
        with self.assertRaisesRegex(ValueError, "OHLC candle geometry"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_timezone_naive_replay_data(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five["timestamp"] = five["timestamp"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            replay_ai_decisions(five, fifteen, symbol="TEST")


    def test_rejects_mismatched_symbol_metadata(self):
        five = session_bars("5min", 75).assign(symbol="OTHER")
        fifteen = session_bars("15min", 25).assign(symbol="TEST")
        with self.assertRaisesRegex(ValueError, "requested symbol TEST"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_mismatched_timeframe_metadata(self):
        five = session_bars("5min", 75).assign(timeframe="15m")
        fifteen = session_bars("15min", 25).assign(timeframe="15m")
        with self.assertRaisesRegex(ValueError, "mismatched timeframe"):
            replay_ai_decisions(five, fifteen, symbol="TEST")


    def test_incomplete_final_bar_cannot_create_false_session_end_exit(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five["is_complete"] = True
        five.loc[74, "is_complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete at the close"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_session_missing_opening_candle(self):
        five = session_bars("5min", 75).iloc[1:].reset_index(drop=True)
        fifteen = session_bars("15min", 25)
        with self.assertRaisesRegex(ValueError, "missing its opening"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_internal_missing_candle_after_completion_filter(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five["is_complete"] = True
        five.loc[64, "is_complete"] = False
        with self.assertRaisesRegex(ValueError, "missing or irregular"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_replay_bars_outside_regular_session(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five.loc[0, "timestamp"] = pd.Timestamp("2026-11-02 08:55", tz="Asia/Kolkata")
        with self.assertRaisesRegex(ValueError, "outside regular NSE"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_ambiguous_completion_flags(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        five["is_complete"] = pd.Series([True] * len(five), dtype=object)
        five.loc[0, "is_complete"] = "unknown"
        with self.assertRaisesRegex(ValueError, "explicit true/false"):
            replay_ai_decisions(five, fifteen, symbol="TEST")

    def test_rejects_invalid_cost_assumptions(self):
        five = session_bars("5min", 75)
        fifteen = session_bars("15min", 25)
        with self.assertRaisesRegex(ValueError, "slippage_bps"):
            replay_ai_decisions(five, fifteen, symbol="TEST", slippage_bps=-1)


if __name__ == "__main__":
    unittest.main()
