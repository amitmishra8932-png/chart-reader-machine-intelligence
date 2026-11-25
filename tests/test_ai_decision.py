"""Tests for the explainable AI trade-decision engine."""
import unittest

import numpy as np
import pandas as pd

from engine.ai_decision import (
    analyse_trade_setup, rank_trade_setups, train_direction_model,
)


def make_bars(periods=90, freq="5min", drift=0.0008, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.002, periods)
    close = 100 * np.cumprod(1 + returns)
    open_ = np.r_[close[0] * (1 - drift), close[:-1]]
    spread = np.maximum(close * 0.0015, 0.05)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 09:15", periods=periods, freq=freq, tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close,
        "volume": rng.integers(1000, 5000, periods),
    })


class DecisionEngineTests(unittest.TestCase):
    def test_insufficient_history_returns_no_trade(self):
        result = analyse_trade_setup(make_bars(10), make_bars(10, "15min"), symbol="AAA")
        self.assertEqual(result["decision"], "NO TRADE")
        self.assertIn("21", result["reason"])

    def test_rejects_invalid_candles(self):
        bad = make_bars()
        bad.loc[0, "high"] = bad.loc[0, "low"] - 1
        with self.assertRaisesRegex(ValueError, "Invalid OHLC"):
            analyse_trade_setup(bad, make_bars(90, "15min"), symbol="AAA")

    def test_duplicate_timestamps_are_rejected(self):
        bad = make_bars()
        bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyse_trade_setup(bad, make_bars(90, "15min"), symbol="AAA")

    def test_ranker_returns_only_qualified_candidates(self):
        candidates = [
            {"decision": "LONG", "status": "SETUP_CANDIDATE", "score": 8, "reward_risk": 2.0},
            {"decision": "SHORT", "status": "SETUP_CANDIDATE", "score": 9, "reward_risk": 1.9},
            {"decision": "NO TRADE", "status": "NO_SETUP", "score": 10, "reward_risk": 4.0},
        ]
        ranked = rank_trade_setups(candidates, limit=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["decision"], "SHORT")

    def test_confirmed_bullish_confluence_produces_long_with_consistent_score_cap(self):
        from unittest.mock import patch

        five = make_bars(30, "5min")
        fifteen = make_bars(30, "15min")
        fifteen["timestamp"] = pd.date_range(
            end=five["timestamp"].iloc[-1].floor("15min"), periods=30, freq="15min"
        )

        def controlled_enrich(frame):
            out = frame.copy()
            n = len(out)
            out["atr_14"] = 1.0
            out["ema_20"] = np.linspace(100.0, 109.0, n)
            out["rsi_14"] = 60.0
            out["feature_relative_volume_20"] = 1.5
            out["vwap"] = 100.0
            out["structure_event"] = ""
            out.loc[out.index[-1], "structure_event"] = "BOS_bullish"
            out.loc[out.index[-21:-1], "high"] = 109.0
            out.loc[out.index[-6:], "low"] = 109.0
            out.loc[out.index[-1], ["open", "high", "low", "close"]] = [109.5, 111.0, 109.0, 110.0]
            out.loc[out.index[-1], "close"] = 110.0
            return out

        with patch("engine.ai_decision._enrich", side_effect=controlled_enrich):
            result = analyse_trade_setup(five, fifteen, symbol="BULL")

        self.assertEqual(result["decision"], "LONG")
        self.assertEqual(result["score"], 10)
        self.assertEqual(result["max_score_without_ml"], 10)
        self.assertGreaterEqual(result["reward_risk"], 1.8)


    def test_rejects_stale_15m_context(self):
        from unittest.mock import patch

        five = make_bars(30, "5min")
        fifteen = make_bars(30, "15min")
        latest_15m = five["timestamp"].iloc[-1].floor("15min") - pd.Timedelta(minutes=30)
        fifteen["timestamp"] = pd.date_range(end=latest_15m, periods=30, freq="15min")
        with patch("engine.ai_decision._enrich", side_effect=lambda frame: frame.assign(
            atr_14=1.0, ema_20=100.0, rsi_14=50.0,
            feature_relative_volume_20=1.0, vwap=100.0, structure_event=""
        )):
            result = analyse_trade_setup(five, fifteen, symbol="STALE")
        self.assertEqual(result["decision"], "NO TRADE")
        self.assertIn("stale", result["reason"])

    def test_rejects_future_15m_context(self):
        from unittest.mock import patch

        five = make_bars(30, "5min")
        fifteen = make_bars(30, "15min")
        latest_15m = five["timestamp"].iloc[-1].floor("15min") + pd.Timedelta(minutes=15)
        fifteen["timestamp"] = pd.date_range(end=latest_15m, periods=30, freq="15min")
        with patch("engine.ai_decision._enrich", side_effect=lambda frame: frame.assign(
            atr_14=1.0, ema_20=100.0, rsi_14=50.0,
            feature_relative_volume_20=1.0, vwap=100.0, structure_event=""
        )):
            result = analyse_trade_setup(five, fifteen, symbol="FUTURE")
        self.assertEqual(result["decision"], "NO TRADE")
        self.assertIn("future", result["reason"])

    def test_confirmed_bearish_confluence_produces_short(self):
        from unittest.mock import patch

        five = make_bars(30, "5min")
        fifteen = make_bars(30, "15min")
        fifteen["timestamp"] = pd.date_range(
            end=five["timestamp"].iloc[-1].floor("15min"), periods=30, freq="15min"
        )

        def controlled_enrich(frame):
            out = frame.copy()
            n = len(out)
            out["atr_14"] = 1.0
            out["ema_20"] = np.linspace(120.0, 111.0, n)
            out["rsi_14"] = 40.0
            out["feature_relative_volume_20"] = 1.5
            out["vwap"] = 120.0
            out["structure_event"] = ""
            out.loc[out.index[-1], "structure_event"] = "BOS_bearish"
            out.loc[out.index[-21:-1], "low"] = 110.5
            out.loc[out.index[-6:], "high"] = 111.0
            out.loc[out.index[-1], ["open", "high", "low", "close"]] = [110.5, 111.0, 109.0, 110.0]
            return out

        with patch("engine.ai_decision._enrich", side_effect=controlled_enrich):
            result = analyse_trade_setup(five, fifteen, symbol="BEAR")

        self.assertEqual(result["decision"], "SHORT")
        self.assertGreaterEqual(result["reward_risk"], 1.8)
        self.assertLess(result["target"], result["entry_reference"])

    def test_ml_training_fails_closed_on_insufficient_history(self):
        frame = make_bars(90).assign(symbol="AAA", timeframe="5m")
        model, report = train_direction_model(frame)
        self.assertIsNone(model)
        self.assertFalse(report["approved"])


if __name__ == "__main__":
    unittest.main()
