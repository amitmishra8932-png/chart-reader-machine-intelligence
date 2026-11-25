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
        result = analyse_trade_setup(make_bars(30), make_bars(30, "15min"), symbol="AAA")
        self.assertEqual(result["decision"], "NO TRADE")
        self.assertIn("55", result["reason"])

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

    def test_ml_training_fails_closed_on_insufficient_history(self):
        frame = make_bars(90).assign(symbol="AAA", timeframe="5m")
        model, report = train_direction_model(frame)
        self.assertIsNone(model)
        self.assertFalse(report["approved"])


if __name__ == "__main__":
    unittest.main()
