import unittest
import pandas as pd

from engine.data_loader import normalize_ohlcv
from engine.indicators import add_indicators
from engine.patterns import detect_patterns
from engine.market_structure import add_market_structure
from engine.outcomes import label_forward_returns, outcome_summary


def sample_frame(n=80):
    dates = pd.date_range("2025-01-01 09:15", periods=n, freq="5min")
    close = pd.Series([100 + i * 0.15 + (i % 4) * 0.08 for i in range(n)])
    open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
    return pd.DataFrame({
        "timestamp": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": 1000
    })


class EngineTests(unittest.TestCase):
    def test_normalize_sorts_and_drops_invalid(self):
        raw = sample_frame(5).iloc[::-1].copy()
        clean = normalize_ohlcv(raw)
        self.assertTrue(clean["timestamp"].is_monotonic_increasing)
        self.assertEqual(len(clean), 5)

    def test_indicators_and_structure_columns(self):
        result = add_market_structure(add_indicators(sample_frame()))
        self.assertIn("rsi_14", result)
        self.assertIn("atr_14", result)
        self.assertIn("trend_state", result)
        for column in ["swing_high_confirmed", "swing_low_confirmed", "swing_high_label", "swing_low_label", "structure_event", "structure_bias"]:
            self.assertIn(column, result)
        self.assertEqual(len(result), 80)

    def test_pattern_detector_adds_labels(self):
        result = detect_patterns(sample_frame())
        self.assertIn("detected_patterns", result)
        self.assertIn("pattern_doji", result)
        expected = [
            "pattern_long_legged_doji", "pattern_dragonfly_doji",
            "pattern_gravestone_doji", "pattern_hanging_man",
            "pattern_inverted_hammer", "pattern_piercing_line",
            "pattern_dark_cloud_cover", "pattern_morning_star",
            "pattern_evening_star", "pattern_tweezer_bottom",
            "pattern_tweezer_top", "pattern_bullish_marubozu",
            "pattern_bearish_marubozu", "pattern_spinning_top",
            "pattern_three_white_soldiers", "pattern_three_black_crows",
        ]
        for column in expected:
            self.assertIn(column, result)
            self.assertFalse(result[column].isna().any())

    def test_forward_return_and_summary(self):
        result = label_forward_returns(detect_patterns(sample_frame()), horizons=(3,))
        self.assertIn("forward_return_3", result)
        summary = outcome_summary(result, "pattern_doji", 3)
        self.assertIn("observations", summary)


if __name__ == "__main__":
    unittest.main()
