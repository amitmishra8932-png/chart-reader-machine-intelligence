import unittest

import pandas as pd

from engine.patterns import detect_patterns


class PatternBoundaryTests(unittest.TestCase):
    @staticmethod
    def candles(rows, timestamps, symbols=None, timeframes=None, index=None):
        frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(timestamps)
        if symbols is not None:
            frame["symbol"] = symbols
        if timeframes is not None:
            frame["timeframe"] = timeframes
        if index is not None:
            frame.index = index
        return frame

    def test_interleaved_symbols_do_not_share_previous_candle(self):
        frame = self.candles(
            [
                [8.0, 9.2, 7.8, 9.0],    # AAA bullish
                [10.0, 10.2, 7.8, 8.0],  # BBB bearish
                [8.0, 11.2, 7.8, 11.0],  # AAA bullish; not engulfing AAA's prior candle
            ],
            ["2026-01-05 09:15", "2026-01-05 09:15", "2026-01-05 09:20"],
            symbols=["AAA", "BBB", "AAA"],
            timeframes=["5m", "5m", "5m"],
        )
        result = detect_patterns(frame)
        self.assertFalse(result.loc[2, "pattern_bullish_engulfing"])
        self.assertFalse(result.loc[1, "pattern_bearish_engulfing"])

    def test_interleaved_timeframes_do_not_share_previous_candle(self):
        frame = self.candles(
            [
                [8.0, 9.2, 7.8, 9.0],
                [10.0, 10.2, 7.8, 8.0],
                [8.0, 11.2, 7.8, 11.0],
            ],
            ["2026-01-05 09:15", "2026-01-05 09:15", "2026-01-05 09:20"],
            symbols=["AAA", "AAA", "AAA"],
            timeframes=["5m", "15m", "5m"],
        )
        result = detect_patterns(frame)
        self.assertFalse(result.loc[2, "pattern_bullish_engulfing"])
        self.assertFalse(result.loc[1, "pattern_bearish_engulfing"])

    def test_patterns_do_not_cross_trading_date(self):
        frame = self.candles(
            [
                [10.0, 10.2, 7.8, 8.0],
                [8.0, 11.2, 7.8, 11.0],
            ],
            ["2026-01-05 15:25", "2026-01-06 09:15"],
        )
        result = detect_patterns(frame)
        self.assertFalse(result["pattern_bullish_engulfing"].any())

    def test_unsorted_timestamps_are_processed_chronologically_and_original_order_is_kept(self):
        frame = self.candles(
            [
                [8.0, 11.2, 7.8, 11.0],  # later bullish candle, first in input
                [10.0, 10.2, 7.8, 8.0],  # earlier bearish candle, second in input
            ],
            ["2026-01-05 09:20", "2026-01-05 09:15"],
            symbols=["AAA", "AAA"],
            timeframes=["5m", "5m"],
            index=[8, 3],
        )
        result = detect_patterns(frame)
        self.assertEqual(result.index.tolist(), [8, 3])
        self.assertTrue(result.loc[8, "pattern_bullish_engulfing"])
        self.assertFalse(result.loc[3, "pattern_bullish_engulfing"])

    def test_single_series_without_metadata_keeps_row_sequence_behaviour(self):
        frame = self.candles(
            [
                [10.0, 10.2, 7.8, 8.0],
                [8.0, 11.2, 7.8, 11.0],
            ],
            ["2026-01-05 09:15", "2026-01-05 09:20"],
        )
        result = detect_patterns(frame)
        self.assertTrue(result.loc[1, "pattern_bullish_engulfing"])


if __name__ == "__main__":
    unittest.main()
