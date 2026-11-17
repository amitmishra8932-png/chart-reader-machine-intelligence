"""Tests for leakage-safe Phase 8.11 event alignment."""
import unittest
import pandas as pd
from engine.ml_event_alignment import prepare_phase_8_11


def frame(rows, timeframe):
    return pd.DataFrame([
        {"timestamp": ts, "open": 100, "high": 102, "low": 99, "close": 101,
         "volume": 10, "symbol": "AAA", "timeframe": timeframe,
         "regular_signal_eligible": True}
        for ts in rows
    ])


class EventAlignmentTests(unittest.TestCase):
    def test_only_closed_higher_timeframe_context_is_joined(self):
        hist = pd.concat([
            frame(["2026-01-05 09:15", "2026-01-05 09:20", "2026-01-05 09:25", "2026-01-05 09:30"], "5m"),
            frame(["2026-01-05 09:15", "2026-01-05 09:30"], "15m"),
        ], ignore_index=True)
        reg = hist.copy()
        events, _, _ = prepare_phase_8_11(hist, reg, exclude_dates=())
        at_0915 = events.loc[events.timestamp == pd.Timestamp("2026-01-05 09:15")].iloc[0]
        self.assertFalse(bool(at_0915["htf_context_available"]))
        # At 09:20/09:25, the 09:15 15m candle is still forming.
        self.assertTrue(events.loc[events.timestamp == pd.Timestamp("2026-01-05 09:20"), "htf_context_available"].eq(False).all())
        self.assertTrue(events.loc[events.timestamp == pd.Timestamp("2026-01-05 09:25"), "htf_context_available"].eq(False).all())
        # At 09:30, the 09:15-09:30 15m candle has just closed.
        row = events.loc[events.timestamp == pd.Timestamp("2026-01-05 09:30")].iloc[0]
        self.assertTrue(bool(row["htf_context_available"]))
        self.assertEqual(row["htf_bar_start"], pd.Timestamp("2026-01-05 09:15"))
        self.assertEqual(row["htf_available_at"], pd.Timestamp("2026-01-05 09:30"))

    def test_quarantines_historical_only_rows_and_excludes_partial_date(self):
        hist = pd.concat([
            frame(["2026-01-05 09:15", "2026-01-05 09:20"], "5m"),
            frame(["2026-01-05 09:15"], "15m"),
            frame(["2026-01-06 09:15"], "5m"),
        ], ignore_index=True)
        reg = hist.iloc[:3].copy()
        events, quarantine, manifest = prepare_phase_8_11(
            hist, reg, exclude_dates=("2026-01-06",)
        )
        self.assertEqual(len(quarantine), 1)
        self.assertEqual(len(events), 2)
        self.assertEqual(manifest["historical_only_rows_quarantined"], 1)

    def test_rejects_duplicate_keys(self):
        hist = pd.concat([frame(["2026-01-05 09:15"], "5m")] * 2, ignore_index=True)
        reg = frame(["2026-01-05 09:15"], "5m")
        reg = pd.concat([reg, frame(["2026-01-05 09:15"], "15m")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_phase_8_11(hist, reg, exclude_dates=())

    def test_requires_both_timeframes(self):
        hist = frame(["2026-01-05 09:15"], "5m")
        reg = hist.copy()
        with self.assertRaisesRegex(ValueError, "both 5m and 15m"):
            prepare_phase_8_11(hist, reg, exclude_dates=())


if __name__ == "__main__":
    unittest.main()
