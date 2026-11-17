"""Tests for Phase 8.12 session coverage and quarantine flags."""
import unittest
import pandas as pd
from engine.ml_data_integrity import audit_session_coverage, classify_quarantine


def rows(times, timeframe="5m", date="2026-01-05"):
    return pd.DataFrame([
        {"symbol":"AAA","timeframe":timeframe,"timestamp":f"{date} {t}",
         "open":100,"high":102,"low":99,"close":101,"volume":10,
         "regular_signal_eligible":True}
        for t in times
    ])


class DataIntegrityTests(unittest.TestCase):
    def test_complete_5m_grid(self):
        times = pd.date_range("09:15", "15:25", freq="5min").strftime("%H:%M")
        sessions, candidate, manifest = audit_session_coverage(rows(times), exclude_dates=())
        self.assertEqual(sessions.iloc[0]["session_quality"], "complete")
        self.assertEqual(len(candidate), 75)
        self.assertEqual(manifest["session_groups"], 1)

    def test_closing_gap_is_labelled_not_dropped(self):
        times = pd.date_range("09:15", "15:15", freq="5min").strftime("%H:%M")
        sessions, candidate, _ = audit_session_coverage(rows(times), exclude_dates=())
        self.assertEqual(sessions.iloc[0]["session_quality"], "closing_bars_missing_only")
        self.assertEqual(sessions.iloc[0]["missing_times"], "15:20;15:25")
        self.assertEqual(len(candidate), 73)

    def test_intraday_gap_is_distinguished(self):
        times = list(pd.date_range("09:15", "15:25", freq="5min").strftime("%H:%M"))
        times.remove("10:30")
        sessions, _, _ = audit_session_coverage(rows(times), exclude_dates=())
        self.assertEqual(sessions.iloc[0]["session_quality"], "intraday_gaps")

    def test_off_grid_timestamp_is_flagged(self):
        data = rows(["09:15", "09:20", "09:21"])
        sessions, candidate, _ = audit_session_coverage(data, exclude_dates=())
        self.assertIn("09:21", sessions.iloc[0]["off_grid_times"])
        self.assertFalse(bool(candidate.loc[candidate["bar_time"]=="09:21","on_expected_grid"].iloc[0]))

    def test_partial_latest_date_can_be_excluded(self):
        data = pd.concat([rows(["09:15"]), rows(["09:15"], date="2026-01-06")])
        sessions, candidate, _ = audit_session_coverage(data, exclude_dates=("2026-01-06",))
        self.assertEqual(candidate["session_date"].nunique(), 1)

    def test_quarantine_never_promoted(self):
        q = rows(["15:30"])
        q.loc[:, ["open","high","low","close"]] = 100
        out = classify_quarantine(q)
        self.assertTrue(bool(out.iloc[0]["is_1530_timestamp"]))
        self.assertTrue(bool(out.iloc[0]["flat_ohlc"]))
        self.assertEqual(out.iloc[0]["promotion_status"], "remain_quarantined")


if __name__ == "__main__":
    unittest.main()
