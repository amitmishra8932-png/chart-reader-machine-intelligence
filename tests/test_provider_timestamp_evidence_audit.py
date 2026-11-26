import unittest

import pandas as pd

from scripts.audit_provider_timestamp_evidence import audit_frame


def sample_frame():
    rows = []
    base = pd.Timestamp("2026-11-02 09:15:00")
    five = [
        (100.0, 101.0, 99.0, 100.5, 10),
        (100.5, 102.0, 100.0, 101.5, 20),
        (101.5, 103.0, 101.0, 102.5, 30),
    ]
    for index, values in enumerate(five):
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "provider_timestamp_raw": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "open": values[0], "high": values[1], "low": values[2],
            "close": values[3], "volume": values[4],
            "symbol": "TEST", "timeframe": "5m",
        })
    rows.append({
        "timestamp": base.strftime("%Y-%m-%d %H:%M:%S"),
        "provider_timestamp_raw": base.strftime("%Y-%m-%dT%H:%M:%S"),
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": 102.5, "volume": 60,
        "symbol": "TEST", "timeframe": "15m",
    })
    return pd.DataFrame(rows)


class ProviderTimestampEvidenceAuditTests(unittest.TestCase):
    def test_naive_provider_timestamps_are_not_promoted_to_verified_timezone(self):
        audited, report = audit_frame(sample_frame())
        self.assertEqual(report["timezone_evidence"], "unverified_naive")
        self.assertEqual(report["replay_timestamp_gate"], "blocked")
        self.assertTrue(any("timezone offsets" in item for item in report["blockers"]))
        self.assertEqual(report["five_to_fifteen_minute_comparison"]["match"], 1)

    def test_explicit_offsets_are_detected_but_session_semantics_remain_unverified(self):
        frame = sample_frame()
        frame["timestamp"] = frame["timestamp"] + "+05:30"
        frame["provider_timestamp_raw"] = frame["provider_timestamp_raw"] + "+05:30"
        audited, report = audit_frame(frame)
        self.assertEqual(report["timezone_evidence"], "explicit_offset_present")
        self.assertEqual(report["rows_with_explicit_offset"], 4)
        self.assertEqual(report["replay_timestamp_gate"], "pass")
        self.assertTrue(any("candle starts or ends" in item for item in report["limitations"]))

    def test_flags_post_continuous_session_rows_without_calling_them_cas(self):
        frame = sample_frame()
        extra = frame.iloc[[0]].copy()
        extra["timestamp"] = "2026-11-02 15:15:00"
        extra["provider_timestamp_raw"] = "2026-11-02T15:15:00"
        extra["timeframe"] = "5m"
        frame = pd.concat([frame, extra], ignore_index=True)
        audited, report = audit_frame(frame)
        row = audited.iloc[-1]
        self.assertTrue(row["outside_regular_continuous_session_candidate"])
        self.assertEqual(row["row_audit_status"], "outside_regular_session_review")

    def test_special_session_date_is_flagged_for_review(self):
        frame = sample_frame()
        frame.loc[0, "timestamp"] = "2026-11-08 14:20:00"
        frame.loc[0, "provider_timestamp_raw"] = "2026-11-08T14:20:00"
        audited, report = audit_frame(frame, {"2026-11-08"})
        self.assertTrue(audited.loc[0, "special_session_candidate"])
        self.assertEqual(audited.loc[0, "row_audit_status"], "special_session_review")

    def test_rejects_missing_required_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            audit_frame(sample_frame().drop(columns=["provider_timestamp_raw"]))


if __name__ == "__main__":
    unittest.main()
