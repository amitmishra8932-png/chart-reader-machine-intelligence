import unittest
import pandas as pd

from engine.historical_validation import audit_historical_ohlcv, audit_nse_session_integrity


def candles(times, volume=None):
    frame = pd.DataFrame({
        "timestamp": times,
        "open": [100.0] * len(times),
        "high": [102.0] * len(times),
        "low": [99.0] * len(times),
        "close": [101.0] * len(times),
    })
    if volume is not None:
        frame["volume"] = volume
    return frame


class HistoricalValidationTests(unittest.TestCase):
    def test_clean_intraday_data_has_no_integrity_warnings(self):
        frame = candles(pd.date_range("2026-01-05 09:15", periods=4, freq="5min"), [10, 11, 12, 13])
        report = audit_historical_ohlcv(frame, "5m")
        self.assertEqual(report["quality_status"], "No basic integrity issues detected")
        self.assertEqual(report["estimated_missing_intraday_bars"], 0)
        self.assertEqual(report["distinct_trading_dates"], 1)

    def test_overnight_gap_is_not_counted_as_missing_intraday_bars(self):
        times = list(pd.date_range("2026-01-05 15:25", periods=2, freq="5min"))
        times += list(pd.date_range("2026-01-06 09:15", periods=2, freq="5min"))
        report = audit_historical_ohlcv(candles(times), "5m")
        self.assertEqual(report["estimated_missing_intraday_bars"], 0)

    def test_same_session_gap_is_flagged(self):
        times = ["2026-01-05 09:15", "2026-01-05 09:20", "2026-01-05 09:35"]
        report = audit_historical_ohlcv(candles(times), "5m")
        self.assertEqual(report["estimated_missing_intraday_bars"], 2)
        self.assertIn("Possible missing intraday candles detected; verify against exchange sessions.", report["warnings"])

    def test_duplicate_unsorted_invalid_and_volume_rows_are_reported(self):
        frame = candles(["2026-01-05 09:20", "2026-01-05 09:15", "2026-01-05 09:15"], [10, -1, 0])
        frame.loc[2, "high"] = 1
        report = audit_historical_ohlcv(frame, "5m")
        self.assertEqual(report["duplicate_timestamps"], 1)
        self.assertEqual(report["out_of_order_steps"], 1)
        self.assertEqual(report["invalid_ohlc_rows"], 1)
        self.assertEqual(report["negative_volume_rows"], 1)
        self.assertEqual(report["zero_volume_rows"], 1)
        self.assertEqual(report["quality_status"], "Review required")

    def test_bad_timestamps_are_counted(self):
        frame = candles(["not-a-date", "2026-01-05 09:15", "2026-01-05 09:20"])
        report = audit_historical_ohlcv(frame)
        self.assertEqual(report["invalid_timestamps"], 1)

    def test_empty_or_missing_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            audit_historical_ohlcv(pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            audit_historical_ohlcv(pd.DataFrame({"timestamp": ["2026-01-01"]}))

    def test_unsupported_interval_rejected(self):
        frame = candles(pd.date_range("2026-01-05 09:15", periods=3, freq="5min"))
        with self.assertRaisesRegex(ValueError, "Unsupported interval"):
            audit_historical_ohlcv(frame, "2m")


if __name__ == "__main__":
    unittest.main()


class NSESessionIntegrityTests(unittest.TestCase):
    def test_regular_session_aligned_candles_pass(self):
        from engine.historical_validation import audit_nse_session_integrity
        frame = candles(["2026-11-09 09:15", "2026-11-09 09:30", "2026-11-09 15:15"])
        report = audit_nse_session_integrity(frame, "15m")
        self.assertEqual(report["session_anomaly_rows"], 0)

    def test_weekend_timestamp_is_flagged(self):
        from engine.historical_validation import audit_nse_session_integrity
        frame = candles(["2026-11-08 10:00"])
        report = audit_nse_session_integrity(frame, "15m")
        self.assertEqual(report["weekend_timestamps"], 1)
        self.assertEqual(report["session_anomaly_rows"], 1)

    def test_out_of_session_and_misaligned_timestamps_are_flagged(self):
        from engine.historical_validation import audit_nse_session_integrity
        frame = candles(["2026-11-09 08:55", "2026-11-09 15:50", "2026-11-09 09:20"])
        report = audit_nse_session_integrity(frame, "15m")
        self.assertEqual(report["outside_session_timestamps"], 1)
        self.assertEqual(report["closing_session_timestamps"], 1)
        self.assertEqual(report["misaligned_timestamps"], 1)
        self.assertEqual(report["session_anomaly_rows"], 2)

    def test_supplied_holiday_dates_are_flagged(self):
        from datetime import date
        from engine.historical_validation import audit_nse_session_integrity
        frame = candles(["2026-11-09 09:15"])
        report = audit_nse_session_integrity(frame, "15m", holidays={date(2026, 11, 9)})
        self.assertEqual(report["known_holiday_timestamps"], 1)
        self.assertEqual(report["session_anomaly_rows"], 1)


    def test_muhurat_special_session_is_not_misclassified_as_weekend_closure(self):
        frame = candles(["2026-11-08 14:20", "2026-11-08 14:25", "2026-11-08 15:35"])
        report = audit_nse_session_integrity(frame, "5m")
        self.assertEqual(report["special_session_timestamps"], 3)
        self.assertEqual(report["weekend_timestamps"], 3)
        self.assertEqual(report["session_anomaly_rows"], 0)

    def test_unscheduled_weekend_candle_remains_an_anomaly(self):
        report = audit_nse_session_integrity(candles(["2026-11-15 14:20"]), "5m")
        self.assertEqual(report["session_anomaly_rows"], 1)
        self.assertEqual(report["special_session_timestamps"], 0)

    def test_non_cas_security_keeps_continuous_session_until_1530(self):
        frame = candles(["2026-11-12 15:15", "2026-11-12 15:25", "2026-11-12 15:29",
                         "2026-11-12 15:30", "2026-11-12 15:45", "2026-11-12 15:50"])
        report = audit_nse_session_integrity(frame, "5m", cas_eligible=False)
        self.assertEqual(report["regular_session_rows"], 3)
        self.assertEqual(report["closing_auction_session_timestamps"], 0)
        self.assertEqual(report["closing_transition_timestamps"], 2)
        self.assertEqual(report["closing_session_timestamps"], 1)
        self.assertEqual(report["non_regular_session_rows"], 3)

    def test_cas_security_uses_1515_to_1535_auction_window(self):
        frame = candles(["2026-11-12 15:10", "2026-11-12 15:15", "2026-11-12 15:25",
                         "2026-11-12 15:30", "2026-11-12 15:35", "2026-11-12 15:50"])
        report = audit_nse_session_integrity(frame, "5m", cas_eligible=True)
        self.assertEqual(report["regular_session_rows"], 1)
        self.assertEqual(report["closing_auction_session_timestamps"], 3)
        self.assertEqual(report["closing_transition_timestamps"], 1)
        self.assertEqual(report["closing_session_timestamps"], 1)
        self.assertEqual(report["non_regular_session_rows"], 5)
        self.assertEqual(report["cas_eligible"], True)

    def test_regular_session_misalignment_is_still_flagged(self):
        report = audit_nse_session_integrity(candles(["2026-11-12 09:17"]), "5m")
        self.assertEqual(report["misaligned_timestamps"], 1)
        self.assertEqual(report["session_anomaly_rows"], 1)
