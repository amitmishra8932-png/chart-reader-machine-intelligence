import unittest

import numpy as np
import pandas as pd

from engine.feature_validation import (
    audit_ohlcv_dataset,
    build_feature_validation_report,
    chronological_date_split,
)
from engine.multitimeframe import align_closed_higher_timeframe
from engine.features import add_features


class FeatureValidationTests(unittest.TestCase):
    @staticmethod
    def data():
        rows = []
        for day in pd.date_range("2026-01-05", periods=8, freq="B"):
            for symbol in ("AAA", "BBB"):
                for timeframe, minutes in (("5m", 5), ("15m", 15)):
                    for step in range(6):
                        close = 100.0 + step
                        rows.append({
                            "timestamp": day + pd.Timedelta(hours=9, minutes=15 + step * minutes),
                            "open": close - 0.2,
                            "high": close + 0.4,
                            "low": close - 0.4,
                            "close": close,
                            "volume": 1000 + step,
                            "symbol": symbol,
                            "timeframe": timeframe,
                        })
        return pd.DataFrame(rows)

    def test_audit_reports_structural_quality_and_group_coverage(self):
        frame = self.data()
        report = audit_ohlcv_dataset(frame)
        self.assertEqual(report["row_count"], len(frame))
        self.assertEqual(report["symbol_count"], 2)
        self.assertEqual(report["duplicate_symbol_timeframe_timestamps"], 0)
        self.assertEqual(report["invalid_ohlc_rows"], 0)
        self.assertEqual(report["negative_volume_rows"], 0)
        self.assertEqual(len(report["rows_by_symbol_timeframe"]), 4)

    def test_chronological_split_keeps_each_date_in_one_partition(self):
        frame = self.data()
        result = chronological_date_split(frame)
        self.assertEqual(set(result["validation_split"]), {"train", "validation", "test"})
        per_date = result.assign(
            date=pd.to_datetime(result["timestamp"]).dt.date
        ).groupby("date")["validation_split"].nunique()
        self.assertTrue((per_date == 1).all())
        dates = pd.to_datetime(result["timestamp"]).dt.date
        self.assertLess(
            dates[result["validation_split"] == "train"].max(),
            dates[result["validation_split"] == "validation"].min(),
        )
        self.assertLess(
            dates[result["validation_split"] == "validation"].max(),
            dates[result["validation_split"] == "test"].min(),
        )

    def test_feature_report_includes_distributions_and_explicit_limitations(self):
        frame = self.data()
        report = build_feature_validation_report(frame)
        self.assertIn("feature_body_to_range", report["feature_distributions"])
        self.assertIn("chronological_split", report)
        self.assertTrue(any("not a predictive-performance" in item for item in report["limitations"]))

    def test_date_split_rejects_invalid_timestamps(self):
        frame = self.data()
        frame["timestamp"] = frame["timestamp"].astype("object")
        frame.loc[0, "timestamp"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "Invalid timestamps"):
            chronological_date_split(frame)


class LongHorizonFeatureTests(unittest.TestCase):
    def test_sma50_and_ema50_warm_across_sessions_without_mixing_symbols(self):
        rows = []
        for symbol in ("AAA", "BBB"):
            for day in pd.date_range("2026-01-05", periods=3, freq="B"):
                for step in range(25):
                    close = 100.0 + step + (0 if symbol == "AAA" else 50)
                    rows.append({
                        "timestamp": day + pd.Timedelta(hours=9, minutes=15 + step * 15),
                        "open": close - 0.2,
                        "high": close + 0.5,
                        "low": close - 0.5,
                        "close": close,
                        "volume": 1000 + step,
                        "symbol": symbol,
                        "timeframe": "15m",
                    })
        frame = pd.DataFrame(rows)
        result = add_features(frame)
        first_symbol = result[result["symbol"] == "AAA"].reset_index(drop=True)
        second_symbol = result[result["symbol"] == "BBB"].reset_index(drop=True)
        self.assertTrue(pd.isna(first_symbol.loc[48, "feature_sma_50_distance_pct"]))
        self.assertTrue(pd.notna(first_symbol.loc[49, "feature_sma_50_distance_pct"]))
        self.assertTrue(pd.notna(first_symbol.loc[49, "feature_ema_50_distance_pct"]))
        self.assertTrue(pd.isna(second_symbol.loc[48, "feature_sma_50_distance_pct"]))
        self.assertTrue(pd.notna(second_symbol.loc[49, "feature_sma_50_distance_pct"]))
        # Intraday returns still reset at the new session boundary.
        self.assertTrue(pd.isna(first_symbol.loc[25, "feature_return_1"]))


class ClosedCandleAlignmentTests(unittest.TestCase):
    @staticmethod
    def frames():
        base = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-05 09:15", "2026-01-05 09:20",
                "2026-01-05 09:25", "2026-01-05 09:30",
                "2026-01-06 09:15", "2026-01-06 09:25",
            ]),
            "symbol": ["AAA"] * 6,
            "close": [100, 101, 102, 103, 104, 105],
        }, index=[10, 11, 12, 13, 14, 15])
        higher = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-05 09:15", "2026-01-05 09:30",
                "2026-01-06 09:15",
            ]),
            "symbol": ["AAA"] * 3,
            "close": [200, 210, 220],
            "volume": [2000, 2100, 2200],
        })
        return base, higher

    def test_start_labeled_higher_bars_only_appear_after_their_close(self):
        base, higher = self.frames()
        result = align_closed_higher_timeframe(
            base, higher,
            base_timestamp_label="start",
            higher_timestamp_label="start",
        )
        self.assertTrue(pd.isna(result.loc[10, "context_15m_close"]))
        self.assertTrue(pd.isna(result.loc[11, "context_15m_close"]))
        self.assertEqual(result.loc[12, "context_15m_close"], 200)
        self.assertEqual(result.loc[13, "context_15m_close"], 200)
        self.assertTrue(pd.isna(result.loc[14, "context_15m_close"]))
        self.assertEqual(result.loc[15, "context_15m_close"], 220)

    def test_end_labeled_bars_match_only_at_or_before_base_close(self):
        base, higher = self.frames()
        result = align_closed_higher_timeframe(
            base, higher,
            base_timestamp_label="end",
            higher_timestamp_label="end",
        )
        self.assertEqual(result.loc[10, "context_15m_close"], 200)
        self.assertEqual(result.loc[11, "context_15m_close"], 200)
        self.assertEqual(result.loc[12, "context_15m_close"], 200)
        self.assertEqual(result.loc[13, "context_15m_close"], 210)
        self.assertEqual(result.loc[14, "context_15m_close"], 220)
        self.assertEqual(result.loc[15, "context_15m_close"], 220)

    def test_does_not_carry_context_across_session(self):
        base, higher = self.frames()
        higher = higher.iloc[:2].copy()
        result = align_closed_higher_timeframe(
            base, higher,
            base_timestamp_label="start",
            higher_timestamp_label="start",
        )
        self.assertTrue(pd.isna(result.loc[14, "context_15m_close"]))
        self.assertTrue(pd.isna(result.loc[15, "context_15m_close"]))

    def test_does_not_mix_symbols(self):
        base, higher = self.frames()
        base.loc[12, "symbol"] = "BBB"
        result = align_closed_higher_timeframe(
            base, higher,
            base_timestamp_label="start",
            higher_timestamp_label="start",
        )
        self.assertTrue(pd.isna(result.loc[12, "context_15m_close"]))

    def test_requires_explicit_timestamp_semantics(self):
        base, higher = self.frames()
        with self.assertRaisesRegex(ValueError, "base_timestamp_label"):
            align_closed_higher_timeframe(
                base, higher,
                base_timestamp_label="unknown",
                higher_timestamp_label="start",
            )

    def test_preserves_base_row_order_and_index(self):
        base, higher = self.frames()
        result = align_closed_higher_timeframe(
            base.iloc[[3, 0, 4, 1, 2]], higher,
            base_timestamp_label="start",
            higher_timestamp_label="start",
        )
        self.assertEqual(result.index.tolist(), [13, 10, 14, 11, 12])


if __name__ == "__main__":
    unittest.main()
