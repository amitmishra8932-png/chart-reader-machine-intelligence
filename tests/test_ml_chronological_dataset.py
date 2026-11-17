"""Tests for strict Phase 8.13 chronological dataset preparation."""
import unittest

import numpy as np
import pandas as pd

from engine.ml_chronological_dataset import prepare_phase_8_13
from engine.ml_data_integrity import EXPECTED


def complete_candidates(days=("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")):
    rows = []
    for day_index, day in enumerate(days):
        for i, clock in enumerate(EXPECTED["5m"]):
            close = 100.0 + day_index * 2 + i * 0.03
            rows.append({
                "timestamp": f"{day} {clock}", "symbol": "AAA", "timeframe": "5m",
                "open": close - 0.02, "high": close + 0.05, "low": close - 0.05,
                "close": close, "volume": 1000 + i,
                "session_quality": "complete", "on_expected_grid": True,
                "is_weekend": False,
            })
    return pd.DataFrame(rows)


class Phase813Tests(unittest.TestCase):
    def test_only_complete_5m_sessions_enter_chronological_dataset(self):
        frame = complete_candidates()
        partial = frame.iloc[:10].copy()
        partial["session_quality"] = "intraday_gaps"
        off_grid = frame.iloc[[0]].copy()
        off_grid["timestamp"] = "2026-01-08 09:16"
        off_grid["on_expected_grid"] = False
        fifteen = frame.iloc[[0]].copy()
        fifteen["timeframe"] = "15m"
        source = pd.concat([frame, partial, off_grid, fifteen], ignore_index=True)

        result, report = prepare_phase_8_13(
            source, neutral_bands_pct={1: 0.05, 3: 0.10}
        )
        self.assertEqual(len(result), 4 * 75)
        self.assertEqual(report["selected_complete_5m_rows"], 300)
        self.assertEqual(set(result["timeframe"]), {"5m"})
        self.assertEqual(set(result["session_quality"]), {"complete"})
        self.assertTrue({"train", "validation", "test"}.issubset(set(result["validation_split"])))

    def test_calendar_dates_do_not_overlap_between_splits(self):
        result, report = prepare_phase_8_13(
            complete_candidates(), neutral_bands_pct={1: 0.05}
        )
        self.assertTrue(all(not dates for dates in report["split_date_overlap"].values()))
        dates_by_split = result.groupby("validation_split")["timestamp"].apply(
            lambda s: set(pd.to_datetime(s).dt.strftime("%Y-%m-%d"))
        )
        self.assertTrue(dates_by_split["train"].isdisjoint(dates_by_split["validation"]))
        self.assertTrue(dates_by_split["validation"].isdisjoint(dates_by_split["test"]))

    def test_rejects_a_session_claimed_complete_but_missing_bar(self):
        frame = complete_candidates()
        frame = frame[~((frame["timestamp"] == "2026-01-05 10:00"))].copy()
        with self.assertRaisesRegex(ValueError, "full 5m grid"):
            prepare_phase_8_13(frame, neutral_bands_pct={1: 0.05})

    def test_requires_explicit_neutral_bands_and_candidate_schema(self):
        with self.assertRaisesRegex(ValueError, "Explicit neutral bands"):
            prepare_phase_8_13(complete_candidates(), neutral_bands_pct={})
        with self.assertRaisesRegex(ValueError, "missing columns"):
            prepare_phase_8_13(pd.DataFrame({"timestamp": []}), neutral_bands_pct={1: 0.1})

    def test_feature_and_target_values_are_generated_without_imputation(self):
        result, report = prepare_phase_8_13(
            complete_candidates(), neutral_bands_pct={1: 0.05, 3: 0.10}
        )
        self.assertIn("feature_atr_14", result.columns)
        self.assertIn("feature_pattern_doji", result.columns)
        self.assertIn("target_1", result.columns)
        self.assertGreater(report["unlabelled_target_rows"]["target_1"], 0)
        self.assertGreater(report["feature_missing_or_nonfinite"]["feature_atr_14"], 0)
        self.assertIn("No model is fitted", " ".join(report["guardrails"]))


if __name__ == "__main__":
    unittest.main()
