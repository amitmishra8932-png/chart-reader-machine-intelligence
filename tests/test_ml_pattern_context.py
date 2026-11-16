"""Tests for validation-only Phase 8.9 pattern/context diagnostics."""
import unittest

import pandas as pd

from engine.ml_pattern_context import analyze_pattern_context


def make_frame():
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(80):
            rows.append({
                "validation_split": split,
                "symbol": "AAA" if index % 2 == 0 else "BBB",
                "feature_pattern_hammer": int(index % 2 == 0),
                "feature_pattern_doji": int(index % 4 == 0),
                "feature_sma_20_distance_pct": 0.01 if index % 3 else -0.01,
                "feature_relative_volume_20": 2.0 if index % 5 == 0 else 1.0,
                "target_3": (index % 3) - 1,
                "target_6": (index % 3) - 1,
            })
    return pd.DataFrame(rows)


class PatternContextTests(unittest.TestCase):
    def test_reports_validation_only_overall_symbol_and_context(self):
        report = analyze_pattern_context(make_frame(), horizons=(3,), min_support=5)
        self.assertEqual(report["validation_rows"], 80)
        hammer = report["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertEqual(hammer["occurrences"], 40)
        self.assertIn("AAA", hammer["by_symbol"])
        self.assertIn("above_sma20__elevated_volume", hammer["by_context"])
        self.assertTrue(hammer["overall"]["eligible"])

    def test_small_subgroups_are_not_ranked(self):
        report = analyze_pattern_context(make_frame(), horizons=(3,), min_support=100)
        hammer = report["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertFalse(hammer["overall"]["eligible"])
        self.assertNotIn("pattern_class_rates", hammer["overall"])

    def test_missing_context_is_not_imputed(self):
        frame = make_frame()
        frame["feature_sma_20_distance_pct"] = float("nan")
        report = analyze_pattern_context(frame, horizons=(3,), min_support=5)
        hammer = report["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertEqual(hammer["by_context"], {})

    def test_rejects_non_binary_pattern_flags(self):
        frame = make_frame()
        frame.loc[frame["validation_split"] == "validation", "feature_pattern_hammer"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            analyze_pattern_context(frame, horizons=(3,))

    def test_rejects_missing_validation_or_target(self):
        with self.assertRaisesRegex(ValueError, "No validation"):
            analyze_pattern_context(make_frame().query("validation_split == 'train'"))
        with self.assertRaisesRegex(ValueError, "Missing target"):
            analyze_pattern_context(make_frame().drop(columns=["target_3"]), horizons=(3,))


if __name__ == "__main__":
    unittest.main()
