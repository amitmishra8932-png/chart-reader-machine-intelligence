"""Tests for Phase 8.10 cross-timeframe robustness diagnostics."""
import unittest

from engine.ml_cross_timeframe import analyze_cross_timeframe_robustness


def make_report(timeframe, offset=0.0):
    patterns = {}
    for name in ("feature_pattern_hammer", "feature_pattern_doji"):
        comparison = {
            "eligible": True,
            "rate_difference_pattern_minus_comparison": {
                "-1": -0.04, "0": 0.0, "1": 0.08 + offset
            },
        }
        by_symbol = {
            symbol: comparison for symbol in ("AAA", "BBB")
        }
        by_context = {"above_sma20__elevated_volume": comparison}
        patterns[name] = {
            "occurrences": 60,
            "overall": comparison,
            "by_symbol": by_symbol,
            "by_context": by_context,
        }
    return {
        "symbols": ["AAA", "BBB"],
        "pattern_columns": list(patterns),
        "horizons": {"3": {"patterns": patterns}},
        "dataset_summary": {
            "timeframes": [timeframe],
            "neutral_bands_pct": {"3": 0.2},
        },
    }


class CrossTimeframeTests(unittest.TestCase):
    def test_same_direction_is_reported(self):
        result = analyze_cross_timeframe_robustness(make_report("5m"), make_report("15m"))
        hammer = result["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertEqual(hammer["overall"]["status"], "same_direction")
        self.assertEqual(hammer["per_symbol"]["same_direction_symbols"], 2)

    def test_direction_conflict_is_reported(self):
        result = analyze_cross_timeframe_robustness(
            make_report("5m"), make_report("15m", offset=-0.20)
        )
        hammer = result["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertEqual(hammer["overall"]["status"], "direction_conflict")

    def test_weak_direction_is_not_mislabeled_as_conflict(self):
        five, fifteen = make_report("5m"), make_report("15m")
        comparison = fifteen["horizons"]["3"]["patterns"]["feature_pattern_hammer"]["overall"]
        comparison["rate_difference_pattern_minus_comparison"] = {
            "-1": 0.0, "0": 0.0, "1": 0.0
        }
        result = analyze_cross_timeframe_robustness(five, fifteen)
        hammer = result["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertEqual(hammer["overall"]["status"], "no_clear_direction")

    def test_insufficient_support_is_not_ranked(self):
        five, fifteen = make_report("5m"), make_report("15m")
        five["horizons"]["3"]["patterns"]["feature_pattern_hammer"]["overall"]["eligible"] = False
        result = analyze_cross_timeframe_robustness(five, fifteen)
        hammer = result["horizons"]["3"]["patterns"]["feature_pattern_hammer"]
        self.assertEqual(hammer["overall"]["status"], "insufficient_support")

    def test_rejects_wrong_timeframe_order(self):
        with self.assertRaisesRegex(ValueError, "5m report first"):
            analyze_cross_timeframe_robustness(make_report("15m"), make_report("5m"))

    def test_rejects_mismatched_neutral_bands(self):
        five, fifteen = make_report("5m"), make_report("15m")
        fifteen["dataset_summary"]["neutral_bands_pct"]["3"] = 0.3
        with self.assertRaisesRegex(ValueError, "identical neutral-band"):
            analyze_cross_timeframe_robustness(five, fifteen)


if __name__ == "__main__":
    unittest.main()
