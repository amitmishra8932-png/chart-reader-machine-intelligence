import json
import unittest
from pathlib import Path

import pandas as pd

from engine.patterns import detect_patterns


class PatternRegistryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1] / "data" / "pattern_registry.json"
        self.registry = json.loads(self.path.read_text(encoding="utf-8"))

    def test_registry_has_unique_ids_and_required_research_fields(self):
        patterns = self.registry["patterns"]
        ids = [item["id"] for item in patterns]
        self.assertGreaterEqual(len(patterns), 20)
        self.assertEqual(len(ids), len(set(ids)))
        required = {
            "name", "family", "bias", "definition", "hypothesis",
            "context", "failure", "caveat", "implemented",
            "detector_status", "detector_columns", "empirical_performance",
        }
        for item in patterns:
            self.assertTrue(required.issubset(item))
            self.assertEqual(item["empirical_performance"], "not_tested")
        self.assertGreaterEqual(len(self.registry["references"]), 2)

    def test_registry_detector_mapping_matches_actual_detector_outputs(self):
        sample = pd.DataFrame({
            "open": [10.0, 11.0, 12.0, 13.0, 12.0],
            "high": [12.0, 13.0, 14.0, 14.0, 13.0],
            "low": [9.0, 10.0, 11.0, 12.0, 11.0],
            "close": [11.0, 12.0, 13.0, 13.0, 12.0],
        })
        detected = detect_patterns(sample)
        actual_columns = {
            column for column in detected.columns if column.startswith("pattern_")
        }

        mapped_columns = []
        for item in self.registry["patterns"]:
            columns = item["detector_columns"]
            mapped_columns.extend(columns)
            has_detector = bool(columns)
            self.assertEqual(item["implemented"], has_detector, item["id"])
            expected_status = "implemented" if has_detector else "catalogued_not_implemented"
            self.assertEqual(item["detector_status"], expected_status, item["id"])
            self.assertTrue(set(columns).issubset(actual_columns), item["id"])

        self.assertEqual(len(mapped_columns), len(set(mapped_columns)))
        self.assertEqual(set(mapped_columns), actual_columns)

    def test_marubozu_registry_entry_maps_both_directional_detectors(self):
        item = next(
            pattern for pattern in self.registry["patterns"]
            if pattern["id"] == "marubozu"
        )
        self.assertEqual(
            set(item["detector_columns"]),
            {"pattern_bullish_marubozu", "pattern_bearish_marubozu"},
        )


if __name__ == "__main__":
    unittest.main()
