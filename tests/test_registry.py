import json
import unittest
from pathlib import Path


class PatternRegistryTests(unittest.TestCase):
    def test_registry_has_unique_ids_and_required_research_fields(self):
        path = Path(__file__).resolve().parents[1] / "data" / "pattern_registry.json"
        registry = json.loads(path.read_text(encoding="utf-8"))
        patterns = registry["patterns"]
        ids = [item["id"] for item in patterns]
        self.assertGreaterEqual(len(patterns), 20)
        self.assertEqual(len(ids), len(set(ids)))
        required = {
            "name", "family", "bias", "definition", "hypothesis",
            "context", "failure", "caveat", "detector_status",
            "empirical_performance",
        }
        for item in patterns:
            self.assertTrue(required.issubset(item))
            self.assertEqual(item["empirical_performance"], "not_tested")
        self.assertGreaterEqual(len(registry["references"]), 2)


if __name__ == "__main__":
    unittest.main()
