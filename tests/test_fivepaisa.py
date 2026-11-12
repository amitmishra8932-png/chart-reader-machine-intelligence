import unittest
import pandas as pd
from unittest.mock import Mock, patch

from providers.fivepaisa import (
    build_oauth_url, parse_historical_candles, parse_scrip_master,
    resolve_scrip_code, fetch_historical_candles,
)


class FivePaisaAdapterTests(unittest.TestCase):
    def test_oauth_url_encodes_callback(self):
        url = build_oauth_url("test-key", "https://example.streamlit.app/")
        self.assertIn("VendorKey=test-key", url)
        self.assertIn("ResponseURL=https%3A%2F%2Fexample.streamlit.app%2F", url)

    def test_scrip_master_resolves_equity_symbol(self):
        csv_text = (
            "Exch,ExchType,ScripCode,ScripData\n"
            "N,C,2885,ITC_EQ\n"
            "B,C,500875,ITC_EQ\n"
            "N,D,9999,ITC_FUT\n"
        )
        master = parse_scrip_master(csv_text)
        self.assertEqual(resolve_scrip_code(master, "ITC"), 2885)

    @patch("providers.fivepaisa.requests.get")
    def test_historical_fetch_uses_get_and_client_code_header(self, mock_get):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "status": "success",
            "data": {"candles": [["2026-11-10T09:15:00", 100, 105, 99, 103, 1200]]},
        }
        mock_get.return_value = response
        frame = fetch_historical_candles(
            "test-access-token", 2885, "5m", "2026-11-10", "2026-11-12",
            client_code="TESTCLIENT",
        )
        self.assertEqual(len(frame), 1)
        args, kwargs = mock_get.call_args
        self.assertIn("/V2/historical/N/C/2885/5m", args[0])
        self.assertEqual(kwargs["headers"]["x-clientcode"], "TESTCLIENT")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-access-token")

    def test_historical_candle_payload_normalizes_columns(self):
        payload = {
            "status": "success",
            "data": {"candles": [
                ["2026-11-10T09:15:00", 100, 105, 99, 103, 1200],
                ["2026-11-10T09:20:00", 103, 106, 102, 105, 1500],
            ]},
        }
        frame = parse_historical_candles(payload)
        self.assertEqual(list(frame.columns), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["close"], 103)


if __name__ == "__main__":
    unittest.main()
