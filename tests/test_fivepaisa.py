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
    def test_historical_fetch_uses_get_and_bearer_token(self, mock_get):
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


    @patch("providers.fivepaisa.requests.get")
    def test_scrip_master_download_accepts_gzip_csv(self, mock_get):
        import gzip
        response = Mock()
        response.ok = True
        response.content = gzip.compress(
            b"Exch,ExchType,ScripCode,ScripData\nN,C,1333,HDFCBANK_EQ\n"
        )
        response.headers = {"Content-Type": "application/gzip"}
        mock_get.return_value = response
        from providers.fivepaisa import fetch_scrip_master
        master = fetch_scrip_master()
        self.assertEqual(resolve_scrip_code(master, "HDFCBANK"), 1333)

    @patch("providers.fivepaisa.requests.get")
    def test_scrip_master_download_reports_http_status(self, mock_get):
        response = Mock()
        response.ok = False
        response.status_code = 503
        response.headers = {"Content-Type": "text/plain"}
        response.text = "Service unavailable"
        mock_get.return_value = response
        from providers.fivepaisa import fetch_scrip_master
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            fetch_scrip_master()


    @patch("providers.fivepaisa.requests.get")
    def test_scrip_master_uses_official_all_segments_dump_and_filters_nse_cash(self, mock_get):
        response = Mock()
        response.ok = True
        response.content = (
            b"Exch,ExchType,ScripCode,ScripData\n"
            b"N,C,1333,HDFCBANK_EQ\n"
            b"B,C,9999,HDFCBANK_EQ\n"
            b"N,D,8888,HDFCBANK_FUT\n"
        )
        response.headers = {"Content-Type": "text/csv"}
        mock_get.return_value = response
        from providers.fivepaisa import fetch_scrip_master, SCRIP_MASTER_URL
        master = fetch_scrip_master()
        self.assertTrue(SCRIP_MASTER_URL.endswith("/ScripMaster/segment/All"))
        self.assertEqual(mock_get.call_args.args[0], SCRIP_MASTER_URL)
        self.assertEqual(resolve_scrip_code(master, "HDFCBANK"), 1333)
        self.assertEqual(len(master), 1)


    def test_scrip_master_rejects_html_response(self):
        from providers.fivepaisa import _decode_scrip_master_response
        with self.assertRaisesRegex(RuntimeError, "HTML instead of CSV"):
            _decode_scrip_master_response(b"<html>maintenance</html>")

    def test_scrip_master_reports_changed_columns(self):
        with self.assertRaisesRegex(RuntimeError, "Received columns"):
            parse_scrip_master("unexpected,columns\na,b\n")

if __name__ == "__main__":
    unittest.main()
