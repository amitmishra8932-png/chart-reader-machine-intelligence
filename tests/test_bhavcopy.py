"""Fixture-driven tests for the isolated NSE Bhavcopy ingestion package."""
import csv
import io
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from nse_bhavcopy.parser import BhavcopyFormatError, parse_bhavcopy_zip
from nse_bhavcopy.storage import ingest_bhavcopy

HEADERS = [
    "TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "ISIN", "TckrSymb",
    "SctySrs", "FinInstrmNm", "OpnPric", "HghPric", "LwPric", "ClsPric",
    "LastPric", "PrvsClsgPric", "AvrgPric", "TtlTradgVol", "TtlTrfVal",
    "TtlNbOfTxsExctd",
]


def fixture_row():
    return {
        "TradDt": "2026-11-20", "BizDt": "2026-11-20", "Sgmt": "CM", "Src": "NSE",
        "FinInstrmTp": "STK", "ISIN": "INE002A01018", "TckrSymb": "RELIANCE",
        "SctySrs": "EQ", "FinInstrmNm": "RELIANCE INDUSTRIES",
        "OpnPric": "100", "HghPric": "110", "LwPric": "95", "ClsPric": "105",
        "LastPric": "105", "PrvsClsgPric": "99", "AvrgPric": "103",
        "TtlTradgVol": "1000", "TtlTrfVal": "103000", "TtlNbOfTxsExctd": "100",
    }


def fixture_zip(trade_date="2026-11-20", rows=None):
    if rows is None:
        rows = [fixture_row()]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=HEADERS)
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    filename = f"BhavCopy_NSE_CM_0_0_0_{trade_date.replace('-', '')}_F_0000.csv"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, text.getvalue())
    return output.getvalue()


class BhavcopyParserTests(unittest.TestCase):
    def test_valid_udiff_row_normalizes(self):
        rows, bad = parse_bhavcopy_zip(fixture_zip(), "2026-11-20")
        self.assertEqual(len(rows), 1)
        self.assertEqual(bad, [])
        self.assertEqual(rows[0]["symbol"], "RELIANCE")
        self.assertEqual(rows[0]["traded_qty"], 1000)
        self.assertEqual(rows[0]["average_price"], 103)

    def test_date_mismatch_rejected(self):
        with self.assertRaises(BhavcopyFormatError):
            parse_bhavcopy_zip(fixture_zip(), "2026-11-19")

    def test_invalid_ohlc_quarantined(self):
        row = fixture_row()
        row["HghPric"] = "90"
        rows, bad = parse_bhavcopy_zip(fixture_zip(rows=[row]), "2026-11-20")
        self.assertEqual(rows, [])
        self.assertEqual(len(bad), 1)

    def test_not_a_zip_rejected(self):
        with self.assertRaises(BhavcopyFormatError):
            parse_bhavcopy_zip(b"not zip", "2026-11-20")

    def test_duplicate_source_key_is_quarantined(self):
        rows, bad = parse_bhavcopy_zip(
            fixture_zip(rows=[fixture_row(), fixture_row()]), "2026-11-20"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(bad), 1)
        self.assertIn("duplicate", bad[0]["reason"])


class BhavcopyStorageTests(unittest.TestCase):
    def test_reimport_is_idempotent_and_raw_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bhavcopy.sqlite3"
            payload = fixture_zip()
            first = ingest_bhavcopy(payload, database, "2026-11-20", "https://nse.test/file.zip", "file.zip")
            second = ingest_bhavcopy(payload, database, "2026-11-20", "https://nse.test/file.zip", "file.zip")
            self.assertEqual(first["valid_rows"], 1)
            self.assertEqual(second["valid_rows"], 1)
            self.assertTrue(Path(first["raw_archive_path"]).exists())
            self.assertEqual(Path(first["raw_archive_path"]).read_bytes(), payload)
            with sqlite3.connect(database) as connection:
                count = connection.execute("SELECT COUNT(*) FROM daily_bhavcopy").fetchone()[0]
                runs = connection.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0]
            self.assertEqual(count, 1)
            self.assertEqual(runs, 2)

    def test_invalid_download_is_preserved_for_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bhavcopy.sqlite3"
            payload = b"not a zip"
            with self.assertRaises(BhavcopyFormatError):
                ingest_bhavcopy(payload, database, "2026-11-20", "https://nse.test/file.zip", "file.zip")
            raw_path = Path(directory) / "raw_bhavcopy" / "2026" / "11" / "20.zip"
            self.assertTrue(raw_path.exists())
            self.assertEqual(raw_path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
