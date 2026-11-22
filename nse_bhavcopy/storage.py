"""SQLite persistence isolated to the calling repository."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .parser import PARSER_VERSION, parse_bhavcopy_zip


def initialize_database(database_path: str | Path) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS daily_bhavcopy (
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            series TEXT NOT NULL,
            business_date TEXT,
            segment TEXT,
            source TEXT,
            instrument_type TEXT,
            isin TEXT,
            instrument_name TEXT,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            last_price REAL NOT NULL,
            prev_close REAL NOT NULL,
            average_price REAL,
            traded_qty REAL NOT NULL,
            turnover REAL NOT NULL,
            trade_count REAL NOT NULL,
            raw_row_json TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            PRIMARY KEY (trade_date, symbol, series)
        );
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            valid_rows INTEGER NOT NULL,
            quarantined_rows INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS quarantined_rows (
            run_id INTEGER NOT NULL,
            trade_date TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            reason TEXT NOT NULL,
            raw_row_json TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_bhavcopy_symbol_date
            ON daily_bhavcopy(symbol, trade_date);
        """)


def ingest_bhavcopy(payload: bytes, database_path: str | Path, trade_date: str,
                    source_url: str, source_filename: str) -> dict:
    """Validate then atomically upsert records and audit the import."""
    rows, quarantined = parse_bhavcopy_zip(payload, trade_date)
    digest = hashlib.sha256(payload).hexdigest()
    initialize_database(database_path)
    path = Path(database_path)
    columns = (
        "trade_date", "symbol", "series", "business_date", "segment", "source",
        "instrument_type", "isin", "instrument_name", "open", "high", "low",
        "close", "last_price", "prev_close", "average_price", "traded_qty",
        "turnover", "trade_count", "raw_row_json", "source_sha256", "parser_version",
    )
    placeholders = ",".join("?" for _ in columns)
    updates = ",".join(f"{name}=excluded.{name}" for name in columns[3:])
    insert_sql = (
        f"INSERT INTO daily_bhavcopy ({','.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(trade_date,symbol,series) DO UPDATE SET {updates}"
    )
    status = "completed" if not quarantined else "partial_quarantined"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        cursor = connection.execute(
            """INSERT INTO ingestion_runs
            (trade_date,source_url,source_filename,source_sha256,parser_version,
             valid_rows,quarantined_rows,status)
            VALUES (?,?,?,?,?,?,?,?)""",
            (trade_date, source_url, source_filename, digest, PARSER_VERSION,
             len(rows), len(quarantined), status),
        )
        run_id = cursor.lastrowid
        connection.executemany(insert_sql, [
            tuple(row.get(name) for name in columns[:20]) + (digest, PARSER_VERSION)
            for row in rows
        ])
        connection.executemany(
            """INSERT INTO quarantined_rows
            (run_id,trade_date,line_number,reason,raw_row_json) VALUES (?,?,?,?,?)""",
            [(run_id, trade_date, item["line_number"], item["reason"], item["raw_row_json"])
             for item in quarantined],
        )
    return {
        "run_id": run_id, "trade_date": trade_date, "sha256": digest,
        "valid_rows": len(rows), "quarantined_rows": len(quarantined),
        "status": status, "database_path": str(path),
    }
