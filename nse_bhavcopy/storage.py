"""SQLite persistence isolated to the calling repository."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import date
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
            raw_archive_path TEXT NOT NULL,
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


def _preserve_raw_archive(payload: bytes, database_path: Path, trade_date: str,
                          raw_archive_root: str | Path | None) -> Path:
    root = Path(raw_archive_root) if raw_archive_root else database_path.parent / "raw_bhavcopy"
    year, month, day = trade_date.split("-")
    destination = root / year / month / f"{day}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def ingest_bhavcopy(payload: bytes, database_path: str | Path, trade_date: str,
                    source_url: str, source_filename: str,
                    raw_archive_root: str | Path | None = None) -> dict:
    """Preserve the exact source ZIP, then validate and atomically upsert its rows."""
    parsed_date = date.fromisoformat(trade_date)
    if parsed_date.isoformat() != trade_date:
        raise ValueError("trade_date must be YYYY-MM-DD")
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    raw_path = _preserve_raw_archive(payload, path, trade_date, raw_archive_root)
    rows, quarantined = parse_bhavcopy_zip(payload, trade_date)
    initialize_database(path)
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
            (trade_date,source_url,source_filename,raw_archive_path,source_sha256,
             parser_version,valid_rows,quarantined_rows,status)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (trade_date, source_url, source_filename, str(raw_path), digest,
             PARSER_VERSION, len(rows), len(quarantined), status),
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
        "status": status, "database_path": str(path), "raw_archive_path": str(raw_path),
    }
