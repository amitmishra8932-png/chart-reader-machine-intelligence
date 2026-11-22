"""Application-facing NSE Bhavcopy ingestion service."""
from pathlib import Path
from urllib.parse import urlparse

from .downloader import download_bhavcopy
from .storage import ingest_bhavcopy


def ingest_trade_date(trade_date: str, database_path: str | Path,
                      raw_archive_root: str | Path | None = None) -> dict:
    payload, source_url = download_bhavcopy(trade_date)
    filename = Path(urlparse(source_url).path).name
    return ingest_bhavcopy(
        payload=payload,
        database_path=database_path,
        trade_date=trade_date,
        source_url=source_url,
        source_filename=filename,
        raw_archive_root=raw_archive_root,
    )
