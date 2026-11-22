"""Bounded downloader for the official NSE CM UDiFF final Bhavcopy archive."""
from __future__ import annotations

from datetime import date
from urllib.request import Request, urlopen

MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
ARCHIVE_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"


def bhavcopy_url(trade_date: str) -> str:
    parsed = date.fromisoformat(trade_date)
    return ARCHIVE_URL.format(date=parsed.strftime("%Y%m%d"))


def download_bhavcopy(trade_date: str, timeout: int = 25) -> tuple[bytes, str]:
    """Download one final report. Absence/HTTP errors are surfaced to the caller."""
    url = bhavcopy_url(trade_date)
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; NSE-Bhavcopy-Research/1.0)",
        "Accept": "application/zip,application/octet-stream,*/*",
        "Referer": "https://www.nseindia.com/all-reports",
    })
    with urlopen(request, timeout=timeout) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("NSE archive exceeds configured download size limit")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("NSE archive is empty or exceeds configured download size limit")
    return payload, url
