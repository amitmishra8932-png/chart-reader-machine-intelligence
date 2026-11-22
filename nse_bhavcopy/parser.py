"""Strict parser for NSE Capital Market UDiFF final Bhavcopy ZIP files."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import date
from typing import Any

PARSER_VERSION = "1.0.0"
MAX_ARCHIVE_BYTES = 30 * 1024 * 1024
MAX_MEMBER_BYTES = 100 * 1024 * 1024
REQUIRED_TAGS = {
    "TradDt", "Sgmt", "Src", "TckrSymb", "SctySrs",
    "OpnPric", "HghPric", "LwPric", "ClsPric", "LastPric",
    "PrvsClsgPric", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd",
}
FIELD_MAP = {
    "trade_date": "TradDt", "business_date": "BizDt",
    "segment": "Sgmt", "source": "Src", "instrument_type": "FinInstrmTp",
    "isin": "ISIN", "symbol": "TckrSymb", "series": "SctySrs",
    "instrument_name": "FinInstrmNm", "open": "OpnPric", "high": "HghPric",
    "low": "LwPric", "close": "ClsPric", "last_price": "LastPric",
    "prev_close": "PrvsClsgPric", "average_price": "AvrgPric",
    "traded_qty": "TtlTradgVol", "turnover": "TtlTrfVal",
    "trade_count": "TtlNbOfTxsExctd",
}
NUMERIC_FIELDS = (
    "open", "high", "low", "close", "last_price", "prev_close",
    "average_price", "traded_qty", "turnover", "trade_count",
)


class BhavcopyFormatError(ValueError):
    """Raised when the source archive or UDiFF schema is not trustworthy."""


def _number(value: Any, field: str, required: bool = True):
    text = "" if value is None else str(value).strip()
    if not text:
        if required:
            raise ValueError(f"missing {field}")
        return None
    try:
        number = float(text.replace(",", ""))
    except (TypeError, ValueError):
        raise ValueError(f"invalid numeric {field}: {text!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {field}")
    return number


def _normalize(raw: dict[str, str], requested_date: str) -> dict:
    mapped = {name: (raw.get(tag) or "").strip() for name, tag in FIELD_MAP.items()}
    trade_date = mapped["trade_date"]
    try:
        parsed_date = date.fromisoformat(trade_date)
    except ValueError:
        raise ValueError(f"invalid TradDt: {trade_date!r}") from None
    if parsed_date.isoformat() != requested_date:
        raise ValueError(f"TradDt {trade_date} does not match requested date {requested_date}")
    if mapped["segment"].upper() != "CM":
        raise ValueError(f"unexpected segment {mapped['segment']!r}")
    if mapped["source"].upper() != "NSE":
        raise ValueError(f"unexpected source {mapped['source']!r}")
    if not mapped["symbol"] or not mapped["series"]:
        raise ValueError("missing symbol or series")

    for field in NUMERIC_FIELDS:
        mapped[field] = _number(mapped[field], field, required=(field != "average_price"))

    if min(mapped[k] for k in ("open", "high", "low", "close", "last_price", "prev_close")) < 0:
        raise ValueError("negative price")
    if mapped["high"] < mapped["low"]:
        raise ValueError("high is below low")
    for field in ("open", "close", "last_price"):
        if not mapped["low"] <= mapped[field] <= mapped["high"]:
            raise ValueError(f"{field} outside daily low/high")
    if mapped["traded_qty"] < 0 or mapped["turnover"] < 0 or mapped["trade_count"] < 0:
        raise ValueError("negative volume, turnover or trade count")

    mapped["symbol"] = mapped["symbol"].upper()
    mapped["series"] = mapped["series"].upper()
    mapped["instrument_type"] = mapped["instrument_type"].upper()
    mapped["raw_row_json"] = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return mapped


def parse_bhavcopy_zip(payload: bytes, requested_date: str) -> tuple[list[dict], list[dict]]:
    """Return (valid normalized rows, quarantined rows); reject unsafe/ambiguous archives."""
    try:
        expected = date.fromisoformat(requested_date).isoformat()
    except ValueError:
        raise ValueError("requested_date must be YYYY-MM-DD") from None
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise BhavcopyFormatError("empty archive payload")
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise BhavcopyFormatError("archive exceeds configured size limit")

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError):
        raise BhavcopyFormatError("download is not a valid ZIP archive") from None

    with archive:
        members = archive.infolist()
        if not members:
            raise BhavcopyFormatError("ZIP archive is empty")
        for member in members:
            if member.file_size > MAX_MEMBER_BYTES:
                raise BhavcopyFormatError("ZIP member exceeds configured size limit")
            if member.filename.startswith(("/", "\\")) or ".." in member.filename.replace("\\", "/").split("/"):
                raise BhavcopyFormatError("unsafe ZIP member path")
        csv_members = [m for m in members if not m.is_dir() and m.filename.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise BhavcopyFormatError(f"expected exactly one CSV member, found {len(csv_members)}")
        member = csv_members[0]
        if not re.fullmatch(r"BhavCopy_NSE_CM_0_0_0_\d{8}_F_0000\.csv", member.filename.split("/")[-1], re.I):
            raise BhavcopyFormatError("unexpected NSE CM final Bhavcopy member filename")
        filename_date = member.filename.split("/")[-1].split("_")[6]
        if filename_date != expected.replace("-", ""):
            raise BhavcopyFormatError("archive filename date does not match requested date")
        if member.file_size > MAX_MEMBER_BYTES:
            raise BhavcopyFormatError("CSV member exceeds configured size limit")
        try:
            text = archive.read(member).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise BhavcopyFormatError("CSV member is not UTF-8") from None

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise BhavcopyFormatError("CSV has no header")
    headers = {str(h).strip() for h in reader.fieldnames if h}
    missing = REQUIRED_TAGS - headers
    if missing:
        raise BhavcopyFormatError("missing required UDiFF tags: " + ", ".join(sorted(missing)))

    valid, quarantined = [], []
    for line_number, raw in enumerate(reader, start=2):
        if not raw or not any(str(v or "").strip() for v in raw.values()):
            continue
        try:
            valid.append(_normalize(raw, expected))
        except ValueError as exc:
            quarantined.append({
                "line_number": line_number,
                "reason": str(exc),
                "raw_row_json": json.dumps(raw, ensure_ascii=False, sort_keys=True),
            })
    if not valid and not quarantined:
        raise BhavcopyFormatError("CSV contains no data rows")
    return valid, quarantined
