#!/usr/bin/env python3
"""Audit exported 5paisa OHLCV candle CSVs without modifying source files.

Usage:
  python tools/live_data_quality_audit.py path/to/5m.csv path/to/15m.csv

The script checks schema, missing values, duplicate symbol/timestamp keys, OHLC
relationships, negative volume, completion flags, per-symbol cadence/gaps,
identical records in overlapping exports, and 5m -> 15m OHLCV reconstruction.
Timestamps are parsed as ISO-8601; timezone-aware values are normalized to UTC.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

REQUIRED = ("timestamp", "symbol", "open", "high", "low", "close", "volume", "is_complete")
FIELDS = ("open", "high", "low", "close", "volume")


def parse_time(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return dt.astimezone(timezone.utc)


def load_csv(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for line, raw in enumerate(reader, start=2):
            row = {"source": str(path), "line": line}
            row["symbol"] = (raw.get("symbol") or "").strip().upper()
            row["timestamp"] = parse_time(raw["timestamp"])
            row["is_complete"] = str(raw["is_complete"]).strip().lower() in {"true", "1", "yes"}
            for field in FIELDS:
                value = (raw.get(field) or "").strip()
                if not value:
                    raise ValueError(f"{path}:{line}: blank {field}")
                try:
                    row[field] = Decimal(value)
                except InvalidOperation as exc:
                    raise ValueError(f"{path}:{line}: invalid {field}={value!r}") from exc
            if not row["symbol"]:
                raise ValueError(f"{path}:{line}: blank symbol")
            rows.append(row)
    return rows


def candle_key(row):
    return row["symbol"], row["timestamp"]


def aggregate_5m(rows):
    grouped = defaultdict(list)
    for row in rows:
        # 5-minute candles are timestamped at the start of their interval.
        minute = (row["timestamp"].minute // 15) * 15
        start = row["timestamp"].replace(minute=minute, second=0, microsecond=0)
        grouped[(row["symbol"], start)].append(row)
    result = {}
    for key, group in grouped.items():
        group.sort(key=lambda item: item["timestamp"])
        if len(group) != 3:
            continue
        if any((group[i + 1]["timestamp"] - group[i]["timestamp"]).total_seconds() != 300
               for i in range(2)):
            continue
        result[key] = {
            "open": group[0]["open"],
            "high": max(item["high"] for item in group),
            "low": min(item["low"] for item in group),
            "close": group[-1]["close"],
            "volume": sum((item["volume"] for item in group), Decimal(0)),
        }
    return result


def audit(paths):
    all_rows = []
    per_file = []
    for path in paths:
        rows = load_csv(path)
        keys = [candle_key(row) for row in rows]
        duplicates = len(keys) - len(set(keys))
        bad_ohlc = sum(
            row["high"] < max(row["open"], row["close"], row["low"])
            or row["low"] > min(row["open"], row["close"], row["high"])
            or row["high"] < row["low"]
            for row in rows
        )
        negative_volume = sum(row["volume"] < 0 for row in rows)
        incomplete = sum(not row["is_complete"] for row in rows)
        by_symbol = defaultdict(list)
        for row in rows:
            by_symbol[row["symbol"]].append(row["timestamp"])
        cadence = defaultdict(int)
        for symbol, times in by_symbol.items():
            ordered = sorted(set(times))
            for left, right in zip(ordered, ordered[1:]):
                cadence[int((right - left).total_seconds())] += 1
        per_file.append({
            "path": str(path), "rows": len(rows), "symbols": len(by_symbol),
            "duplicate_keys": duplicates, "bad_ohlc": bad_ohlc,
            "negative_volume": negative_volume, "incomplete": incomplete,
            "cadence_seconds": dict(sorted(cadence.items())),
        })
        all_rows.extend(rows)

    # Cross-export conflicts: same symbol/time but different OHLCV.
    snapshots = defaultdict(list)
    for row in all_rows:
        snapshots[candle_key(row)].append(row)
    conflicts = []
    for key, copies in snapshots.items():
        if len(copies) > 1 and any(
            any(copy[field] != copies[0][field] for field in FIELDS)
            or copy["is_complete"] != copies[0]["is_complete"]
            for copy in copies[1:]
        ):
            conflicts.append(key)

    five = [row for row in all_rows if any(
        seconds == 300 for seconds in []
    )]
    # Infer 5m/15m datasets from within-file cadence, not filename conventions.
    five_rows, fifteen_rows = [], []
    for report, path in zip(per_file, paths):
        cadence = report["cadence_seconds"]
        if 300 in cadence:
            five_rows.extend(row for row in all_rows if row["source"] == str(path))
        elif 900 in cadence:
            fifteen_rows.extend(row for row in all_rows if row["source"] == str(path))
    five_by_interval = aggregate_5m(five_rows)
    fifteen_by_key = {candle_key(row): row for row in fifteen_rows}
    comparisons = []
    for key, calculated in five_by_interval.items():
        observed = fifteen_by_key.get(key)
        if observed:
            comparisons.append((key, calculated, observed))
    mismatches = [
        (key, field) for key, calculated, observed in comparisons
        for field in FIELDS if calculated[field] != observed[field]
    ]

    return {
        "files": per_file,
        "cross_export_conflicts": conflicts,
        "reconstructed_15m_intervals": len(five_by_interval),
        "matched_5m_to_15m": len(comparisons),
        "ohlcv_field_comparisons": len(comparisons) * len(FIELDS),
        "ohlcv_mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="5paisa candle CSV exports")
    args = parser.parse_args()
    try:
        report = audit(args.csv)
    except (OSError, ValueError) as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2

    print("LIVE DATA QUALITY AUDIT")
    for item in report["files"]:
        print(f"\n{item['path']}: {item['rows']} rows, {item['symbols']} symbols")
        print(f"  duplicate keys={item['duplicate_keys']}, bad OHLC={item['bad_ohlc']}, "
              f"negative volume={item['negative_volume']}, incomplete={item['incomplete']}")
        print(f"  observed cadence (seconds)={item['cadence_seconds']}")
    print(f"\nCross-export conflicting duplicate candles: {len(report['cross_export_conflicts'])}")
    print(f"Reconstructed 15m intervals: {report['reconstructed_15m_intervals']}")
    print(f"Matched 5m-to-15m candles: {report['matched_5m_to_15m']}")
    print(f"OHLCV field mismatches: {len(report['ohlcv_mismatches'])}")
    return 1 if (report["cross_export_conflicts"] or report["ohlcv_mismatches"]
                 or any(item["duplicate_keys"] or item["bad_ohlc"]
                        or item["negative_volume"] or item["incomplete"]
                        for item in report["files"])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
