"""Run the Phase 8.4 historical feature/data audit.

Usage:
    python scripts/phase_8_4_validate.py input.csv --output report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.feature_validation import build_feature_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Historical OHLCV CSV with timestamp/symbol/timeframe")
    parser.add_argument("--output", default="phase_8_4_feature_validation.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    report = build_feature_validation_report(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    audit = report["dataset_audit"]
    print(f"Rows: {audit['row_count']:,}")
    print(f"Symbols: {', '.join(audit['symbols'])}")
    print(f"Timeframes: {', '.join(audit['timeframes'])}")
    print(f"Duplicate keys: {audit['duplicate_symbol_timeframe_timestamps']}")
    print(f"Invalid OHLC rows: {audit['invalid_ohlc_rows']}")
    print(f"Feature columns: {len(report['feature_distributions'])}")
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()
