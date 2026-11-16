"""Run Phase 8.9 validation-only pattern/context diagnostics.

Usage:
  python scripts/phase_8_9_pattern_context.py historical_ohlcv.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.ml_dataset import build_ml_dataset
from engine.ml_pattern_context import analyze_pattern_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Raw OHLCV CSV with timestamp, symbol, timeframe and OHLCV columns")
    parser.add_argument("--output-report", default="phase_8_9_pattern_context.json")
    parser.add_argument("--neutral-band", type=float, default=0.20)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12])
    args = parser.parse_args()

    source = pd.read_csv(args.csv)
    bands = {horizon: args.neutral_band for horizon in args.horizons}
    dataset, dataset_summary = build_ml_dataset(source, neutral_bands_pct=bands)
    report = analyze_pattern_context(
        dataset, horizons=tuple(args.horizons), min_support=args.min_support
    )
    report["dataset_summary"] = dataset_summary

    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print("Phase 8.9 pattern/context diagnostics (validation only)")
    print(f"Validation rows: {report['validation_rows']}")
    print(f"Patterns: {len(report['pattern_columns'])}")
    print(f"Report: {output}")
    print("Associations are descriptive only; no pattern is promoted as a signal.")


if __name__ == "__main__":
    main()
