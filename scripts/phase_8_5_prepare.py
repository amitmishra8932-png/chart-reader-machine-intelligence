"""Prepare a leakage-safe ML research dataset (no model fitting).

Usage:
  python scripts/phase_8_5_prepare.py candles.csv --band-3 0.15 --band-6 0.20 --band-12 0.30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.ml_dataset import build_ml_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="OHLCV CSV with timestamp, symbol and timeframe")
    parser.add_argument("--band-3", type=float, required=True, help="Neutral band in percent for 3 bars")
    parser.add_argument("--band-6", type=float, required=True, help="Neutral band in percent for 6 bars")
    parser.add_argument("--band-12", type=float, required=True, help="Neutral band in percent for 12 bars")
    parser.add_argument("--output-csv", default="phase_8_5_ml_dataset.csv")
    parser.add_argument("--output-report", default="phase_8_5_ml_dataset_report.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    dataset, report = build_ml_dataset(
        frame,
        neutral_bands_pct={3: args.band_3, 6: args.band_6, 12: args.band_12},
    )
    csv_path = Path(args.output_csv)
    report_path = Path(args.output_report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(csv_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Prepared rows: {len(dataset):,}")
    print(f"Features: {len(report['feature_columns'])}")
    print(f"Dataset: {csv_path}")
    print(f"Report: {report_path}")
    print("No model was fitted.")


if __name__ == "__main__":
    main()
