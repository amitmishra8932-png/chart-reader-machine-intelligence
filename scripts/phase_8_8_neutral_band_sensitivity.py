"""Run neutral-band sensitivity analysis on raw historical OHLCV.

Usage:
  python scripts/phase_8_8_neutral_band_sensitivity.py historical_ohlcv.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.ml_sensitivity import DEFAULT_NEUTRAL_BANDS_PCT, run_neutral_band_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Raw OHLCV CSV with timestamp, symbol, timeframe and OHLCV columns")
    parser.add_argument("--output-report", default="phase_8_8_neutral_band_sensitivity.json")
    parser.add_argument(
        "--bands",
        nargs="+",
        type=float,
        default=list(DEFAULT_NEUTRAL_BANDS_PCT),
        help="Exploratory neutral-band percentages (default: 0.10 0.15 0.20 0.30)",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.csv)
    report = run_neutral_band_sensitivity(source, bands_pct=args.bands)
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print("Phase 8.8 neutral-band sensitivity (research only)")
    for band, result in report["results_by_band_pct"].items():
        print(f"\\nNeutral band {band}%")
        for horizon, groups in result["feature_ablation"]["logistic_minus_majority_deltas"].items():
            all_features = groups["all_features"]["validation"]
            print(
                f"  {horizon}-candle horizon: validation all-features "
                f"balanced_accuracy delta={all_features['balanced_accuracy']:+.4f}, "
                f"macro_f1 delta={all_features['macro_f1']:+.4f}"
            )
    print(f"Report: {output}")
    print("Held-out test metrics are report-only; no winning band is selected.")


if __name__ == "__main__":
    main()
