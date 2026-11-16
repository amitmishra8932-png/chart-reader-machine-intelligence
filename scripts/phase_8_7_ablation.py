"""Run Phase 8.7 feature-family ablations on a Phase 8.5 dataset.

Usage:
  python scripts/phase_8_7_ablation.py phase_8_5_ml_dataset.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.ml_ablation import evaluate_feature_ablations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV produced by scripts/phase_8_5_prepare.py")
    parser.add_argument("--output-report", default="phase_8_7_feature_ablation.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    report = evaluate_feature_ablations(frame)
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print("Phase 8.7 feature-family ablation")
    for horizon, groups in report["logistic_minus_majority_deltas"].items():
        print(f"\nHorizon {horizon}:")
        for group, splits in groups.items():
            validation = splits["validation"]
            test = splits["test"]
            print(
                f"  {group}: validation balanced_accuracy delta="
                f"{validation['balanced_accuracy']:+.4f}, macro_f1 delta={validation['macro_f1']:+.4f}; "
                f"held-out test balanced_accuracy delta={test['balanced_accuracy']:+.4f}, "
                f"macro_f1 delta={test['macro_f1']:+.4f}"
            )
    print(f"Report: {output}")
    print("Research comparison only; no strategy backtest or profitability claim.")


if __name__ == "__main__":
    main()
