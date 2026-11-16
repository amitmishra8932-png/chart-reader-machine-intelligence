"""Run Phase 8.6 baseline ML evaluation on a Phase 8.5 prepared dataset.

Usage:
  python scripts/phase_8_6_evaluate.py phase_8_5_ml_dataset.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.ml_evaluation import evaluate_baseline_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV produced by scripts/phase_8_5_prepare.py")
    parser.add_argument("--output-report", default="phase_8_6_baseline_evaluation.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    report = evaluate_baseline_models(frame)
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Features used: {len(report['feature_columns'])}")
    for horizon, details in report["horizons"].items():
        print(f"\nHorizon {horizon}:")
        for model_name, splits in details["models"].items():
            test = splits["test"]
            print(
                f"  {model_name}: test balanced_accuracy="
                f"{test['balanced_accuracy']:.4f}, macro_f1={test['macro_f1']:.4f}"
            )
    print(f"Report: {output}")
    print("Research evaluation only; no strategy backtest or profitability claim.")


if __name__ == "__main__":
    main()
