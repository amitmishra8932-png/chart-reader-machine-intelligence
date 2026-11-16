"""Compare Phase 8.9 validation reports from 5m and 15m timeframes.

Usage:
  python scripts/phase_8_10_cross_timeframe.py report_5m.json report_15m.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.ml_cross_timeframe import analyze_cross_timeframe_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_5m", help="Phase 8.9 JSON report generated from 5m data")
    parser.add_argument("report_15m", help="Phase 8.9 JSON report generated from 15m data")
    parser.add_argument("--min-effect-pp", type=float, default=2.0)
    parser.add_argument("--output-report", default="phase_8_10_cross_timeframe.json")
    args = parser.parse_args()

    with Path(args.report_5m).open(encoding="utf-8") as handle:
        report_5m = json.load(handle)
    with Path(args.report_15m).open(encoding="utf-8") as handle:
        report_15m = json.load(handle)

    report = analyze_cross_timeframe_robustness(
        report_5m, report_15m, min_effect_pp=args.min_effect_pp
    )
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Phase 8.10 cross-timeframe robustness (validation aggregates only)")
    print(f"Common symbols: {report['common_symbol_count']}")
    print(f"Common patterns: {report['common_pattern_count']}")
    print(f"Horizons: {', '.join(report['horizons'])}")
    print(f"Report: {output}")
    print("Associations are descriptive only; no pattern is promoted as a trading signal.")


if __name__ == "__main__":
    main()
