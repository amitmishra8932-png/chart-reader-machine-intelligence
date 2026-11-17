"""Build Phase 8.12 coverage and quarantine audit outputs.

Usage:
  python scripts/phase_8_12_data_integrity.py historical.csv regular.csv --output-dir phase_8_12_output
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from engine.ml_event_alignment import prepare_phase_8_11
from engine.ml_data_integrity import audit_session_coverage, classify_quarantine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_csv")
    parser.add_argument("regular_csv")
    parser.add_argument("--output-dir", default="phase_8_12_output")
    parser.add_argument("--exclude-date", action="append", default=["2026-11-17"])
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(args.historical_csv)
    regular = pd.read_csv(args.regular_csv)
    events, quarantine, phase_811_manifest = prepare_phase_8_11(
        historical, regular, exclude_dates=args.exclude_date
    )
    sessions, candidates, phase_812_manifest = audit_session_coverage(
        regular, exclude_dates=args.exclude_date
    )
    classified_quarantine = classify_quarantine(quarantine)
    events.to_csv(out / "phase_8_11_event_level_5m.csv", index=False)
    sessions.to_csv(out / "phase_8_12_symbol_session_coverage.csv", index=False)
    candidates.to_csv(out / "phase_8_12_research_candidate_rows.csv", index=False)
    classified_quarantine.to_csv(out / "phase_8_12_quarantine_classification.csv", index=False)
    manifest = {"phase_8_11": phase_811_manifest, "phase_8_12": phase_812_manifest,
                "quarantine_rows_remain_excluded": int(len(classified_quarantine))}
    (out / "phase_8_12_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
