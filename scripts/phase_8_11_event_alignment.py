"""Build Phase 8.11 event-level data from historical and regular CSV exports.

Usage:
  python scripts/phase_8_11_event_alignment.py historical.csv regular.csv --output-dir phase_8_11_output
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from engine.ml_event_alignment import prepare_phase_8_11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_csv")
    parser.add_argument("regular_csv")
    parser.add_argument("--output-dir", default="phase_8_11_output")
    parser.add_argument("--exclude-date", action="append", default=["2026-11-17"])
    parser.add_argument("--max-context-age-minutes", type=int, default=25)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events, quarantine, manifest = prepare_phase_8_11(
        pd.read_csv(args.historical_csv), pd.read_csv(args.regular_csv),
        exclude_dates=args.exclude_date,
        max_context_age_minutes=args.max_context_age_minutes,
    )
    events.to_csv(out / "phase_8_11_event_level_5m.csv", index=False)
    quarantine.to_csv(out / "phase_8_11_historical_only_quarantine.csv", index=False)
    (out / "phase_8_11_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
