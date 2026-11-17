"""Build the Phase 8.13 clean chronological research dataset.

Example:
python scripts/phase_8_13_chronological_dataset.py phase_8_12_output/phase_8_12_research_candidate_rows.csv --band 3:0.10 --band 6:0.20 --band 12:0.30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from engine.ml_chronological_dataset import prepare_phase_8_13


def parse_band(value: str) -> tuple[int, float]:
    try:
        horizon_text, band_text = value.split(":", 1)
        horizon, band = int(horizon_text), float(band_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Band must use HORIZON:PERCENT, e.g. 3:0.10") from exc
    if horizon <= 0 or band <= 0:
        raise argparse.ArgumentTypeError("Horizon and neutral band must be positive.")
    return horizon, band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_csv", help="Phase 8.12 research_candidate_rows.csv")
    parser.add_argument("--band", action="append", type=parse_band, required=True,
                        help="Explicit target neutral band HORIZON:PERCENT; repeat per horizon.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--output-dir", default="phase_8_13_output")
    args = parser.parse_args()

    bands = dict(args.band)
    if len(bands) != len(args.band):
        parser.error("Each target horizon may be supplied only once.")
    candidates = pd.read_csv(args.candidate_csv)
    dataset, summary = prepare_phase_8_13(
        candidates,
        neutral_bands_pct=bands,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output / "phase_8_13_chronological_research_dataset.csv", index=False)
    (output / "phase_8_13_dataset_manifest.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
