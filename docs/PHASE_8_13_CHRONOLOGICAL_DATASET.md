# Phase 8.13 — Chronological research dataset preparation

## Input
The Phase 8.12 `phase_8_12_research_candidate_rows.csv` output.

## Strict selection
Only rows from complete, eligible, on-grid 5-minute sessions are selected. The module verifies that each selected symbol/session contains the entire assumed 75-bar grid (09:15–15:25). 15-minute rows, incomplete sessions, off-grid rows, weekend rows and historical-only quarantined records are excluded.

## Features, targets and splits
- Reuses the repository's causal feature and candle-pattern generation.
- Adds forward directional targets only for explicitly supplied neutral bands.
- Targets are bounded by symbol, timeframe and calendar date; no target crosses a session.
- Assigns whole calendar dates chronologically to train, validation and test.
- Checks that split date sets do not overlap.
- Retains feature warm-up missing values and unlabelled session tails; no imputation or row deletion is performed.

## Run
```bash
python scripts/phase_8_13_chronological_dataset.py phase_8_12_output/phase_8_12_research_candidate_rows.csv --band 3:0.10 --band 6:0.20 --band 12:0.30
```
The example bands are illustrative explicit inputs, not recommended or calibrated parameters. Select bands based on the approved research protocol.

## Outputs
- `phase_8_13_chronological_research_dataset.csv`
- `phase_8_13_dataset_manifest.json`

## Guardrails
No model fitting, train-time preprocessing, threshold tuning, trading signals or orders. Fit imputers/scalers on training data only in a later phase. Timestamp semantics remain inferred pending provider confirmation.
