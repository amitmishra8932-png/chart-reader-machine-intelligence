# Phase 8.14 — Target stability review

## Scope

This is a descriptive audit of the Phase 8.13 chronological research dataset (192,450 rows, 50 symbols, 52 session dates). It is not a model evaluation. All target distributions below use the exploratory neutral band of ±0.20% at each horizon; this is not a calibrated transaction-cost threshold.

## Target distributions by chronological split

Percentages are among all rows in the split, so the unlabelled session-tail rows are shown separately.

| Horizon | Split | Down | Neutral | Up | Unlabelled |
|---|---|---:|---:|---:|---:|
| 3 candles | Train | 15.3% | 66.3% | 14.4% | 4.0% |
| 3 candles | Validation | 12.1% | 71.9% | 12.0% | 4.0% |
| 3 candles | Test | 12.6% | 70.0% | 13.5% | 4.0% |
| 6 candles | Train | 22.1% | 50.2% | 19.7% | 8.0% |
| 6 candles | Validation | 18.0% | 56.5% | 17.5% | 8.0% |
| 6 candles | Test | 17.5% | 55.7% | 18.7% | 8.0% |
| 12 candles | Train | 26.6% | 34.6% | 22.8% | 16.0% |
| 12 candles | Validation | 23.3% | 39.6% | 21.1% | 16.0% |
| 12 candles | Test | 21.8% | 39.3% | 22.9% | 16.0% |

Labelled row counts:
- 3 candles: train 127,152; validation 28,800; test 28,800.
- 6 candles: train 121,854; validation 27,600; test 27,600.
- 12 candles: train 111,258; validation 25,200; test 25,200.

## Symbol coverage

For every horizon, all 50 symbols have non-zero up, neutral and down labels. The per-symbol neutral share ranges are:
- 3 candles: 55.7%–81.7% (median 71.4%).
- 6 candles: 41.7%–69.7% (median 56.9%).
- 12 candles: 30.5%–53.9% (median 43.7%).

This establishes class presence, not equal representativeness or predictive quality.

## Temporal variability

Daily class proportions vary within each split. For example, the daily neutral share ranges:
- 3 candles: train 56.2%–78.4%; validation 72.6%–77.8%; test 67.2%–78.9%.
- 6 candles: train 40.5%–65.7%; validation 58.2%–64.9%; test 54.9%–68.1%.
- 12 candles: train 30.1%–51.7%; validation 43.7%–51.5%; test 41.4%–53.8%.

The training distribution is not identical to validation/test. This is a real temporal distribution shift in the labels and must be reflected in evaluation.

## Implications for the next phase

1. A neutral-only classifier is a mandatory baseline. Accuracy must be reported beside macro-F1, balanced accuracy, per-class precision/recall and confusion matrices.
2. Baselines must be fit using training data only. Hyperparameter selection and any threshold selection must use validation only. The test partition must remain untouched until the model and evaluation protocol are frozen.
3. Preprocessing (including missing-value handling and scaling) must be fitted on training rows only, then applied unchanged to validation/test.
4. Report results per horizon; do not pool 3/6/12-candle labels.
5. Add date-level and symbol-level breakdowns so aggregate metrics cannot hide regime or symbol concentration.
6. Keep the ±0.20% neutral band explicitly exploratory. Later sensitivity checks may compare predeclared alternatives, but must not choose a band by test performance.
7. No profitability conclusion follows from directional classification. Execution costs, spread, slippage, liquidity and trade rules are not represented by these labels.

## Decision

Proceed to a leakage-safe baseline evaluation on the existing clean 5-minute sample only as a research experiment, with the above guardrails. Do not promote the dataset to a production signal source. The limited 52-date coverage and unresolved synchronized closing-bar cutoff remain material limitations.
