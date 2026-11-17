# Phase 8.15 — First validation-only baseline

## Experiment boundary

This is a first diagnostic baseline on the Phase 8.13 clean 5-minute research dataset. It is not a trading strategy and it does not establish predictive edge.

- Dataset: 192,450 rows, 50 symbols, 52 dates.
- Features: 45 numeric engineered features.
- Targets: 3, 6 and 12 candles; exploratory ±0.20% neutral band.
- Training rows: target-specific labelled rows from the train partition only.
- Validation rows: target-specific labelled rows from the validation partition only (8 dates).
- Test partition: not loaded for scoring, model selection or threshold selection.
- Preprocessing: median imputation (with missingness indicators) and standard scaling, fitted inside a pipeline on training rows only.
- Models: train-majority constant baseline; ordinary multinomial logistic regression; class-weight-balanced multinomial logistic regression.
- Logistic regression used default regularization settings, lbfgs solver, max_iter=500 and random_state=42. No hyperparameter search was performed.

## Validation results

| Horizon | Baseline | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| 3 | Train-majority (neutral) | 74.88% | 33.33% | 28.54% |
| 3 | Logistic, ordinary weights | 74.84% | 33.63% | 29.29% |
| 3 | Logistic, balanced weights | 59.95% | 41.66% | 40.19% |
| 6 | Train-majority (neutral) | 61.43% | 33.33% | 25.37% |
| 6 | Logistic, ordinary weights | 61.38% | 34.00% | 27.33% |
| 6 | Logistic, balanced weights | 51.36% | 39.61% | 38.98% |
| 12 | Train-majority (neutral) | 47.15% | 33.33% | 21.36% |
| 12 | Logistic, ordinary weights | 47.28% | 36.71% | 31.97% |
| 12 | Logistic, balanced weights | 43.54% | 38.36% | 37.54% |

## Balanced logistic regression: validation class recall

| Horizon | Down recall | Neutral recall | Up recall |
|---|---:|---:|---:|
| 3 | 20.47% | 71.03% | 33.50% |
| 6 | 15.95% | 67.59% | 35.29% |
| 12 | 17.90% | 64.27% | 32.91% |

## Interpretation

1. Ordinary logistic regression is close to the neutral-only baseline at 3 and 6 candles. Its accuracy is not meaningfully better.
2. Class weighting improves balanced accuracy and macro-F1 at every horizon, but lowers overall accuracy as expected.
3. Down-class recall remains weak (about 16%–20%) even with balanced weights. Directional discrimination is not strong enough for a trade signal.
4. The 12-candle horizon has the best macro-F1 of these initial runs, but still has low directional recall and was evaluated on only eight validation dates.
5. These are single fixed-parameter baselines, not a tuned or statistically validated model. No confidence intervals, walk-forward repeats, cost model or execution simulation are included.

## Decisions

- Do not score the held-out test partition yet.
- Do not deploy or generate live signals from this model.
- Keep all 45 features for this first benchmark only; feature selection must be train/validation controlled.
- Next compare regularized model variants and simple tree-based baselines using the same fixed chronological split, with all preprocessing inside training-fitted pipelines.
- Add per-date and per-symbol validation metrics, calibration checks and prediction coverage. If no stable improvement over the majority baseline appears, revisit the target definition and data coverage rather than forcing model complexity.
- Before any strategy claim, require a frozen model and one-time untouched test evaluation, followed by a separate cost-aware event/backtest design.

## Reproducibility note

The metrics above were produced in the analysis environment from the Phase 8.13 dataset file. This report documents the experiment and its exact high-level settings; it does not yet add a repository CLI runner. The Phase 8.14 gap-safe target code is merged, and Phase 8.13's strict complete-session dataset contains no internal timestamp gaps in the selected sessions.
