# Phase 8.19 — Frozen protocol and final chronological test evaluation

## Status and purpose

This phase records the final evaluation of the research candidates selected using training and validation data. The test partition is evaluated once for the frozen candidates. This is a diagnostic ML evaluation, not a trading strategy or deployment approval.

The feature choices and model settings were selected from the earlier train/validation experiments before the final test evaluation. This report is the durable audit record; it does not claim that the protocol was independently time-stamped in a prior repository commit.

## Frozen protocol

### Dataset and splits

- Source: Phase 8.13 chronological research dataset.
- Universe: 50 NIFTY 50 symbols, 5-minute bars.
- Date ranges:
  - Train: 2026-05-19 to 2026-07-09 (36 dates).
  - Validation: 2026-07-10 to 2026-07-21 (8 dates).
  - Test: 2026-07-22 to 2026-07-31 (8 dates).
- Rows are evaluated only when the corresponding target is labelled.
- Target classes: -1 = down, 0 = neutral, +1 = up.
- Exploratory neutral band remains ±0.20%; it is not calibrated to spread, slippage or execution costs.

### Feature sets fixed from validation research

- 3-bar horizon: 18 features = volatility (4) + price action (12) + volume (2).
- 6-bar horizon: volatility only (4).
- 12-bar horizon: volatility only (4).
- No feature set was changed after examining test results.

### Model and preprocessing

- Balanced multinomial LogisticRegression.
- C=1.0, solver=lbfgs, max_iter=250, random_state=42.
- Pipeline: median imputation with missingness indicators, StandardScaler, logistic regression.
- Imputer and scaler fitted on the training partition only.
- Model fitted on labelled training rows only.
- Validation and test rows are never used to fit preprocessing or model parameters.
- Comparator: train-prior DummyClassifier, which predicts the most frequent training class.
- Metrics: accuracy, balanced accuracy, macro-F1 and per-class precision/recall/F1.
- No hyperparameter tuning, threshold tuning, feature selection or refitting was performed on validation/test data during this final evaluation.

## Final test results

| Horizon | Feature set | Test N | Accuracy | Balanced accuracy | Macro-F1 |
|---:|---|---:|---:|---:|---:|
| 3 bars | Volatility + price action + volume | 28,800 | 58.22% | 40.93% | 39.89% |
| 3 bars | Train-prior baseline | 28,800 | 72.87% | 33.33% | 28.10% |
| 6 bars | Volatility only | 27,600 | 50.75% | 38.51% | 38.17% |
| 6 bars | Train-prior baseline | 27,600 | 60.58% | 33.33% | 25.15% |
| 12 bars | Volatility only | 25,200 | 43.59% | 37.92% | 37.18% |
| 12 bars | Train-prior baseline | 25,200 | 46.83% | 33.33% | 21.26% |

The balanced model improves macro-F1 and balanced accuracy over the majority-class baseline at every horizon, while overall accuracy is lower. This is expected when the model attempts to identify minority directional classes rather than predicting neutral for nearly every row.

## Test class recall and precision

| Horizon | Class | Precision | Recall | F1 |
|---:|---|---:|---:|---:|
| 3 | Down | 20.00% | 21.42% | 20.69% |
| 3 | Neutral | 79.86% | 70.00% | 74.61% |
| 3 | Up | 19.92% | 31.38% | 24.37% |
| 6 | Down | 22.04% | 15.93% | 18.49% |
| 6 | Neutral | 67.74% | 68.22% | 67.98% |
| 6 | Up | 25.35% | 31.40% | 28.05% |
| 12 | Down | 27.00% | 16.95% | 20.83% |
| 12 | Neutral | 53.46% | 65.48% | 58.86% |
| 12 | Up | 32.37% | 31.33% | 31.84% |

## Test-set label and prediction counts

Actual labels:
- 3 bars: down 3,773; neutral 20,986; up 4,041.
- 6 bars: down 5,262; neutral 16,720; up 5,618.
- 12 bars: down 6,542; neutral 11,802; up 6,856.

Model predictions:
- 3 bars: down 4,039; neutral 18,394; up 6,367.
- 6 bars: down 3,803; neutral 16,839; up 6,958.
- 12 bars: down 4,108; neutral 14,457; up 6,635.

## Interpretation

1. All three frozen candidates beat the train-prior baseline on macro-F1 and balanced accuracy. This indicates some class-discrimination signal under this particular label definition and test window, but does not establish a tradable edge.
2. The 3-bar candidate has the highest macro-F1 (39.89%) and balanced accuracy (40.93%) of the three frozen horizons. Its directional precision is low: about 20% for both down and up predictions.
3. The 6-bar volatility-only model has macro-F1 38.17%; down recall is only 15.93%.
4. The 12-bar volatility-only model has macro-F1 37.18%; down recall is 16.95%, and up recall is 31.33%.
5. The model's directional class precision remains weak across horizons. A large fraction of predicted directional labels are incorrect.
6. Accuracy alone is misleading because the neutral class is dominant, especially at 3 bars.
7. The final test window contains only eight dates from one historical period. This does not demonstrate stability across different market regimes, future periods, or changing constituents.

## Decision

- The frozen research candidates pass the limited benchmark comparison against a train-prior baseline, but none is approved for live recommendations, paper-trading signals, or automated orders.
- Keep the 3-bar volatility + price action + volume set as the leading research candidate, not a production model.
- Keep 6-bar and 12-bar volatility-only results as horizon-specific diagnostics.
- Do not tune these models against the just-evaluated test results. Any future iteration requires a new untouched chronological test period.
- Do not claim profitability or predictive reliability. The labels do not incorporate spread, slippage, liquidity, execution timing, position sizing, stop/exit rules or transaction costs.

## Next work

Before any deployment discussion, build a separate cost-aware event/backtest protocol with realistic entry/exit rules, fees, spread/slippage assumptions, turnover and risk controls. That work must use a new out-of-sample period or walk-forward design; the Phase 8.19 test results remain a final holdout for this experiment.
