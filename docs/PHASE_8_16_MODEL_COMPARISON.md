# Phase 8.16 — Model and candle-pattern comparison

## Scope and split discipline

This experiment compares simple models on the Phase 8.13 clean 5-minute dataset. It uses the chronological train and validation partitions only. The held-out test partition was not loaded for scoring or model selection.

- 50 symbols; 52 session dates.
- 45 numeric features: 22 non-pattern features and 23 binary candle-pattern flags.
- Horizons: 3, 6 and 12 bars.
- Neutral band: ±0.20% for each horizon, exploratory only.
- Train-only median imputation; scaling is applied to logistic regression only. Each preprocessing pipeline is fitted on training data only.
- Validation sample sizes: 28,800 (3 bars), 27,600 (6 bars), 25,200 (12 bars).

Models:
- Majority-class dummy baseline.
- Balanced multinomial logistic regression (C=0.1, 1, 10; lbfgs, max_iter=500, random_state=42).
- ExtraTrees: 250 trees, max_depth=14, min_samples_leaf=40, max_features=sqrt, class_weight=balanced.
- HistGradientBoosting: 150 iterations, 15 max leaf nodes, learning_rate=0.08, l2_regularization=1.0, class_weight=balanced.

## Validation metrics

| Horizon | Model | Accuracy | Balanced accuracy | Macro-F1 | Down recall | Neutral recall | Up recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 3 | Majority | 74.88% | 33.33% | 28.54% | 0.00% | 100.00% | 0.00% |
| 3 | Logistic balanced | 59.95% | 41.66% | 40.19% | 20.47% | 71.03% | 33.50% |
| 3 | ExtraTrees | 63.73% | 41.12% | 40.08% | 12.98% | 77.42% | 32.97% |
| 3 | HistGradientBoosting | 58.24% | 41.44% | 39.72% | 23.39% | 68.40% | 32.53% |
| 6 | Majority | 61.43% | 33.33% | 25.37% | 0.00% | 100.00% | 0.00% |
| 6 | Logistic balanced | 51.36% | 39.61% | 38.98% | 15.95% | 67.59% | 35.29% |
| 6 | ExtraTrees | 54.32% | 39.21% | 38.01% | 9.43% | 75.19% | 33.01% |
| 6 | HistGradientBoosting | 49.50% | 39.46% | 38.80% | 18.86% | 63.38% | 36.15% |
| 12 | Majority | 47.15% | 33.33% | 21.36% | 0.00% | 100.00% | 0.00% |
| 12 | Logistic balanced | 43.54% | 38.36% | 37.54% | 17.90% | 64.27% | 32.91% |
| 12 | ExtraTrees | 43.96% | 37.11% | 34.55% | 8.61% | 71.56% | 31.17% |
| 12 | HistGradientBoosting | 41.58% | 37.67% | 37.21% | 20.91% | 57.37% | 34.74% |

The logistic C=0.1, 1 and 10 runs were effectively tied at all horizons. No meaningful regularization optimum was established.

## Candle-pattern feature ablation

The same balanced logistic regression (C=1) was fitted three ways: all 45 features, 22 non-pattern features only, and 23 pattern flags only.

| Horizon | Feature set | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| 3 | All 45 | 59.95% | 41.66% | 40.19% |
| 3 | Without 23 patterns | 60.39% | 42.00% | 40.51% |
| 3 | Patterns only | 40.37% | 34.83% | 30.10% |
| 6 | All 45 | 51.36% | 39.61% | 38.98% |
| 6 | Without 23 patterns | 51.64% | 39.74% | 39.09% |
| 6 | Patterns only | 40.53% | 34.74% | 32.75% |
| 12 | All 45 | 43.54% | 38.36% | 37.54% |
| 12 | Without 23 patterns | 43.65% | 38.38% | 37.49% |
| 12 | Patterns only | 36.24% | 33.97% | 31.10% |

## Findings

1. Balanced logistic regression has the best macro-F1 at 3 and 6 bars, but HistGradientBoosting is close. At 12 bars, logistic regression is marginally ahead.
2. ExtraTrees does not improve balanced accuracy or macro-F1 over logistic regression and has notably weak down recall.
3. HistGradientBoosting has the strongest down recall in each horizon among these compared models, but down recall remains only about 19%–23%.
4. Removing all 23 candle-pattern flags slightly improves validation macro-F1 at 3 and 6 bars and is essentially tied at 12 bars. Pattern-only performance is materially weaker.
5. This does not prove candle patterns have no value. It shows that these current binary flags add no measurable incremental value to this particular feature set, split, target and linear benchmark.
6. All models remain far from a reliable three-class directional classifier. No model is suitable for live recommendations.

## Decision and next work

- Retain balanced logistic regression as a transparent benchmark, not as a deployed model.
- Keep HistGradientBoosting as a challenger because its down-class recall is somewhat higher; do not select it based on that one metric alone.
- Do not spend time increasing tree complexity yet.
- Next perform feature-group ablation on training/validation only (price/returns, volatility, volume, trend, candle patterns), then date- and symbol-level validation diagnostics.
- Keep the final test partition untouched until the candidate model and protocol are frozen.
- Do not claim profitability. These labels omit spread, slippage, liquidity, execution timing and a complete trade-management rule.

## Reproducibility caveat

The numerical experiments were executed in the analysis environment against the generated Phase 8.13 CSV. This document records settings and outcomes; it is not a repository training CLI or a saved deployable model.
