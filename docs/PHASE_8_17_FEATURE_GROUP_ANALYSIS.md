# Phase 8.17 — Feature-group analysis

## Objective

Measure the standalone and incremental validation contribution of the five existing feature families. This is a research comparison, not a model deployment or trading-signal exercise.

## Data and protocol

- Source: Phase 8.13 chronological research dataset.
- Universe: 50 symbols; 5-minute bars; 52 session dates.
- Feature inventory: 45 numeric features.
- Horizons: 3, 6 and 12 bars; existing exploratory neutral band ±0.20%.
- Splits: chronological train and validation only. The held-out test partition was not loaded or evaluated.
- Model: balanced multinomial logistic regression, C=1.0, lbfgs, random_state=42, max_iter=250 for this experiment.
- Preprocessing: median imputation with missingness indicators and standard scaling; each transform fitted on training rows only.
- Label rows with missing targets are excluded separately for each horizon.
- Train/validation labelled sample sizes: 127,152 / 28,800 (3 bars); 121,854 / 27,600 (6 bars); 111,258 / 25,200 (12 bars).
- Metrics: accuracy, balanced accuracy, macro-F1 and per-class recall (down, neutral, up).

Feature groups:
- Price action (12): candle anatomy, candle direction, gap and 1/3/6/12-bar returns.
- Volatility (4): ATR, range/ATR, body/ATR and realized volatility.
- Volume (2): relative volume and volume z-score.
- Trend (4): price distance from 20/50 SMA and EMA.
- Candle patterns (23): existing binary candle-pattern flags.

## Standalone feature-group results

| Horizon | Group | Features | Accuracy | Balanced accuracy | Macro-F1 | Down recall | Neutral recall | Up recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | Price action | 12 | 63.51% | 39.67% | 38.94% | 11.40% | 77.94% | 29.67% |
| 3 | Volatility | 4 | 60.61% | 41.84% | 39.78% | 13.94% | 71.98% | 39.60% |
| 3 | Volume | 2 | 71.36% | 36.24% | 34.79% | 1.29% | 92.61% | 14.81% |
| 3 | Trend | 4 | 42.67% | 33.86% | 30.95% | 33.83% | 47.98% | 19.77% |
| 3 | Candle patterns | 23 | 40.37% | 34.83% | 30.10% | 14.88% | 43.74% | 45.87% |
| 6 | Price action | 12 | 53.58% | 38.04% | 37.17% | 9.73% | 75.03% | 29.36% |
| 6 | Volatility | 4 | 52.04% | 39.75% | 39.17% | 15.67% | 69.02% | 34.57% |
| 6 | Volume | 2 | 59.41% | 35.65% | 32.39% | 2.84% | 92.07% | 12.04% |
| 6 | Trend | 4 | 37.03% | 33.04% | 31.90% | 27.97% | 42.50% | 28.64% |
| 6 | Candle patterns | 23 | 40.53% | 34.74% | 32.75% | 13.09% | 48.63% | 42.52% |
| 12 | Price action | 12 | 42.99% | 36.50% | 34.57% | 10.60% | 69.00% | 29.89% |
| 12 | Volatility | 4 | 44.06% | 38.64% | 37.87% | 18.53% | 65.63% | 31.76% |
| 12 | Volume | 2 | 46.94% | 35.25% | 28.21% | 1.93% | 92.27% | 11.55% |
| 12 | Trend | 4 | 34.90% | 32.57% | 31.96% | 17.92% | 44.87% | 34.93% |
| 12 | Candle patterns | 23 | 36.24% | 33.97% | 31.10% | 8.06% | 47.28% | 46.58% |

## Selected combinations

| Horizon | Feature set | Features | Accuracy | Balanced accuracy | Macro-F1 | Down recall | Neutral recall | Up recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | Volatility + price action + volume | 18 | 60.39% | 42.16% | 40.58% | 19.86% | 71.42% | 35.19% |
| 3 | Without patterns (all 22 other features) | 22 | 60.39% | 42.00% | 40.51% | 20.17% | 71.51% | 34.33% |
| 3 | All 45 features | 45 | 59.95% | 41.66% | 40.19% | 20.47% | 71.03% | 33.50% |
| 6 | Volatility + volume | 6 | 52.07% | 39.80% | 39.21% | 15.65% | 69.02% | 34.72% |
| 6 | Volatility only | 4 | 52.04% | 39.75% | 39.17% | 15.67% | 69.02% | 34.57% |
| 6 | All 45 features | 45 | 51.36% | 39.61% | 38.98% | 15.95% | 67.59% | 35.29% |
| 12 | Volatility + candle patterns | 27 | 44.00% | 38.71% | 37.99% | 18.91% | 65.04% | 32.20% |
| 12 | Volatility + volume | 6 | 44.09% | 38.67% | 37.90% | 18.55% | 65.66% | 31.79% |
| 12 | Volatility only | 4 | 44.06% | 38.64% | 37.87% | 18.53% | 65.63% | 31.76% |
| 12 | All 45 features | 45 | 43.54% | 38.36% | 37.54% | 17.90% | 64.27% | 32.91% |

## Findings

1. Volatility is the strongest standalone feature group by macro-F1 and balanced accuracy at all three horizons. Four volatility features alone are close to, and in some cases slightly above, the 45-feature benchmark.
2. At 3 bars, volatility + price action + volume is the best tested combination by macro-F1 (40.58%), a small gain over all 45 features (40.19%).
3. At 6 bars, volatility + volume is marginally ahead of volatility alone and the full feature set. The gain is very small.
4. At 12 bars, volatility + candle patterns has the highest macro-F1 among tested sets (37.99%), but the improvement over volatility alone (37.87%) is negligible. This does not overturn the Phase 8.16 finding that patterns did not show useful incremental value in the earlier ablation.
5. Volume-only accuracy is deceptively high because the model overwhelmingly predicts neutral: neutral recall exceeds 92%, while down recall is below 3% and up recall is about 12–15%. Accuracy must not be used alone.
6. Trend-only and pattern-only groups are weak overall. Price action alone is more useful than trend, volume or patterns alone, but trails volatility on balanced metrics.
7. Down recall remains poor across all candidate sets (roughly 14–20% for the leading combinations). These results do not support dependable directional calls.

## Decision

- Carry volatility as the core feature group into the next controlled experiment.
- Keep price action and volume as candidate additions for the 3-bar horizon; test whether their small gain is stable across dates and symbols.
- Do not promote volume-only, trend-only or pattern-only models.
- Do not remove pattern features from the codebase on this evidence alone; the observed differences are small and validation-specific.
- Do not tune on or inspect the held-out test partition yet.
- No model is approved for live recommendations, paper-trading signals or automated orders.

## Limitations and next steps

This is a single chronological validation window and one fixed logistic-regression benchmark. No confidence intervals, date-block bootstrap, symbol-level robustness analysis, transaction costs, spread, slippage or executable trade rules are included. The ±0.20% neutral band remains exploratory and is not calibrated to costs.

Next: assess whether the volatility-led result is stable by validation date and symbol, then freeze the feature set and evaluation protocol before any final test-set evaluation.
