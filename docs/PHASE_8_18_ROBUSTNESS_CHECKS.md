# Phase 8.18 — Date-wise and symbol-wise robustness

## Objective

Check whether the Phase 8.17 volatility-led feature sets retain their validation performance across individual dates and symbols. This is a robustness diagnostic, not a new model-selection exercise.

## Data and protocol

- Source: Phase 8.13 chronological research dataset.
- Universe: 50 symbols; 5-minute bars.
- Validation window: 8 dates, 2026-07-10 through 2026-07-21.
- Training window: 36 dates, 2026-05-19 through 2026-07-09.
- The held-out test partition (2026-07-22 through 2026-07-31) was not loaded or evaluated.
- Model: balanced multinomial logistic regression, C=1.0, lbfgs, max_iter=250, random_state=42.
- Preprocessing: median imputation with missingness indicators and standard scaling, fitted on training rows only.
- Each horizon has its own target-valid rows.
- Compare volatility-only against the Phase 8.17 selected candidate:
  - 3 bars: volatility + price action + volume (18 features).
  - 6 bars: volatility + volume (6 features).
  - 12 bars: volatility + candle-pattern flags (27 features).
- Robustness summaries use the same fixed fitted model's validation predictions, grouped by validation date and symbol. No per-date or per-symbol refitting.

## Overall validation comparison

| Horizon | Feature set | N | Macro-F1 | Balanced accuracy | Accuracy | Down recall | Neutral recall | Up recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | Volatility only | 28,800 | 39.78% | 41.84% | 60.61% | 13.94% | 71.98% | 39.60% |
| 3 | Candidate combination | 28,800 | 40.58% | 42.16% | 60.39% | 19.86% | 71.42% | 35.19% |
| 6 | Volatility only | 27,600 | 39.17% | 39.75% | 52.04% | 15.67% | 69.02% | 34.57% |
| 6 | Candidate combination | 27,600 | 39.21% | 39.80% | 52.07% | 15.65% | 69.02% | 34.72% |
| 12 | Volatility only | 25,200 | 37.87% | 38.64% | 44.06% | 18.53% | 65.63% | 31.76% |
| 12 | Candidate combination | 25,200 | 37.99% | 38.71% | 44.00% | 18.91% | 65.04% | 32.20% |

## Date-level robustness

Mean macro-F1 across the 8 validation dates, followed by standard deviation and observed minimum–maximum:

| Horizon | Feature set | Mean date Macro-F1 | SD | Min–max |
|---:|---|---:|---:|---:|
| 3 | Volatility only | 39.61% | 1.75 pp | 36.53–42.20% |
| 3 | Candidate combination | 40.44% | 1.26 pp | 38.57–41.69% |
| 6 | Volatility only | 38.95% | 2.44 pp | 34.76–43.34% |
| 6 | Candidate combination | 38.99% | 2.61 pp | 34.53–43.73% |
| 12 | Volatility only | 37.64% | 3.09 pp | 31.84–40.64% |
| 12 | Candidate combination | 37.78% | 2.95 pp | 32.19–41.05% |

Candidate-minus-volatility date-level macro-F1 differences:
- 3 bars: mean +0.83 percentage points; positive on 5 of 8 dates; observed range -0.65 to +2.28 pp.
- 6 bars: mean +0.04 pp; positive on 5 of 8 dates; observed range -0.23 to +0.39 pp.
- 12 bars: mean +0.14 pp; positive on 6 of 8 dates; observed range -0.40 to +0.44 pp.

A paired date-block bootstrap (20,000 resamples of the 8 validation dates, seed 42) gave 95% percentile intervals for the mean difference:
- 3 bars: +0.07 to +1.56 pp.
- 6 bars: -0.08 to +0.17 pp.
- 12 bars: -0.09 to +0.33 pp.

There are only eight validation dates, so these intervals are exploratory and should not be interpreted as strong inferential evidence.

## Symbol-level robustness

Mean macro-F1 across the 50 symbols, followed by standard deviation and observed minimum–maximum:

| Horizon | Feature set | Mean symbol Macro-F1 | SD | Min–max |
|---:|---|---:|---:|---:|
| 3 | Volatility only | 38.36% | 2.69 pp | 32.15–44.86% |
| 3 | Candidate combination | 39.35% | 2.50 pp | 33.12–44.73% |
| 6 | Volatility only | 37.64% | 3.63 pp | 29.67–45.67% |
| 6 | Candidate combination | 37.67% | 3.62 pp | 29.75–46.25% |
| 12 | Volatility only | 35.72% | 4.99 pp | 26.09–47.90% |
| 12 | Candidate combination | 35.90% | 4.60 pp | 26.30–46.59% |

Candidate-minus-volatility symbol-level macro-F1 differences:
- 3 bars: mean +1.00 pp; positive for 36 of 50 symbols.
- 6 bars: mean +0.03 pp; positive for 30 of 50 symbols.
- 12 bars: mean +0.18 pp; positive for 29 of 50 symbols.

## Findings

1. The 3-bar volatility + price action + volume combination has the clearest robustness signal of the three candidates: it improves mean date-level macro-F1 on average and is better on 5/8 dates and 36/50 symbols. Its paired date-block bootstrap interval is positive, but the sample contains only eight dates and one validation window.
2. At 6 bars, adding volume to volatility produces a negligible difference. The date-level interval crosses zero and only 30/50 symbols improve. Treat this as a tie; prefer the simpler volatility-only set unless later evidence changes the decision.
3. At 12 bars, the volatility + pattern combination has only a very small mean gain. The date-level interval crosses zero and symbol results are mixed. No reliable incremental pattern benefit is established.
4. Date-level performance varies, particularly at 6 and 12 bars. Symbol-level variation is also material; the mean result does not describe every stock.
5. Down-class recall remains low: approximately 14–20% for these leading sets. Robustness across groups does not make the classifier dependable for directional trading calls.

## Decision

- Carry volatility + price action + volume as a provisional 3-bar research candidate only.
- Keep volatility-only as the simpler reference for 6- and 12-bar horizons; additions have not demonstrated a robust practical gain.
- Do not remove candle-pattern features from the feature-engineering codebase, but do not claim they improve generalization.
- Do not select a deployable model, issue live recommendations, or start automated orders.
- Keep the final test partition sealed until the model, features, targets, and evaluation protocol are frozen.

## Limitations and next step

This analysis uses one validation window with eight dates and 50 symbols. The same model is evaluated by subgroup; these are not independent samples because symbols and dates share market conditions. The bootstrap is descriptive and has limited power. There is no second untouched market regime, no cost-aware trade simulation, and no evidence of profitability.

Next: freeze the provisional feature protocol and run a pre-registered final evaluation on the held-out chronological test partition once all decisions are locked. Report all horizons and classes, including failures; do not tune after seeing test results.
