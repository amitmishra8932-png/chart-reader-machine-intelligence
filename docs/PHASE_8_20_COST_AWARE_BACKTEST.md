# Phase 8.20 — Cost-aware fixed-horizon event simulation

## Status and research boundary

This is a first mechanical simulation of the frozen Phase 8.19 classifier predictions. It is not an independent out-of-sample validation because the same 2026-07-22 to 2026-07-31 period was already used for Phase 8.19's final classification report. The simulation is retrospective and diagnostic only. It must not be used to tune the classifier or claim a tradable edge.

## Frozen inputs and execution assumptions

- Source: Phase 8.13 complete-session 5-minute research dataset.
- Period simulated: existing Phase 8.19 test window, 2026-07-22 through 2026-07-31 (8 dates; 50 symbols).
- Model: balanced multinomial LogisticRegression, C=1.0, lbfgs, max_iter=250, random_state=42, fitted on labelled training rows only.
- Feature sets: 3 bars = volatility + price action + volume; 6 and 12 bars = volatility only.
- Preprocessing: median imputation with missingness indicators and StandardScaler fitted on training rows only.
- Prediction: argmax class prediction; neutral predictions do not trade. No confidence threshold was added.
- Signal timing: prediction is assumed available after signal candle close.
- Entry: next 5-minute candle open.
- Exit: close after exactly the selected number of full 5-minute bars from entry.
- Direction: +1 = long; -1 = short. Short trades are hypothetical intraday shorts only.
- Session safety: no trades cross a session boundary; missing timestamp intervals invalidate a trade.
- Overlap: at most one open trade per symbol; subsequent signals are ignored until the prior trade exits.
- Exit: fixed time horizon only. No stop-loss, profit target, trailing stop or discretionary exit.
- Costs: sensitivity scenarios of 0, 5, 10, 20, 30 and 50 basis points per completed round trip, applied to notional return. These are all-in scenario assumptions, not broker fee calculations or a claim about actual 5paisa/Zerodha charges.
- Position sizing: equal-weight per trade diagnostics only. No portfolio capital, concurrent exposure cap, compounding or rupee P&L is inferred.

## Trade counts and gross results before costs

| Horizon | Trades | Long | Short | Mean gross return/trade | Gross win rate |
|---:|---:|---:|---:|---:|---:|
| 3 bars | 2,652 | 1,153 | 1,499 | +0.78 bps | 51.47% |
| 6 bars | 1,922 | 1,051 | 871 | -2.16 bps | 46.25% |
| 12 bars | 1,167 | 589 | 578 | -5.53 bps | 45.24% |

The 3-bar candidate has only a very small positive average gross return before costs. The 6- and 12-bar candidates are negative before costs.

## Cost sensitivity

All-in round-trip cost is deducted once from each completed trade's gross return.

| Horizon | Cost (bps RT) | Mean net bps/trade | Net win rate | Profit factor |
|---:|---:|---:|---:|---:|
| 3 | 0 | +0.78 | 51.47% | 1.067 |
| 3 | 5 | -4.22 | 43.29% | 0.705 |
| 3 | 10 | -9.22 | 35.33% | 0.469 |
| 3 | 20 | -19.22 | 23.53% | 0.213 |
| 3 | 30 | -29.22 | 15.01% | 0.101 |
| 3 | 50 | -49.22 | 5.20% | 0.029 |
| 6 | 0 | -2.16 | 46.25% | 0.870 |
| 6 | 5 | -7.16 | 40.32% | 0.631 |
| 6 | 10 | -12.16 | 34.65% | 0.460 |
| 6 | 20 | -22.16 | 24.51% | 0.250 |
| 6 | 30 | -32.16 | 17.07% | 0.142 |
| 6 | 50 | -52.16 | 7.80% | 0.053 |
| 12 | 0 | -5.53 | 45.24% | 0.760 |
| 12 | 5 | -10.53 | 40.62% | 0.593 |
| 12 | 10 | -15.53 | 34.62% | 0.464 |
| 12 | 20 | -25.53 | 25.71% | 0.290 |
| 12 | 30 | -35.53 | 20.05% | 0.186 |
| 12 | 50 | -55.53 | 11.23% | 0.084 |

bps = basis points; RT = round trip. Profit factor is total positive net trade returns divided by the absolute total negative net trade returns.

## Direction breakdown before costs

| Horizon | Side | Trades | Mean gross bps | Gross win rate |
|---:|---|---:|---:|---:|
| 3 | Long | 1,153 | -0.45 | 47.70% |
| 3 | Short | 1,499 | +1.73 | 54.37% |
| 6 | Long | 1,051 | -3.37 | 42.82% |
| 6 | Short | 871 | -0.70 | 50.40% |
| 12 | Long | 589 | -5.77 | 42.95% |
| 12 | Short | 578 | -5.28 | 47.58% |

## Findings

1. The 3-bar candidate is only marginally positive before costs (+0.78 bps per trade on average; profit factor 1.067). A 5 bps round-trip cost assumption turns it negative (-4.22 bps; profit factor 0.705).
2. The 6- and 12-bar candidates lose on average before any costs. They are not viable under this fixed-horizon rule.
3. The 3-bar gross result is concentrated on the short side; its long trades are negative on average. This is a diagnostic observation, not permission to tune a short-only strategy on this already-used test window.
4. Even the no-cost 3-bar result is too small and fragile to establish a trading edge. Real execution friction, market impact and operational constraints can only worsen this simplified result.
5. Classification metrics beating a majority baseline did not translate into a robust cost-tolerant event result under this particular execution rule.

## Decision

- Do not deploy any horizon or issue live/paper signals from this experiment.
- Do not tune direction filters, probability thresholds, stop-losses or targets on this same test period.
- Preserve the 3-bar candidate only as a research hypothesis; the current evidence does not support trading it.
- A proper next evaluation requires a new untouched chronological period, timestamped predictions generated before execution, and broker-specific cost inputs.

## Limitations

- This simulation uses a previously consumed test window and is therefore not independent validation.
- The 5/10/20/30/50 bps values are sensitivity assumptions, not verified brokerage schedules.
- Brokerage, STT, exchange transaction charges, GST, SEBI charges, stamp duty, spread, slippage, impact, borrow/short availability and rejected/partial orders are not itemized.
- No intrabar stop/target path, latency, queue position, liquidity filter, price bands or corporate-action adjustment is modelled.
- The fixed-horizon close exit is a deliberately simple rule; it is not optimized and may not represent a viable execution policy.
- Trade-level averages do not represent a portfolio equity curve or account-level drawdown because capital allocation and concurrent exposure were not specified.
- The classifier's test predictions were generated using training-fitted preprocessing and model parameters; no model fitting used test rows. The subsequent event simulation does not restore the test period's independence.

## Reproducibility

The reusable event simulator is implemented in engine/ml_cost_aware_backtest.py. Unit tests cover next-bar entry, fixed-horizon exit, cost deduction, gap/session safety, non-overlap, neutral skips and input validation.
