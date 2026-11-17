# Phase 8.20 — Cost-aware fixed-horizon event simulation

## Status

**Implementation and runner tests pass GitHub Actions (run #213). Corrected empirical results are pending. This phase is not approved for deployment.**

This phase converts frozen model predictions into retrospective, non-overlapping event trades. It is diagnostic research, not independent validation: the evaluation window was already used for Phase 8.19 model reporting.

## Superseded results notice

The first simulator version held each event one candle longer than its documented horizon. The simulator and unit tests have since been corrected. All previously published trade counts, returns, win rates, profit factors, and direction statistics were generated with the faulty timing and are withdrawn. They must not be quoted, used for model selection, or used to claim a trading edge.

The corrected simulator and reproducible runner are now in the PR branch. A successful code test does not establish empirical performance. The corrected full-dataset experiment has **not** yet been run against the Phase 8.13 archive, so no replacement performance figures are reported here.

## Frozen experiment assumptions

- Source: Phase 8.13 complete-session 5-minute research dataset.
- Historical window: the Phase 8.19 evaluation period, 2026-07-22 through 2026-07-31. This is a previously consumed window, not an untouched holdout.
- Model: balanced multinomial LogisticRegression, C=1.0, lbfgs, max_iter=250, random_state=42.
- Preprocessing: median imputation with missingness indicators and StandardScaler, fitted on labelled training rows only.
- Feature sets: 3 bars = volatility (4) + price action (12) + volume (2); 6 and 12 bars = volatility only (4).
- Prediction: frozen argmax class prediction; neutral predictions do not trade.
- Signal timing: prediction becomes available only after the signal candle closes.
- Entry: next 5-minute candle open.
- Exit: close of the h-th full candle after entry. For a signal at index t, entry is t+1 and exit is t+h.
- Direction: +1 = long; -1 = short. Short events remain hypothetical and require separate borrow/availability assumptions before any real-world interpretation.
- Safety: no session crossing; any missing 5-minute interval in the holding window invalidates the event.
- Overlap: at most one open event per symbol; ignore signals until the previous event exits.
- Costs: evaluate explicit round-trip sensitivity assumptions of 0, 5, 10, 20, 30 and 50 bps. These are scenario inputs, not broker-specific fee estimates.
- No stop-loss, target, trailing exit, portfolio sizing, compounding, live quote, or order-placement logic is included.

## Reproduction

From the repository root, run:

```bash
python scripts/phase_8_20_cost_aware_backtest.py path/to/phase_8_13_chronological_research_dataset.csv --manifest-json path/to/phase_8_13_dataset_manifest.json --output-dir phase_8_20_output
```

The runner validates chronological date splits and the 5-minute schema, checks that the Phase 8.13 manifest matches the frozen dataset identity (192,450 candles, 2,566 sessions, 50 symbols), exact train/validation/test date ranges, and the Phase 8.19 ±0.20% neutral band for all horizons, fits each horizon model on labelled training rows only, predicts labelled test rows, and applies the corrected event simulator. If the archived manifest records different bands, the runner stops rather than silently evaluating a different experiment. It writes test predictions per horizon and a JSON summary containing model settings, cost scenarios, symbol/date breakdowns, software version, and the input CSV SHA-256. Set `GITHUB_SHA` when running in CI to record the code revision.

The runner deliberately does not tune on validation or test data. The test partition is retained only for the already-defined Phase 8.19 diagnostic and remains a previously consumed window.

## Corrected results

**Pending rerun.** Do not populate this section until results have been regenerated from the corrected simulator and reconciled against the frozen Phase 8.13 manifest.

The rerun must report, for each horizon (3, 6 and 12 bars):
- valid event count and skipped-event reasons;
- long/short counts;
- gross and net mean/median return;
- gross and net win rate;
- profit factor and total return in basis points;
- cost-sensitivity table;
- symbol-level and date-level concentration;
- reproducible input hashes, code commit, model/configuration identifiers, and output hashes.

## Decision and guardrails

- No model, horizon, direction, or threshold is approved for live or paper-trading signals by this phase.
- Do not tune parameters on the already-consumed Phase 8.19 evaluation period.
- Do not treat classification metrics as evidence of tradable profitability.
- A future performance claim requires timestamped predictions generated before execution on a new untouched chronological holdout, realistic execution assumptions, and broker-specific cost inputs.
- Preserve all rejected and invalid events in audit outputs so exclusions can be reviewed.

## Known limitations

This simplified fixed-horizon simulator does not model portfolio-level capital constraints, concurrent exposure, spread, market impact, latency, queue position, partial fills, rejected orders, price bands, corporate actions, or the full statutory charge schedule. Trade-level averages are not an account equity curve or drawdown estimate.

## Reproducibility files

- Runner: `scripts/phase_8_20_cost_aware_backtest.py`
- Simulator: `engine/ml_cost_aware_backtest.py`
- Simulator tests: `tests/test_ml_cost_aware_backtest.py`
- Runner tests: `tests/test_phase_8_20_runner.py`
