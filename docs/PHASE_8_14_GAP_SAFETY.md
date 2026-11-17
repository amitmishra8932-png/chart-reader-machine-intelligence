# Phase 8.14 — Coverage diagnosis and gap-safe targets

## Findings from the Phase 8.12 audit export

The audit contains 6,200 symbol/session records per timeframe across 50 symbols and 124 session dates.

### 5-minute coverage
- Complete: 2,566 sessions.
- Closing-only incomplete: 3,545 sessions.
- Intraday gaps: 34 sessions.
- Intraday plus closing gaps: 55 sessions.
- Of the closing-only incomplete sessions, 2,761 omit exactly 15:20 and 15:25; 784 omit 15:15, 15:20 and 15:25.
- Beginning 3 August 2026, many dates show all 50 symbols missing the final two expected bars. This synchronized pattern is consistent with a systematic collection/export cutoff. It does not prove the upstream cause.

### 15-minute coverage
- Complete: 5,374 sessions.
- Closing-only incomplete: 793 sessions, predominantly missing 15:15.
- Intraday gaps: 17 sessions.
- Intraday plus closing gaps: 16 sessions.
- The pattern is less uniform than the 5-minute cutoff.

## Decisions

1. Keep Phase 8.13 strict complete-session selection unchanged.
2. Do not synthesize, forward-fill, shift, or infer missing OHLCV bars.
3. Do not admit partial sessions into the training dataset until the source/export cutoff is understood.
4. Make target creation gap-safe: a horizon is labelled only if its endpoint timestamp is exactly the requested number of 5-minute or 15-minute intervals after the source candle.
5. Fail closed for unsupported timeframe labels.
6. Keep all target horizons inside the same symbol, timeframe and calendar session.

## Code change

`engine/ml_targets.py` now validates elapsed timestamp continuity for each horizon. This closes a gap in the earlier implementation, which was session-bounded but could jump over an internal missing timestamp. Regression tests cover missing intraday bars, normal 15-minute cadence, session boundaries and unsupported timeframes.

## Remaining investigation

The Phase 8.12 candidate export does not preserve enough provenance to identify which original CSV/export job omitted the closing bars. The next data request should capture source filename/batch, retrieval time, provider timestamp semantics, and the final timestamp returned for each symbol/timeframe. Compare the raw 5paisa API response with the saved CSV before deciding whether this is an API limit, pagination issue, request-window issue, or local export truncation.

## Guardrails

No model is fitted. No partial session is promoted. No synthetic data is created. No trading signal or profitability claim is made. The expected timestamp grid remains an assumption until confirmed against provider documentation.
