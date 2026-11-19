# Phase 8.21 — Quality-first forward excursion diagnostics

## Status
Research-only foundation. Adds a separate, gap-safe diagnostic for measuring favourable and adverse price excursions after each completed candle. It does not modify production features, existing directional targets, live market monitoring, broker authentication, paper trading or order handling.

## Why
The existing directional target is based on close-to-close return crossing a fixed neutral band. That describes the future close, but not the path taken to get there. It does not show whether a move first experienced substantial adverse movement or whether intraperiod extremes differed materially from the final close.

## Measurement contract
For each source candle and horizon H:
- The source candle close is the reference price.
- The observation window is the next H candles; the source candle is excluded.
- Every expected 5m or 15m timestamp must be present in the same symbol, timeframe and calendar date. A missing bar or session boundary invalidates the observation; the implementation never jumps a gap.
- Long MFE = (maximum future high / source close - 1) × 100.
- Long MAE = (minimum future low / source close - 1) × 100.
- Short MFE = (source close / minimum future low - 1) × 100.
- Short MAE = (source close / maximum future high - 1) × 100.
- Incomplete windows are NaN, not neutral or zero.
- Results are restored to the input row order.

## Quality and safety guardrails
1. Future-derived columns are outcome diagnostics only. They must never be included in model feature selection or live inference inputs.
2. High/low extrema do not reveal intrabar event ordering. A bar may contain both favourable and adverse extremes; this module does not assume which occurred first.
3. These measurements are not fills or a strategy backtest. They omit spread, slippage, fees, liquidity, queue position, latency and execution rules.
4. Do not select thresholds or models using the Phase 8.19 final test period. Keep that experiment as a historical benchmark and use fresh chronological data for new selection and evaluation.
5. Existing production app and 5paisa integration remain unchanged.

## Included
- engine/ml_excursions.py: independent forward MFE/MAE diagnostic.
- tests/test_ml_excursions.py: coverage for horizon semantics, gap and session isolation, input order, duplicates and invalid price geometry.

## Acceptance criteria
- Unit tests pass on the repository's supported Python environment.
- No production module imports the new diagnostic.
- No live signal, paper-trading rule or broker order is changed.
- A later research phase may use these outcomes to study quality filters only after a separate chronological protocol and realistic execution assumptions are specified.

## Not claimed
This phase does not improve predictive accuracy, establish profitability, validate a trade setup, train a new model or approve live recommendations.
