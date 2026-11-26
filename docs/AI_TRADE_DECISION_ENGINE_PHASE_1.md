# AI Trade Decision Engine — Phase 1

## Purpose
Provide an explainable decision-support engine that independently evaluates completed 5-minute entries against 15-minute context and returns LONG, SHORT, or NO TRADE. It is read-only and cannot place orders.

## Decision process
- Validates candle schema, timestamps, duplicate keys, OHLC geometry and volume.
- Computes EMA, RSI, ATR, VWAP, causal features and confirmed market structure.
- Scores 15m trend, 5m trend, confirmed BOS/CHoCH, RSI, relative volume, VWAP and 20-bar breakout.
- Requires a minimum confluence score, directional agreement, valid risk levels and a minimum 1.8R target.
- Produces a reference close, stop, target, score and evidence. Entry requires next-bar confirmation.
- Ranks at most three qualified setups; no setup is a valid decision.

## Multi-timeframe integrity
- The latest 5m entry and 15m context bars are checked using interval end times.
- Timezone-naive, misaligned, cross-session, stale, and future-dated context is rejected with NO TRADE.
- The AI Trade Desk shows completed-bar counts per subscribed symbol so warm-up status is visible.
- Historical candles are not silently mixed into the live buffer until 5paisa timestamp timezone and candle-boundary semantics are verified against an authenticated account. The historical adapter retains the exact provider timestamp text in `provider_timestamp_raw` beside the parsed timestamp to support that audit.
- Historical replay supports an explicit CAS-eligible mode: it excludes auction bars and applies the dated continuous-session close (15:15 before 2026-12-07; 15:30 from 2026-12-07). The caller must supply correct per-security CAS eligibility from the exchange security master.
- Configured special-session dates (including the 2026-11-08 Muhurat session) are excluded from the regular-session replay; they require a separate special-session research path.

## Machine-learning overlay
The optional RandomForest overlay trains on 3-bar forward returns grouped by symbol, timeframe and session. It uses a chronological 80/20 session holdout, training-only median imputation, and a conservative approval gate: holdout balanced accuracy must exceed the validation majority-class baseline by at least 2 percentage points. If the gate fails, the model is disabled and cannot influence recommendations.

The overlay is not automatically persisted. A new dataset must be supplied and trained in a new app session. Probabilities are model class scores and are not guaranteed calibrated probabilities.

## Important limitations
- This phase is a decision engine, not proof of profitability.
- Rule scores are confluence points, not win probabilities.
- A signal is a candidate, not an instruction to trade; confirm next-bar price, spread, liquidity and market conditions.
- It does not model exact brokerage, taxes, slippage, market impact or order fills.
- Live-feed quality and 5paisa candle aggregation must be independently verified. Uploaded RELIANCE samples show naive timestamps and are consistent with start-labelled candles when 5m bars are aggregated to 15m; timezone and provider convention still require authenticated verification.
- No automatic order placement, broker order endpoints, or unattended trading are included.
- Keep this branch separate from production until code review, CI, historical replay and forward paper-trading gates pass.
