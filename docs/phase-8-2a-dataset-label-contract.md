# Phase 8.2A — Historical ML Dataset and Label Contract

**Status:** Initial implementation proposal on a feature branch. Not merged or deployed.

## Universe and grain

- Universe: the repository's NIFTY 50 constituent list, refreshed and versioned against the applicable NSE Indices constituent file.
- Timeframes: 5-minute and 15-minute candles. Treat them as separate samples and separate model experiments; do not mix their horizon meanings.
- Row grain: one completed candle per symbol, timeframe and trading date.
- Preserve symbol, timeframe, timestamp, source/provider, and data-quality/session eligibility metadata on every row.

## Prediction target

For each row at candle close t, create forward close-to-close returns for 3, 6 and 12 future candles:

forward_return_h = 100 * (close[t+h] / close[t] - 1)

Direction classes are +1 (up), 0 (neutral), and -1 (down).

Initial neutral-band proposal:

0.5 * (ATR14[t] / close[t]) * sqrt(h) * 100

The multiplier is a research hyperparameter, not a proven optimum. Keep it fixed for the first baseline experiment and tune only inside chronological training/validation folds. ATR14 must use information available at time t.

Horizons are candle counts, not elapsed minutes: 3/6/12 bars mean 15/30/60 minutes on 5m data and 45/90/180 minutes on 15m data.

## Leakage and session rules

- Labels may not cross a trading-date boundary, symbol boundary, or timeframe boundary.
- Sort each symbol/timeframe/session independently by timestamp before shifting.
- Reject duplicate timestamps within a symbol/timeframe/session.
- Incomplete future horizons remain unlabelled; do not fill them with zero.
- Do not use future returns, post-event outcomes, or any feature computed with future candles as model inputs.
- Use only completed candles for feature rows.
- Keep the final contiguous date block as a locked test set. Use earlier dates for training and chronological validation.
- Apply a purge/embargo at least as long as the maximum 12-candle label horizon around split boundaries; do not let overlapping label windows leak across splits.
- Run 5m and 15m experiments separately before any combined comparison.

## Features: proposed groups

1. Candle geometry: body/range, upper/lower wick ratios, close location, gap/return, range expansion.
2. Existing pattern flags: deterministic detector outputs, after registry reconciliation.
3. Trend and structure: EMA distance/slope, confirmed swing context, HH/HL/LH/LL, BOS/CHoCH.
4. Volatility: ATR/close, rolling return volatility, range percentile.
5. Volume: relative volume and volume change, only where provider data quality supports it.
6. Context: time-of-day bucket and session position; avoid using raw symbol identity as a shortcut without a separate generalisation test.

All rolling features must be causal and computed within symbol/timeframe history. Session VWAP resets each trading day.

## Baselines and model comparison

1. Majority-class baseline.
2. Simple transparent rule baseline using only past information.
3. Logistic regression as an interpretable ML baseline.
4. Random Forest and gradient-boosted trees only after the baseline dataset is stable.

Report per class: sample count, precision, recall, F1, confusion matrix, and probability calibration. Also report per-symbol and per-timeframe results. Do not rank models on accuracy alone.

Classification quality is not equivalent to trading profitability. Any later strategy simulation must separately account for execution timing, slippage, brokerage, taxes, and other costs.

## Data quality gate

Before labels are accepted:
- Validate OHLC constraints, positive prices, timestamps, duplicates, chronological ordering, and volume anomalies.
- Confirm provider candle timestamp semantics and exchange-session alignment.
- Quarantine special sessions, closing-auction/transition records, misaligned timestamps, and estimated/missing bars from the first training cohort unless separately validated.
- Require adequate coverage across the NIFTY 50 universe; report coverage by symbol, date and timeframe. Do not silently treat absent candles as zero-return observations.

## Current repository limitation

The repository contains a NIFTY 50 symbol list and deterministic analysis code, but the inspected repo does not contain a complete raw NIFTY 50 5m/15m historical OHLCV corpus. Earlier analysis exports are summary CSVs, not substitutes for raw candle data. The full training dataset must be supplied or ingested from a verified provider before model fitting.

## Scope of this change

This phase adds only a leakage-aware forward-label helper and tests. It does not train a model, alter live scanning, change broker integration, or place orders.
