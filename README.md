# Chart Reader — Machine Intelligence

An evidence-led stock-market chart research application built with Python and Streamlit. It supports user-supplied historical OHLCV files, 5paisa historical candles, and an authenticated read-only 5paisa live market monitor.

## Current capabilities

- CSV ingestion and OHLCV validation
- Candlestick chart with EMA 20 and EMA 50
- RSI 14, ATR 14 and session VWAP calculations
- Explainable candle-shape detectors: Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing and Inside Bar
- Basic EMA trend state and confirmed swing-point markers
- Forward-return outcome labels at 3, 6 and 12 candles
- Download of analysed candles
- 5paisa OAuth session authentication and historical NSE cash candles (5m/15m)
- 5paisa Xstream WebSocket live tick monitor with completed 5m/15m OHLCV aggregation for a selected NIFTY 50 watchlist
- Manual paper-trade journal with LONG/SHORT entries, stop/target validation, stop-risk and cash-exposure sizing, simulated exits, estimated-friction-adjusted P&L, performance metrics, and CSV export/restore
- Unit tests and GitHub Actions test workflow

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Required columns: `timestamp` (or `date`), `open`, `high`, `low`, `close`. `volume` is optional. Column names are case-insensitive. Prices must be positive and candle highs/lows must contain the open and close.

## 5paisa integration

The historical-data provider is isolated in `providers/fivepaisa.py`; live streaming is isolated in `providers/fivepaisa_live.py`. The live adapter uses the official Xstream WebSocket `MarketFeedV3` subscription for NSE cash instruments and aggregates received ticks into 5m/15m candles. The Streamlit Live Market tab is read-only. Before relying on it, verify the user's 5paisa market-data permissions, registered OAuth callback, live WebSocket access and actual tick timestamps. Do not commit API keys, passwords, encryption keys, request tokens or access tokens. Store long-lived credentials in Streamlit Community Cloud Secrets; the access token remains in the current Streamlit session.

## Safety and research notes

- Read-only research application; no order placement.
- Paper trading is manual simulation only: prices are entered by the user; there are no live quotes or automated executions.
- Paper-trade records are session-based; download and restore the CSV ledger to preserve them across app restarts.
- Estimated friction is a configurable approximation, not an exact brokerage, tax or exchange-fee calculation.
- Detected candle patterns are descriptions, not trading signals.
- Forward-return summaries are not strategy backtests and do not include costs, slippage or execution constraints.
- No performance or profitability claim is made without out-of-sample validation.
- Historical and live data must be timestamped and source-labelled.

## Roadmap

1. Expand CSV edge-case tests and data-quality reporting.
2. Verify the 5paisa OAuth callback, historical API access and live WebSocket connection with the account.
3. Validate live tick timestamps, volume deltas and candle boundaries against the 5paisa terminal.
4. Recover the frozen Phase 8.13 dataset and rerun corrected cost-aware historical evaluation.
5. Complete a separate untouched chronological holdout and controlled paper-trading observations.
6. Add model-driven live signal display only after independent validation; keep order placement out of scope.
