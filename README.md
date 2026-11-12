# Chart Reader — Machine Intelligence

An evidence-led stock-market chart research application built with Python and Streamlit. The initial build supports user-supplied historical OHLCV CSV files while 5paisa market-data access is being confirmed.

## Current capabilities

- CSV ingestion and OHLCV validation
- Candlestick chart with EMA 20 and EMA 50
- RSI 14, ATR 14 and session VWAP calculations
- Explainable candle-shape detectors: Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing and Inside Bar
- Basic EMA trend state and confirmed swing-point markers
- Forward-return outcome labels at 3, 6 and 12 candles
- Download of analysed candles
- Unit tests and GitHub Actions test workflow

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## CSV format

Required columns: `timestamp` (or `date`), `open`, `high`, `low`, `close`. `volume` is optional. Column names are case-insensitive. Prices must be positive and candle highs/lows must contain the open and close.

## 5paisa integration

The provider boundary is isolated in `providers/fivepaisa.py`. Authentication and endpoint calls will only be implemented against the current official 5paisa Xstream documentation and after the account's market-data permissions and callback configuration are verified. Do not commit API keys, passwords, encryption keys, request tokens or access tokens. Store secrets in Streamlit Community Cloud Secrets.

## Safety and research notes

- Read-only research application; no order placement.
- Detected candle patterns are descriptions, not trading signals.
- Forward-return summaries are not strategy backtests and do not include costs, slippage or execution constraints.
- No performance or profitability claim is made without out-of-sample validation.
- Historical and live data must be timestamped and source-labelled.

## Roadmap

1. Expand CSV edge-case tests and data-quality reporting.
2. Verify 5paisa OAuth and historical candle API access.
3. Add a broker adapter with normalized candle schema.
4. Add historical replay and leakage-safe evaluation.
5. Add live read-only monitoring.
6. Run walk-forward model experiments and paper validation.
