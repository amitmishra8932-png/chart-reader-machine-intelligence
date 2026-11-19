# Live Data Quality Audit

Use `tools/live_data_quality_audit.py` to audit one or more exported 5paisa candle CSVs. The tool is read-only and does not alter the exports or the live feed.

```bash
python tools/live_data_quality_audit.py exports/5m.csv exports/15m.csv
```

## Checks

- Required OHLCV schema and timezone-aware ISO-8601 timestamps
- Blank/non-numeric values, duplicate symbol/timestamp keys
- OHLC consistency, negative volume, and completion flags
- Observed per-symbol timestamp cadence (gaps are visible in the cadence report)
- Conflicting candles repeated across separate exports
- Independent reconstruction of 15-minute OHLCV from three consecutive 5-minute candles, where both timeframes are present

The tool infers timeframe from timestamp cadence within each file; filenames do not determine timeframe. It reports only intervals with all three correctly spaced 5-minute constituents. A successful comparison validates aggregation consistency, not tick completeness or equivalence with the provider's historical API.

## Interpretation limits

- A file containing only a short snapshot cannot prove full-session continuity.
- No missing rows in an export does not prove that no market ticks were missed before candle creation.
- Volume semantics must still be checked against the 5paisa feed contract and historical candles.
- Run the audit on a full regular-session export for all 50 NIFTY 50 symbols before treating the feed as ML-training grade.
