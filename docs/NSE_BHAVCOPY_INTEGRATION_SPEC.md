# NSE CM UDiFF Bhavcopy Integration — Common Specification

Status: isolated development only; production deployment and merge are prohibited without owner review.

## Authoritative source
- Exchange: National Stock Exchange of India (NSE), Capital Market (CM) segment.
- Report: **CM-UDiFF Common Bhavcopy Final (ZIP)**.
- Official report catalogue: https://www.nseindia.com/all-reports
- Archive URL pattern: https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
- File name date is YYYYMMDD; the CSV's `TradDt` is authoritative for the trading date.
- Legacy CM Bhavcopy CSV reports were discontinued from 8 July 2024. Do not use the legacy report as a silent fallback.

## Source fields and normalized contract
Preserve every source column and the complete source row. Normalize the following UDiFF tags where present:

| Normalized field | UDiFF tag |
|---|---|
| trade_date | TradDt |
| business_date | BizDt |
| segment | Sgmt |
| source | Src |
| instrument_type | FinInstrmTp |
| isin | ISIN |
| symbol | TckrSymb |
| series | SctySrs |
| instrument_name | FinInstrmNm |
| open | OpnPric |
| high | HghPric |
| low | LwPric |
| close | ClsPric |
| last_price | LastPric |
| prev_close | PrvsClsgPric |
| average_price | AvrgPric (optional; preserve if supplied) |
| traded_qty | TtlTradgVol |
| turnover | TtlTrfVal |
| trade_count | TtlNbOfTxsExctd |

Do not derive delivery quantity or delivery percentage from this Bhavcopy. Those belong to a separate NSE report. Do not treat turnover as volume. Do not infer missing values.

## Ingestion contract
1. Download only the official NSE CM final ZIP for a requested trading date.
2. Reject non-ZIP, empty ZIP, unsafe archive paths, multiple ambiguous CSV members, malformed CSV, wrong segment/source, or a mismatch between requested date, filename date and row `TradDt`.
3. Preserve original ZIP bytes in a date-partitioned raw archive and calculate SHA-256.
4. Parse with explicit UDiFF tag mapping; tolerate extra future source columns but fail closed if required tags are absent.
5. Validate symbol/series, date, numeric fields and OHLC consistency. Quarantine invalid rows with reason and raw payload; do not silently drop them.
6. Store valid rows transactionally. Idempotency key: `(trade_date, symbol, series)`. Re-import is an upsert, never a duplicate append.
7. Record ingestion run, URL, filename, SHA-256, parser version, row counts and status.
8. Never mark a date complete when download, validation, or database commit fails.

## Storage isolation
Each repository owns its own SQLite database and raw archive directory. No shared database, file path, credentials, journals, model artefacts or cross-repository writes.
- Scanner: own Bhavcopy store; downstream research scope is the existing official NIFTY 500 cohort.
- Chart Reader: own Bhavcopy store; downstream research scope is the existing NIFTY 50 cohort.
- Existing signal journals, forecast archives, paper-trading records, model datasets and performance metrics are read-only to this integration.
- Do not migrate, rename, delete, overwrite or backfill existing application data.

## Data use boundaries
Bhavcopy is end-of-day data. It may enrich daily context, historical features and next-session research; it must not be represented as intraday/live data or used to fabricate 5m/15m candles. Existing scoring and ML features remain unchanged in this integration stage.

## Scheduling and operational controls
- One final-file attempt after NSE publishes the completed session report; retries use bounded exponential backoff.
- No polling during market hours and no minute-level repeated downloads.
- Skip weekends/holidays only after confirming no final report exists; absence is not a successful ingestion.
- Manual date import is allowed for controlled backfill.
- Keep download timeout, response-size limit and archive member-size limit.
- No paid provider, broker order route, auto-order or new credential is introduced.

## Test and release gates
- Unit tests for ZIP validation, tag mapping, date validation, OHLC checks, malformed/empty files, quarantine and idempotent upsert.
- Fixture-based tests must not depend on NSE availability.
- Integration smoke test against one official, completed trading-day ZIP; record its hash and reconciliation counts.
- Verify source rows against the original CSV and compare selected symbols/prices with the NSE report.
- Run all pre-existing tests and compare existing journals/metrics before and after.
- Keep branch isolated. No production deployment, PR merge or database migration until explicit owner review and approval.

## Implementation note
The NSE report catalogue currently identifies CM-UDiFF Common Bhavcopy Final (ZIP) as the current report and marks the old CM Bhavcopy CSV and CM Common Bhavcopy CSV reports discontinued from 8 July 2024. Reconfirm report naming and archive availability during each integration test; do not hardcode a web-page download token.


## Raw archive immutability and manual import
- Store raw archives under `raw_bhavcopy/YYYY/MM/DD/<sha256>.zip`; never overwrite an archive for a date when the downloaded bytes differ.
- Both apps expose a manual ZIP upload path as a fallback when NSE blocks automated HTTP downloads. Uploaded archives pass through the same parser, date/filename checks, SHA-256, quarantine and idempotent storage path as downloaded archives.
- A manual upload is not considered an NSE-origin authenticated transport; its file content is still validated against the NSE CM-UDiFF contract.
