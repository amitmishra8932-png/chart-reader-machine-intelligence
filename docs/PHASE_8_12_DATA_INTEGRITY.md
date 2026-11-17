# Phase 8.12 — Data integrity and coverage

This phase adds explicit session-quality labels and quarantine flags to the Phase 8.11 pipeline.

## Outputs
- `phase_8_12_symbol_session_coverage.csv`: one row per symbol/timeframe/session date, with missing expected bars and off-grid timestamps.
- `phase_8_12_research_candidate_rows.csv`: eligible regular-signal rows retained with session-quality flags. No rows are silently removed for partial sessions.
- `phase_8_12_quarantine_classification.csv`: historical-only records receive descriptive flags and remain quarantined.
- `phase_8_12_audit_manifest.json`: assumptions, counts and guardrails.
- Existing Phase 8.11 event export is preserved.

## Expected grids (assumptions)
- 5m: 09:15–15:25 inclusive (75 timestamps).
- 15m: 09:15–15:15 inclusive (25 timestamps).

These are explicit audit assumptions, not verified provider timestamp semantics. Confirm 5paisa candle timestamp conventions before using the data for live decisions.

## Safety
No missing candles are synthesized. No timestamps are shifted. No quarantined row is promoted. Partial sessions remain available with quality labels. No model is trained and no trading signals or orders are produced.
