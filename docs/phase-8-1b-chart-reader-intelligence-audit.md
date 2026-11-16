# Phase 8.1B — Chart Reader Intelligence Audit

**Audit date:** 16 November 2026  
**Audited repository:** `amitmishra8932-png/chart-reader-machine-intelligence`  
**Audited baseline:** `ea1185775709ec0475610dd762f949f63c2167b3`  
**Scope:** Static source, dependency, registry, and connected-repository review. No production code or model files were changed.

## Executive finding

The audited `main` branch is an explainable, deterministic OHLCV analysis application. It does **not** contain a verifiable trained machine-learning training pipeline, saved model artifact, or model-inference pipeline.

The user clarified that the earlier Chart Reader V3 was **not ML-trained**. This resolves the historical V3 training question for the project as currently understood; there is no prior trained V3 model to recover. The separate **Amit AI Scanner V3.1** is a different project and must not be conflated with Chart Reader.

## What is implemented in this repository

- OHLCV ingestion and validation.
- Deterministic candle geometry and sequence detectors in `engine/patterns.py`.
- EMA/SMA, RSI, ATR, and session VWAP in `engine/indicators.py`.
- Rule-based trend and confirmed swing / BOS / CHoCH descriptions in `engine/market_structure.py`.
- Forward-return labels and descriptive summaries in `engine/outcomes.py`.
- Historical session-integrity and signal-eligibility policy.
- Manual paper-trade journal.

These components are useful chart-analysis building blocks, but they are not machine learning. Forward-return labels are research outcomes; they do not train a model.

## Pattern detector versus registry reconciliation

The pattern registry contains **26** entries. Its metadata currently reports:

- `implemented: true`: 6 entries.
- `implemented: false`: 20 entries.
- `detector_status: implemented`: 21 entries.
- `detector_status: catalogued_not_implemented`: 5 entries.
- `empirical_performance: not_tested`: all 26 entries.

The actual `engine/patterns.py` detector creates **23** pattern boolean columns:

- Doji family: Doji, long-legged doji, dragonfly doji, gravestone doji.
- Shape / trend-conditioned: hammer, hanging man, inverted hammer, shooting star.
- Two-candle: bullish engulfing, bearish engulfing, inside bar, outside bar, piercing line, dark cloud cover.
- Three-candle: morning star, evening star, three white soldiers, three black crows.
- Repeated-level: tweezer bottom, tweezer top.
- Other geometry: bullish marubozu, bearish marubozu, spinning top.

The catalogue's 26 entries not fully aligned with those 23 detector columns are:
- Bullish harami — catalogued, not detected.
- Bearish harami — catalogued, not detected.
- Rising three methods — catalogued, not detected.
- Falling three methods — catalogued, not detected.
- Marubozu — registry uses one generic entry while the detector emits bullish and bearish variants.

There is a broader metadata mismatch: many entries have `detector_status: implemented` while `implemented: false`. Registry metadata should not be used as proof of implementation until reconciled.

## ML artefact and dependency check

The inspected root `requirements.txt` contains Streamlit, pandas, NumPy, Plotly, and requests. It does not declare common ML frameworks such as scikit-learn, TensorFlow, or PyTorch.

Repository source searches for model training, model loading, `joblib`, `pickle`, `sklearn`, `tensorflow`, `torch`, `predict_proba`, and classifier/inference terms did not surface a model pipeline or artefact in the indexed default branch.

No saved model file, training notebook, feature/label training pipeline, model version metadata, or inference call was identified in the inspected application files.

A negative code search is not proof that no untracked, external, or unindexed artefact exists.

## Chart structure capability

`engine/market_structure.py` implements causal, deterministic calculations:
- EMA-based bullish / bearish / neutral trend state.
- Confirmed swing highs and lows using a configurable pivot window.
- HH / HL / LH / LL / equal-level labels.
- BOS / CHoCH events based on close beyond confirmed levels.

These are rule-based chart-structure descriptions. They are not a learned chart-image or chart-pattern classifier.

## Separate connected repository check

The connected private repository `amitmishra8932-png/amit-ai-intraday-scanner` was checked only as a possible source of related work. Its README describes heuristic screening scores and its dependencies do not show a chart-pattern ML stack. It is a separate intraday scanner and does not establish the provenance of earlier Chart Reader V3 training.

## Required work before a new ML phase

1. Reconcile the 26-entry registry with the 23 detector columns and correct metadata in a separate reviewed change.
2. Build a reproducible, leakage-safe historical dataset from validated OHLCV candles.
3. Define explicit prediction targets and forward horizons before training; keep chart-pattern labels distinct from market-condition filters.
4. Split data chronologically by date and prevent overlapping forward-label windows from crossing split boundaries.
5. Establish simple rule-based and statistical baselines before fitting an ML model.
6. Evaluate out-of-sample performance, calibration, sample counts, and realistic transaction-cost sensitivity. Do not claim predictive edge until results are reproducible.

## Current disposition

**Repository static audit: complete.**  
**Historical V3 training provenance: resolved — user confirms V3 was not ML-trained.**  
**Registry reconciliation: identified, not changed in this audit.**  
**New Chart Reader ML phase: not started; begin only after registry reconciliation and dataset/label design.**

No broker order functionality was added. No credentials or tokens were accessed or committed. No production code was changed by this audit.
