# Phase 8.1B — Chart Reader Intelligence Audit

**Audit date:** 16 November 2026  
**Audited repository:** `amitmishra8932-png/chart-reader-machine-intelligence`  
**Audited baseline:** `ea1185775709ec0475610dd762f949f63c2167b3`  
**Scope:** Static source, dependency, registry, and connected-repository review. No production code or model files were changed.

## Executive finding

The audited `main` branch is an explainable, deterministic OHLCV analysis application. It does **not** contain a verifiable trained machine-learning training pipeline, saved model artifact, or model-inference pipeline.

This is a finding about the inspected repository and baseline only. It does not prove that model training never occurred in earlier V3/V3.1 deployments, another repository, an unmerged branch, a local workspace, or a separate training environment.

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

## Required reconciliation before Phase 8.2

1. Locate the earlier V3 / V3.1 Chart Reader source, training notebooks, model artefacts, and training datasets, if they still exist.
2. Identify the exact model family, feature inputs, target labels, training date range, validation split, and saved model version.
3. Verify that inference on unseen candles actually loads the saved model rather than calling deterministic pattern rules.
4. Reconcile the 26-entry registry with the 23 detector columns and correct metadata in a separate reviewed change.
5. Measure chart-pattern intelligence alone, indicator filters alone, and their combination using chronological, leakage-safe out-of-sample evaluation.
6. Do not claim ML capability or predictive edge until reproducible artefacts and evaluation results support it.

## Current disposition

**Repository static audit: complete.**  
**Historical V3 training provenance: pending source/artifact recovery.**  
**Registry reconciliation: identified, not changed in this audit.**  
**Phase 8.2: hold until the historical ML question and registry mismatch are resolved.**

No broker order functionality was added. No credentials or tokens were accessed or committed. No production code was changed by this audit.
