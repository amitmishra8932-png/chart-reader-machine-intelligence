"""Autonomous, explainable multi-timeframe trade decision support.

The engine may recommend LONG, SHORT, or NO TRADE. It never sends orders.
A machine-learning overlay is optional and is accepted only after a chronological
holdout beats a majority-class baseline. Rule scores are transparent heuristics,
not probabilities or guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from engine.features import add_features
from engine.indicators import add_indicators
from engine.market_structure import add_market_structure


FEATURES = (
    "feature_return_1", "feature_return_3", "feature_return_6",
    "feature_body_to_range", "feature_upper_wick_to_range",
    "feature_lower_wick_to_range", "feature_close_location",
    "feature_range_atr", "feature_body_atr",
    "feature_realized_volatility_20", "feature_relative_volume_20",
    "feature_volume_zscore_20", "feature_sma_20_distance_pct",
    "feature_ema_20_distance_pct", "feature_sma_50_distance_pct",
    "feature_ema_50_distance_pct",
)


def _normalise_candles(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    if timeframe not in {"5m", "15m"}:
        raise ValueError("timeframe must be '5m' or '15m'")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    if result["timestamp"].isna().any():
        raise ValueError("Candle timestamps must be valid.")
    for col in ("open", "high", "low", "close", "volume"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if not np.isfinite(result[list(("open", "high", "low", "close", "volume"))].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite.")
    if (result[["open", "high", "low", "close"]] <= 0).any().any() or (result["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume cannot be negative.")
    invalid = (
        (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError("Invalid OHLC candle geometry.")
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps must be resolved.")
    return result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_indicators(frame)
    result = add_market_structure(result)
    result = add_features(result)
    return result


def analyse_trade_setup(
    candles_5m: pd.DataFrame,
    candles_15m: pd.DataFrame,
    *,
    symbol: str,
    ml_model: Any | None = None,
    ml_feature_columns: list[str] | None = None,
    ml_approved: bool = False,
    min_score: int = 7,
    min_reward_risk: float = 1.8,
) -> dict[str, Any]:
    """Make one end-of-bar decision using completed 5m and 15m candles."""
    five = _enrich(_normalise_candles(candles_5m, "5m"))
    fifteen = _enrich(_normalise_candles(candles_15m, "15m"))
    if len(five) < 55 or len(fifteen) < 55:
        return _no_trade(symbol, "Insufficient history: at least 55 completed bars per timeframe are required.")

    a, b = five.iloc[-1], fifteen.iloc[-1]
    if not np.isfinite([a["atr_14"], b["ema_20"], b["ema_50"], a["ema_20"], a["ema_50"]]).all() or a["atr_14"] <= 0:
        return _no_trade(symbol, "Indicators are not ready or ATR is invalid.")

    long_score = 0
    short_score = 0
    long_reasons: list[str] = []
    short_reasons: list[str] = []
    close = float(a["close"])
    atr = float(a["atr_14"])

    if b["close"] > b["ema_20"] > b["ema_50"]:
        long_score += 2
        long_reasons.append("15m trend is bullish (close > EMA20 > EMA50)")
    elif b["close"] < b["ema_20"] < b["ema_50"]:
        short_score += 2
        short_reasons.append("15m trend is bearish (close < EMA20 < EMA50)")

    if a["close"] > a["ema_20"] > a["ema_50"]:
        long_score += 2
        long_reasons.append("5m trend confirms bullish alignment")
    elif a["close"] < a["ema_20"] < a["ema_50"]:
        short_score += 2
        short_reasons.append("5m trend confirms bearish alignment")

    if str(a["structure_event"]).endswith("bullish"):
        long_score += 2
        long_reasons.append("5m confirmed bullish BOS/CHoCH")
    elif str(a["structure_event"]).endswith("bearish"):
        short_score += 2
        short_reasons.append("5m confirmed bearish BOS/CHoCH")

    rsi = float(a["rsi_14"]) if pd.notna(a["rsi_14"]) else np.nan
    if np.isfinite(rsi) and 52 <= rsi <= 70:
        long_score += 1
        long_reasons.append(f"5m RSI supports bullish momentum ({rsi:.1f})")
    elif np.isfinite(rsi) and 30 <= rsi <= 48:
        short_score += 1
        short_reasons.append(f"5m RSI supports bearish momentum ({rsi:.1f})")

    relative_volume = float(a["feature_relative_volume_20"]) if pd.notna(a["feature_relative_volume_20"]) else np.nan
    if np.isfinite(relative_volume) and relative_volume >= 1.3:
        if close > float(a["open"]):
            long_score += 1
            long_reasons.append(f"Bullish candle has elevated relative volume ({relative_volume:.2f}x)")
        elif close < float(a["open"]):
            short_score += 1
            short_reasons.append(f"Bearish candle has elevated relative volume ({relative_volume:.2f}x)")

    if pd.notna(a["vwap"]):
        if close > float(a["vwap"]):
            long_score += 1
            long_reasons.append("Price is above session VWAP")
        elif close < float(a["vwap"]):
            short_score += 1
            short_reasons.append("Price is below session VWAP")

    prior_high = float(five["high"].iloc[-21:-1].max())
    prior_low = float(five["low"].iloc[-21:-1].min())
    if close > prior_high:
        long_score += 1
        long_reasons.append("5m close broke the prior 20-bar high")
    if close < prior_low:
        short_score += 1
        short_reasons.append("5m close broke the prior 20-bar low")

    ml_long = ml_short = None
    if ml_model is not None and ml_approved and ml_feature_columns:
        row = a.reindex(ml_feature_columns).to_frame().T
        try:
            probabilities = ml_model.predict_proba(row)[0]
            classes = list(ml_model.classes_)
            ml_long = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
            ml_short = float(probabilities[classes.index(-1)]) if -1 in classes else 0.0
            if ml_long >= 0.60:
                long_score += 2
                long_reasons.append(f"Validated ML overlay favours LONG ({ml_long:.0%})")
            if ml_short >= 0.60:
                short_score += 2
                short_reasons.append(f"Validated ML overlay favours SHORT ({ml_short:.0%})")
        except (ValueError, IndexError, AttributeError):
            ml_long = ml_short = None

    if long_score == short_score or max(long_score, short_score) < min_score:
        reason = "No side passed the minimum confluence threshold." if long_score != short_score else "Conflicting or balanced evidence; no directional edge."
        return _no_trade(symbol, reason, long_score, short_score, ml_long, ml_short)

    side = "LONG" if long_score > short_score else "SHORT"
    score = max(long_score, short_score)
    reasons = long_reasons if side == "LONG" else short_reasons
    if side == "LONG":
        stop = min(close - 1.2 * atr, float(five["low"].iloc[-6:].min()))
        risk = close - stop
        target = close + max(2.0 * risk, min_reward_risk * risk)
    else:
        stop = max(close + 1.2 * atr, float(five["high"].iloc[-6:].max()))
        risk = stop - close
        target = close - max(2.0 * risk, min_reward_risk * risk)

    if not np.isfinite([stop, target, risk]).all() or risk <= 0 or target <= 0:
        return _no_trade(symbol, "Risk levels could not be calculated safely.", long_score, short_score, ml_long, ml_short)
    reward_risk = abs(target - close) / risk
    if reward_risk < min_reward_risk:
        return _no_trade(symbol, "Calculated reward/risk is below the required threshold.", long_score, short_score, ml_long, ml_short)

    return {
        "symbol": str(symbol).upper(),
        "decision": side,
        "status": "SETUP_CANDIDATE",
        "time": pd.Timestamp(a["timestamp"]).isoformat(),
        "timeframe": "5m entry / 15m context",
        "entry_reference": round(close, 2),
        "stop_loss": round(stop, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk, 2),
        "reward_risk": round(reward_risk, 2),
        "score": int(score),
        "max_score_without_ml": 9,
        "ml_long_probability": ml_long,
        "ml_short_probability": ml_short,
        "ml_used": ml_long is not None and ml_short is not None,
        "reasons": reasons,
        "warnings": ["Signal is generated after candle close; entry requires next-bar confirmation.", "No slippage, liquidity or brokerage guarantee is implied."],
    }


def _no_trade(symbol: str, reason: str, long_score: int = 0, short_score: int = 0,
              ml_long: float | None = None, ml_short: float | None = None) -> dict[str, Any]:
    return {
        "symbol": str(symbol).upper(), "decision": "NO TRADE", "status": "NO_SETUP",
        "reason": reason, "score_long": int(long_score), "score_short": int(short_score),
        "ml_long_probability": ml_long, "ml_short_probability": ml_short,
        "ml_used": ml_long is not None and ml_short is not None,
    }


def train_direction_model(
    candles: pd.DataFrame,
    *,
    neutral_band_pct: float = 0.20,
    min_train_rows: int = 1000,
    min_validation_rows: int = 200,
) -> tuple[Any | None, dict[str, Any]]:
    """Train a conservative RandomForest overlay on a chronological holdout.

    Targets use the close three bars ahead within the same symbol/timeframe/date.
    The model is disabled unless holdout balanced accuracy exceeds the majority
    class baseline by at least two percentage points.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import Pipeline

    required = {"timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"}
    if required.difference(candles.columns):
        raise ValueError(f"ML training requires columns: {sorted(required)}")
    if not np.isfinite(neutral_band_pct) or neutral_band_pct <= 0:
        raise ValueError("neutral_band_pct must be finite and positive.")
    data = candles.copy()
    # Train only on the 5m entry horizon; mixing 5m and 15m labels would
    # make a three-bar target represent two different holding periods.
    data = data[data["timeframe"].astype(str).str.lower() == "5m"].copy()
    if data.empty:
        return None, {"approved": False, "reason": "Training requires 5m entry candles.", "rows": 0}
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    if data["timestamp"].isna().any():
        raise ValueError("ML training timestamps must be valid.")
    data = data.sort_values(["symbol", "timeframe", "timestamp"], kind="mergesort").reset_index(drop=True)
    # Indicators and VWAP must be computed independently per symbol/timeframe.
    enriched = [
        _enrich(_normalise_candles(group, "5m"))
        for _, group in data.groupby(["symbol", "timeframe"], sort=False)
    ]
    data = pd.concat(enriched, ignore_index=True).sort_values(
        ["symbol", "timeframe", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = data.groupby(
        [data["symbol"].astype(str), data["timeframe"].astype(str), data["timestamp"].dt.date],
        sort=False,
    )
    data["future_close_3"] = grouped["close"].shift(-3)
    forward = (data["future_close_3"] / data["close"] - 1.0) * 100.0
    data["target"] = np.select(
        [forward > neutral_band_pct, forward < -neutral_band_pct], [1, -1], default=0
    ).astype(float)
    data.loc[data["future_close_3"].isna(), "target"] = np.nan
    data = data.dropna(subset=["target"]).copy()
    dates = sorted(data["timestamp"].dt.date.unique())
    if len(dates) < 5:
        return None, {"approved": False, "reason": "At least five distinct sessions are required.", "rows": len(data)}
    split_index = max(1, int(len(dates) * 0.8))
    validation_dates = set(dates[split_index:])
    train = data[~data["timestamp"].dt.date.isin(validation_dates)]
    validation = data[data["timestamp"].dt.date.isin(validation_dates)]
    if len(train) < min_train_rows or len(validation) < min_validation_rows:
        return None, {"approved": False, "reason": "Insufficient training or holdout rows.", "train_rows": len(train), "validation_rows": len(validation)}
    features = [name for name in FEATURES if name in data.columns]
    train = train.dropna(subset=["target"])
    if train["target"].nunique() < 2 or validation["target"].nunique() < 2:
        return None, {"approved": False, "reason": "Training and validation must each contain at least two outcome classes."}
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
    ])
    pipeline.fit(train[features], train["target"].astype(int))
    predicted = pipeline.predict(validation[features])
    score = float(balanced_accuracy_score(validation["target"].astype(int), predicted))
    majority_class = int(validation["target"].mode().iloc[0])
    majority_prediction = np.full(len(validation), majority_class, dtype=int)
    baseline = float(balanced_accuracy_score(validation["target"].astype(int), majority_prediction))
    approved = score >= baseline + 0.02
    report = {
        "approved": bool(approved), "reason": "Holdout gate passed." if approved else "Holdout balanced accuracy did not beat the majority-class balanced-accuracy baseline by 2 percentage points.",
        "train_rows": int(len(train)), "validation_rows": int(len(validation)),
        "first_validation_date": str(min(validation_dates)), "last_validation_date": str(max(validation_dates)),
        "balanced_accuracy": round(score, 4), "majority_baseline": round(baseline, 4),
        "features": features, "target_definition": f"3-bar close return; LONG > {neutral_band_pct:.2f}%, SHORT < -{neutral_band_pct:.2f}%, otherwise neutral",
    }
    return (pipeline if approved else None), report


def rank_trade_setups(results: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Return only qualified candidates, ordered by score then reward/risk."""
    if limit < 1:
        raise ValueError("limit must be positive")
    eligible = [item for item in results if item.get("decision") in {"LONG", "SHORT"} and item.get("status") == "SETUP_CANDIDATE"]
    return sorted(eligible, key=lambda item: (item.get("score", 0), item.get("reward_risk", 0)), reverse=True)[:limit]
