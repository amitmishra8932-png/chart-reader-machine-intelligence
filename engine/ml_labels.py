"""Leakage-aware forward labels for historical ML research.

Labels are generated independently for each symbol, timeframe and trading date.
They are research targets, not trading signals or recommendations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_forward_labels(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (3, 6, 12),
    volatility_multiplier: float = 0.5,
) -> pd.DataFrame:
    """Add forward-return and volatility-scaled direction labels.

    Each horizon is measured in candles, not minutes. A label compares the
    current close with the close horizon candles ahead, but only within the
    same symbol, timeframe and calendar trading date.

    Direction classes are -1 (down), 0 (neutral), +1 (up). The neutral band
    is volatility_multiplier * ATR(14)/close * sqrt(horizon). Rows without a
    complete future horizon or valid ATR remain unlabelled.
    """
    required = {"timestamp", "symbol", "timeframe", "close", "atr_14"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Historical dataset is empty.")
    if not horizons or any(
        not isinstance(h, (int, np.integer)) or isinstance(h, bool) or h <= 0
        for h in horizons
    ):
        raise ValueError("Horizons must be positive integers.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be unique.")
    if not np.isfinite(volatility_multiplier) or volatility_multiplier < 0:
        raise ValueError("volatility_multiplier must be finite and non-negative.")

    df = frame.copy()
    df["_label_original_order"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("All timestamps must be valid.")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["atr_14"] = pd.to_numeric(df["atr_14"], errors="coerce")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("Close prices must be valid and greater than zero.")
    if df[["symbol", "timeframe"]].isna().any().any():
        raise ValueError("Symbol and timeframe cannot be missing.")

    df["_session_date"] = df["timestamp"].dt.date
    group_keys = ["symbol", "timeframe", "_session_date"]
    if df.duplicated(group_keys + ["timestamp"]).any():
        raise ValueError("Duplicate timestamp within a symbol/timeframe session.")

    df = df.sort_values(group_keys + ["timestamp"]).copy()
    grouped = df.groupby(group_keys, sort=False, dropna=False)
    for horizon in horizons:
        future_close = grouped["close"].shift(-int(horizon))
        forward_return = (future_close / df["close"] - 1.0) * 100.0
        atr_pct = (df["atr_14"] / df["close"]) * 100.0
        neutral_band = volatility_multiplier * atr_pct * np.sqrt(int(horizon))

        valid = future_close.notna() & np.isfinite(neutral_band) & (df["atr_14"] > 0)
        target = pd.Series(pd.NA, index=df.index, dtype="Int8")
        target.loc[valid & (forward_return > neutral_band)] = 1
        target.loc[valid & (forward_return < -neutral_band)] = -1
        target.loc[valid & (forward_return.abs() <= neutral_band)] = 0

        df[f"forward_return_{horizon}"] = forward_return.where(valid)
        df[f"neutral_band_{horizon}_pct"] = neutral_band.where(valid)
        df[f"target_direction_{horizon}"] = target

    return (
        df.sort_values("_label_original_order")
        .drop(columns=["_label_original_order", "_session_date"])
    )
