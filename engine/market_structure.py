"""Basic trend and confirmed swing-point analysis."""
from __future__ import annotations
import pandas as pd


def add_market_structure(frame: pd.DataFrame, pivot_window: int = 3) -> pd.DataFrame:
    if pivot_window < 1:
        raise ValueError("pivot_window must be at least 1")
    df = frame.copy()
    if "ema_20" not in df or "ema_50" not in df:
        raise ValueError("Add indicators before market-structure analysis.")

    df["trend_state"] = "Neutral"
    bullish = (df["close"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"])
    bearish = (df["close"] < df["ema_20"]) & (df["ema_20"] < df["ema_50"])
    df.loc[bullish, "trend_state"] = "Bullish"
    df.loc[bearish, "trend_state"] = "Bearish"

    # A pivot is marked only after pivot_window candles have closed to its right.
    width = 2 * pivot_window + 1
    rolling_high = df["high"].rolling(width, center=True, min_periods=width).max()
    rolling_low = df["low"].rolling(width, center=True, min_periods=width).min()
    raw_high = df["high"].eq(rolling_high)
    raw_low = df["low"].eq(rolling_low)
    df["swing_high_confirmed"] = raw_high.shift(pivot_window).fillna(False).astype(bool)
    df["swing_low_confirmed"] = raw_low.shift(pivot_window).fillna(False).astype(bool)
    return df
