"""Transparent technical indicators; no future data is used."""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close, high, low = df["close"], df["high"], df["low"]
    df["sma_20"] = close.rolling(20, min_periods=20).mean()
    df["sma_50"] = close.rolling(50, min_periods=50).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14, min_periods=14).mean()

    typical = (high + low + close) / 3
    volume = df["volume"].fillna(0)
    if isinstance(df["timestamp"].iloc[0], pd.Timestamp):
        session = df["timestamp"].dt.date
        cumulative_pv = (typical * volume).groupby(session).cumsum()
        cumulative_volume = volume.groupby(session).cumsum().replace(0, np.nan)
        df["vwap"] = cumulative_pv / cumulative_volume
    else:
        df["vwap"] = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    return df
