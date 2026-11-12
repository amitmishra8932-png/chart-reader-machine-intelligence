"""Simple, explainable single- and two-candle pattern detectors."""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    prev_o, prev_c = o.shift(1), c.shift(1)

    df["pattern_doji"] = (body / candle_range) <= 0.10
    df["pattern_hammer"] = (
        (lower >= body * 2) & (upper <= body * 0.6)
        & ((c - l) / candle_range >= 0.6)
    )
    df["pattern_shooting_star"] = (
        (upper >= body * 2) & (lower <= body * 0.6)
        & ((h - c) / candle_range >= 0.6)
    )
    df["pattern_bullish_engulfing"] = (
        (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    )
    df["pattern_bearish_engulfing"] = (
        (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
    )
    df["pattern_inside_bar"] = (h < h.shift(1)) & (l > l.shift(1))

    names = {
        "pattern_doji": "Doji",
        "pattern_hammer": "Hammer",
        "pattern_shooting_star": "Shooting star",
        "pattern_bullish_engulfing": "Bullish engulfing",
        "pattern_bearish_engulfing": "Bearish engulfing",
        "pattern_inside_bar": "Inside bar",
    }
    df["detected_patterns"] = df.apply(
        lambda row: ", ".join(name for col, name in names.items() if bool(row[col])),
        axis=1,
    )
    return df
