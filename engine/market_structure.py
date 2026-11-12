"""Causal swing, trend and market-structure event analysis.

Swing points are only made available on the bar where the right-side
confirmation window has completed. BOS/CHoCH events use confirmed levels and
bar closes; they are descriptive events, not trading instructions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_structure(frame: pd.DataFrame, pivot_window: int = 3) -> pd.DataFrame:
    if pivot_window < 1:
        raise ValueError("pivot_window must be at least 1")
    df = frame.copy()
    required = {"high", "low", "close", "ema_20", "ema_50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for market structure: {sorted(missing)}")

    bullish = (df["close"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"])
    bearish = (df["close"] < df["ema_20"]) & (df["ema_20"] < df["ema_50"])
    df["trend_state"] = np.select([bullish, bearish], ["Bullish", "Bearish"], default="Neutral")

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n = len(df)

    # Event columns are stamped at confirmation time, never backdated to pivot time.
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    swing_high_label = np.full(n, "", dtype=object)
    swing_low_label = np.full(n, "", dtype=object)
    structure_event = np.full(n, "", dtype=object)
    broken_level = np.full(n, np.nan)

    previous_pivot_high = None
    previous_pivot_low = None
    last_active_high = None
    last_active_low = None
    directional_bias = "Neutral"
    high_already_broken = False
    low_already_broken = False

    for i in range(n):
        pivot_index = i - pivot_window
        if pivot_index >= pivot_window:
            left = pivot_index - pivot_window
            right = i + 1
            candidate_high = highs[pivot_index]
            candidate_low = lows[pivot_index]
            # Strict unique extrema avoid marking every bar in a flat plateau.
            if candidate_high == np.max(highs[left:right]) and np.count_nonzero(highs[left:right] == candidate_high) == 1:
                swing_high[i] = candidate_high
                if previous_pivot_high is not None:
                    swing_high_label[i] = "HH" if candidate_high > previous_pivot_high else "LH" if candidate_high < previous_pivot_high else "EQH"
                previous_pivot_high = candidate_high
                last_active_high = candidate_high
                high_already_broken = False
            if candidate_low == np.min(lows[left:right]) and np.count_nonzero(lows[left:right] == candidate_low) == 1:
                swing_low[i] = candidate_low
                if previous_pivot_low is not None:
                    swing_low_label[i] = "HL" if candidate_low > previous_pivot_low else "LL" if candidate_low < previous_pivot_low else "EQL"
                previous_pivot_low = candidate_low
                last_active_low = candidate_low
                low_already_broken = False

        # Only confirmed levels can trigger a break. Use close, not wick.
        broke_high = last_active_high is not None and not high_already_broken and closes[i] > last_active_high
        broke_low = last_active_low is not None and not low_already_broken and closes[i] < last_active_low
        if broke_high and broke_low:
            structure_event[i] = "Outside break — ambiguous"
            broken_level[i] = np.nan
        elif broke_high:
            structure_event[i] = "CHoCH bullish" if directional_bias == "Bearish" else "BOS bullish"
            broken_level[i] = last_active_high
            directional_bias = "Bullish"
            high_already_broken = True
        elif broke_low:
            structure_event[i] = "CHoCH bearish" if directional_bias == "Bullish" else "BOS bearish"
            broken_level[i] = last_active_low
            directional_bias = "Bearish"
            low_already_broken = True

    df["swing_high_confirmed"] = pd.Series(swing_high, index=df.index)
    df["swing_low_confirmed"] = pd.Series(swing_low, index=df.index)
    df["swing_high_label"] = pd.Series(swing_high_label, index=df.index)
    df["swing_low_label"] = pd.Series(swing_low_label, index=df.index)
    df["structure_event"] = pd.Series(structure_event, index=df.index)
    df["structure_break_level"] = pd.Series(broken_level, index=df.index)
    df["structure_bias"] = directional_bias
    # Per-row bias history is useful for charts and replay.
    bias_history = []
    current_bias = "Neutral"
    for event in structure_event:
        if event == "BOS bullish" or event == "CHoCH bullish":
            current_bias = "Bullish"
        elif event == "BOS bearish" or event == "CHoCH bearish":
            current_bias = "Bearish"
        bias_history.append(current_bias)
    df["structure_bias"] = bias_history
    return df
