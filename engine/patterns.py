"""Explainable candlestick geometry detectors.

Detectors identify candle shapes/sequences only. They do not issue trading
signals; trend, location, confirmation and empirical outcomes are separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _detect_patterns_single(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    body_ratio = body / candle_range
    upper_ratio = upper / candle_range
    lower_ratio = lower / candle_range
    close_location = (c - l) / candle_range

    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_h, prev_l = h.shift(1), l.shift(1)
    prev_body = (prev_c - prev_o).abs()
    prev_range = (prev_h - prev_l).replace(0, np.nan)
    midpoint_prev_body = (prev_o + prev_c) / 2

    # Single-candle shapes.
    df["pattern_doji"] = body_ratio <= 0.10
    df["pattern_long_legged_doji"] = (
        df["pattern_doji"] & (upper_ratio >= 0.30) & (lower_ratio >= 0.30)
    )
    df["pattern_dragonfly_doji"] = (
        df["pattern_doji"] & (upper_ratio <= 0.10) & (lower_ratio >= 0.60)
        & (close_location >= 0.80)
    )
    df["pattern_gravestone_doji"] = (
        df["pattern_doji"] & (lower_ratio <= 0.10) & (upper_ratio >= 0.60)
        & (close_location <= 0.20)
    )
    df["pattern_hammer"] = (
        (lower >= body * 2) & (upper <= body * 0.6)
        & (close_location >= 0.60) & (body_ratio <= 0.40)
    )
    df["pattern_shooting_star"] = (
        (upper >= body * 2) & (lower <= body * 0.6)
        & (close_location <= 0.40) & (body_ratio <= 0.40)
    )

    # Trend-conditioned names for shapes that share geometry.
    prior_fall = c.shift(1) < c.shift(4)
    prior_rise = c.shift(1) > c.shift(4)
    df["pattern_hanging_man"] = df["pattern_hammer"] & prior_rise
    df["pattern_inverted_hammer"] = df["pattern_shooting_star"] & prior_fall
    df["pattern_hammer"] = df["pattern_hammer"] & prior_fall
    df["pattern_shooting_star"] = df["pattern_shooting_star"] & prior_rise

    # Two-candle body relationships.
    df["pattern_bullish_engulfing"] = (
        (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    )
    df["pattern_bearish_engulfing"] = (
        (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
    )
    df["pattern_inside_bar"] = (h < prev_h) & (l > prev_l)
    df["pattern_outside_bar"] = (h > prev_h) & (l < prev_l)
    df["pattern_piercing_line"] = (
        (prev_c < prev_o) & (c > o) & (o < prev_c)
        & (c > midpoint_prev_body) & (c < prev_o)
    )
    df["pattern_dark_cloud_cover"] = (
        (prev_c > prev_o) & (c < o) & (o > prev_c)
        & (c < midpoint_prev_body) & (c > prev_o)
    )

    # Three-candle stars: gap is not required, making these intraday variants.
    first_o, first_c = o.shift(2), c.shift(2)
    first_body = (first_c - first_o).abs()
    first_mid = (first_o + first_c) / 2
    middle_body = body.shift(1)
    first_bear = first_c < first_o
    first_bull = first_c > first_o
    middle_small = middle_body < first_body * 0.50
    df["pattern_morning_star"] = (
        first_bear & middle_small & (c > o) & (c > first_mid)
        & (body >= first_body * 0.50)
    )
    df["pattern_evening_star"] = (
        first_bull & middle_small & (c < o) & (c < first_mid)
        & (body >= first_body * 0.50)
    )

    # Repeated highs/lows: tolerance scales with recent price, not exact equality.
    tolerance = c.abs() * 0.001
    df["pattern_tweezer_bottom"] = (
        (l - prev_l).abs() <= tolerance
    ) & (prev_c < prev_o) & (c > o)
    df["pattern_tweezer_top"] = (
        (h - prev_h).abs() <= tolerance
    ) & (prev_c > prev_o) & (c < o)

    # Large directional real body with small shadows.
    df["pattern_bullish_marubozu"] = (c > o) & (body_ratio >= 0.90)
    df["pattern_bearish_marubozu"] = (c < o) & (body_ratio >= 0.90)
    df["pattern_spinning_top"] = (
        (body_ratio <= 0.30) & (upper_ratio >= 0.20) & (lower_ratio >= 0.20)
    )

    # Three-candle momentum sequences; intentionally conservative geometry.
    bull1 = (c.shift(2) > o.shift(2))
    bull2 = (c.shift(1) > o.shift(1))
    bull3 = c > o
    bear1 = (c.shift(2) < o.shift(2))
    bear2 = (c.shift(1) < o.shift(1))
    bear3 = c < o
    upper_shadow_ratio = upper / candle_range
    lower_shadow_ratio = lower / candle_range
    df["pattern_three_white_soldiers"] = (
        bull1 & bull2 & bull3
        & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        & (body_ratio >= 0.50) & (body_ratio.shift(1) >= 0.50)
        & (body_ratio.shift(2) >= 0.50)
        & (upper_shadow_ratio <= 0.25)
        & (upper_shadow_ratio.shift(1) <= 0.25)
        & (upper_shadow_ratio.shift(2) <= 0.25)
    )
    df["pattern_three_black_crows"] = (
        bear1 & bear2 & bear3
        & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        & (body_ratio >= 0.50) & (body_ratio.shift(1) >= 0.50)
        & (body_ratio.shift(2) >= 0.50)
        & (lower_shadow_ratio <= 0.25)
        & (lower_shadow_ratio.shift(1) <= 0.25)
        & (lower_shadow_ratio.shift(2) <= 0.25)
    )

    names = {
        "pattern_doji": "Doji",
        "pattern_long_legged_doji": "Long-legged doji",
        "pattern_dragonfly_doji": "Dragonfly doji",
        "pattern_gravestone_doji": "Gravestone doji",
        "pattern_hammer": "Hammer (after decline)",
        "pattern_hanging_man": "Hanging man (after advance)",
        "pattern_inverted_hammer": "Inverted hammer (after decline)",
        "pattern_shooting_star": "Shooting star (after advance)",
        "pattern_bullish_engulfing": "Bullish engulfing",
        "pattern_bearish_engulfing": "Bearish engulfing",
        "pattern_inside_bar": "Inside bar",
        "pattern_outside_bar": "Outside bar",
        "pattern_piercing_line": "Piercing line",
        "pattern_dark_cloud_cover": "Dark cloud cover",
        "pattern_morning_star": "Morning star",
        "pattern_evening_star": "Evening star",
        "pattern_tweezer_bottom": "Tweezer bottom",
        "pattern_tweezer_top": "Tweezer top",
        "pattern_bullish_marubozu": "Bullish marubozu",
        "pattern_bearish_marubozu": "Bearish marubozu",
        "pattern_spinning_top": "Spinning top",
        "pattern_three_white_soldiers": "Three white soldiers",
        "pattern_three_black_crows": "Three black crows",
    }
    for column in names:
        df[column] = df[column].fillna(False).astype(bool)
    df["detected_patterns"] = df.apply(
        lambda row: ", ".join(name for col, name in names.items() if row[col]),
        axis=1,
    )
    return df



def detect_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    """Detect patterns independently per instrument, timeframe, and session.

    Multi-series inputs are partitioned before any rolling/shift-based detector
    logic runs. If timestamps are available, each partition is processed in
    chronological order within its calendar date. The returned rows retain the
    caller's original order and index. Inputs without metadata keep the original
    single-series behaviour.
    """
    if frame.empty:
        return _detect_patterns_single(frame.copy())

    work = pd.DataFrame({"__position": np.arange(len(frame), dtype=np.int64)})
    group_columns = []
    for column in ("symbol", "timeframe"):
        if column in frame.columns:
            work[column] = frame[column].to_numpy()
            group_columns.append(column)

    timestamp_order_available = False
    if "timestamp" in frame.columns and not pd.api.types.is_numeric_dtype(frame["timestamp"]):
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if parsed.notna().any():
            timestamp_order_available = True
            work["__timestamp_order"] = parsed.to_numpy()
            session_dates = parsed.dt.strftime("%Y-%m-%d")
            # Invalid timestamps are isolated row-by-row: their session is unknown.
            work["__session_date"] = [
                value if pd.notna(value) else f"__invalid_timestamp_{position}"
                for position, value in enumerate(session_dates)
            ]
            group_columns.append("__session_date")

    if group_columns:
        groups = (
            group["__position"].to_numpy()
            for _, group in work.groupby(group_columns, dropna=False, sort=False)
        )
    else:
        groups = [work["__position"].to_numpy()]

    pieces = []
    for positions in groups:
        if timestamp_order_available and "__timestamp_order" in work.columns:
            ordered_positions = (
                work.loc[work["__position"].isin(positions)]
                .sort_values("__timestamp_order", kind="mergesort")["__position"]
                .to_numpy()
            )
        else:
            ordered_positions = positions

        detected = _detect_patterns_single(frame.iloc[ordered_positions].copy())
        detected["__position"] = ordered_positions
        pieces.append(detected)

    combined = pd.concat(pieces, axis=0)
    combined = combined.sort_values("__position", kind="mergesort")
    combined = combined.drop(columns="__position")
    combined.index = frame.index
    return combined
