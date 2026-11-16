"""Causal, explainable feature engineering for OHLCV candles.

Every row uses only information available at that candle's close. Baseline
volume statistics use prior candles only. Feature calculations are isolated
by instrument, timeframe and (when parseable) calendar session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _features_single(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    open_, high, low, close = (
        pd.to_numeric(df[c], errors="coerce") for c in ("open", "high", "low", "close")
    )
    volume = (
        pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        if "volume" in df.columns else pd.Series(0.0, index=df.index)
    )

    candle_range = (high - low).where((high - low) > 0)
    body_signed = close - open_
    body_abs = body_signed.abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

    df["feature_candle_range"] = candle_range
    df["feature_body_abs"] = body_abs
    df["feature_body_to_range"] = body_abs / candle_range
    df["feature_upper_wick_to_range"] = upper_wick / candle_range
    df["feature_lower_wick_to_range"] = lower_wick / candle_range
    df["feature_close_location"] = (close - low) / candle_range
    df["feature_candle_direction"] = np.sign(body_signed.fillna(0)).astype("int8")
    df["feature_gap_pct"] = open_ / close.shift(1).replace(0, np.nan) - 1.0

    for period in (1, 3, 6, 12):
        df[f"feature_return_{period}"] = close.pct_change(periods=period, fill_method=None)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    df["feature_atr_14"] = atr
    df["feature_range_atr"] = candle_range / atr.replace(0, np.nan)
    df["feature_body_atr"] = body_abs / atr.replace(0, np.nan)

    returns = close.pct_change(fill_method=None)
    df["feature_realized_volatility_20"] = returns.rolling(20, min_periods=10).std()

    # Compare current volume with completed candles before it, never itself.
    prior_volume = volume.shift(1)
    prior_volume_mean = prior_volume.rolling(20, min_periods=5).mean()
    prior_volume_std = prior_volume.rolling(20, min_periods=5).std()
    df["feature_relative_volume_20"] = volume / prior_volume_mean.replace(0, np.nan)
    df["feature_volume_zscore_20"] = (
        (volume - prior_volume_mean) / prior_volume_std.replace(0, np.nan)
    )

    # Confirmed swing levels are forward-filled only after their confirmation row.
    if "swing_high_confirmed" in df.columns:
        active_high = pd.to_numeric(df["swing_high_confirmed"], errors="coerce").ffill()
        df["feature_distance_to_confirmed_swing_high_pct"] = (
            close / active_high.replace(0, np.nan) - 1.0
        )
    if "swing_low_confirmed" in df.columns:
        active_low = pd.to_numeric(df["swing_low_confirmed"], errors="coerce").ffill()
        df["feature_distance_to_confirmed_swing_low_pct"] = (
            close / active_low.replace(0, np.nan) - 1.0
        )
    return df


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal candle, volatility, volume, trend and structure features.

    Rows are processed chronologically within each available symbol/timeframe
    and calendar-date partition, then restored to the caller's original order
    and index. Numeric timestamps are deliberately not interpreted because
    their epoch unit is ambiguous; callers should provide datetime timestamps.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns for feature engineering: {sorted(missing)}")
    if frame.empty:
        return _features_single(frame.copy())

    work = pd.DataFrame({"__position": np.arange(len(frame), dtype=np.int64)})
    group_columns: list[str] = []
    for column in ("symbol", "timeframe"):
        if column in frame.columns:
            work[column] = frame[column].to_numpy()
            group_columns.append(column)

    has_datetime_order = False
    if "timestamp" in frame.columns and not pd.api.types.is_numeric_dtype(frame["timestamp"]):
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if parsed.notna().any():
            has_datetime_order = True
            work["__timestamp_order"] = parsed.to_numpy()
            dates = parsed.dt.strftime("%Y-%m-%d")
            work["__session_date"] = [
                value if pd.notna(value) else f"__invalid_timestamp_{pos}"
                for pos, value in enumerate(dates)
            ]
            group_columns.append("__session_date")

    if group_columns:
        groups = [
            group["__position"].to_numpy()
            for _, group in work.groupby(group_columns, dropna=False, sort=False)
        ]
    else:
        groups = [work["__position"].to_numpy()]

    position_column = "__chart_reader_feature_position__"
    while position_column in frame.columns:
        position_column += "_"

    pieces = []
    for positions in groups:
        if has_datetime_order:
            ordered = (
                work.loc[work["__position"].isin(positions)]
                .sort_values("__timestamp_order", kind="mergesort")["__position"]
                .to_numpy()
            )
        else:
            ordered = positions
        calculated = _features_single(frame.iloc[ordered].copy())
        calculated[position_column] = ordered
        pieces.append(calculated)

    result = pd.concat(pieces, axis=0).sort_values(position_column, kind="mergesort")

    # Long-horizon trend context intentionally spans prior sessions for the
    # same instrument/timeframe. Session-sensitive returns, ATR and volume
    # baselines remain isolated above.
    series_group_columns = [
        column for column in ("symbol", "timeframe") if column in work.columns
    ]
    if series_group_columns:
        series_groups = [
            group["__position"].to_numpy()
            for _, group in work.groupby(
                series_group_columns, dropna=False, sort=False
            )
        ]
    else:
        series_groups = [work["__position"].to_numpy()]

    trend_values = {
        f"feature_{kind}_{period}_distance_pct": np.full(len(frame), np.nan)
        for kind in ("sma", "ema") for period in (20, 50)
    }
    for positions in series_groups:
        if has_datetime_order:
            ordered = (
                work.loc[work["__position"].isin(positions)]
                .sort_values("__timestamp_order", kind="mergesort")["__position"]
                .to_numpy()
            )
        else:
            ordered = positions
        close = pd.to_numeric(frame.iloc[ordered]["close"], errors="coerce")
        for period in (20, 50):
            sma = close.rolling(period, min_periods=period).mean()
            ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
            trend_values[f"feature_sma_{period}_distance_pct"][ordered] = (
                close.to_numpy() / sma.replace(0, np.nan).to_numpy() - 1.0
            )
            trend_values[f"feature_ema_{period}_distance_pct"][ordered] = (
                close.to_numpy() / ema.replace(0, np.nan).to_numpy() - 1.0
            )

    result = result.drop(columns=position_column)
    for column, values in trend_values.items():
        result[column] = values
    result.index = frame.index
    return result
