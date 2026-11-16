"""Forward-return labelling for research; labels are not trade recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_forward_returns(frame: pd.DataFrame, horizons=(3, 6, 12)) -> pd.DataFrame:
    """Add forward returns without crossing instrument, timeframe, or session boundaries.

    When metadata is available, observations are grouped by symbol and timeframe,
    then split by the calendar date of timestamp. Rows are ordered chronologically
    inside each group for labelling, while the returned frame retains its original
    row order and index.

    For simple single-series inputs without metadata, the function retains the
    historical row-sequence behaviour.
    """
    if "close" not in frame.columns:
        raise ValueError("Missing required close column.")

    df = frame.copy()
    work = pd.DataFrame({
        "__row_position": np.arange(len(df), dtype=np.int64),
        "__close": pd.to_numeric(df["close"], errors="coerce").to_numpy(),
    })

    group_columns = [column for column in ("symbol", "timeframe") if column in df.columns]
    for column in group_columns:
        work[column] = df[column].to_numpy()
    sort_columns = list(group_columns)

    if "timestamp" in df.columns and not pd.api.types.is_numeric_dtype(df["timestamp"]):
        parsed_timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed_timestamp.notna().any():
            work["__session_date"] = parsed_timestamp.dt.strftime("%Y-%m-%d").to_numpy()
            work["__timestamp_order"] = parsed_timestamp.to_numpy()
            group_columns.append("__session_date")
            sort_columns.extend(["__session_date", "__timestamp_order"])

    ordered = work.sort_values(sort_columns if sort_columns else ["__row_position"], kind="mergesort")

    for horizon in horizons:
        try:
            steps = int(horizon)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Horizons must be positive integers.") from exc
        if isinstance(horizon, (float, np.floating)) and not float(horizon).is_integer():
            raise ValueError("Horizons must be positive integers.")
        if isinstance(horizon, (bool, np.bool_)) or steps <= 0:
            raise ValueError("Horizons must be positive integers.")

        if group_columns:
            future = ordered.groupby(group_columns, dropna=False, sort=False)["__close"].shift(-steps)
        else:
            future = ordered["__close"].shift(-steps)

        future_by_position = np.full(len(df), np.nan, dtype=float)
        future_by_position[ordered["__row_position"].to_numpy()] = future.to_numpy(dtype=float)
        current_close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        df[f"forward_return_{steps}"] = (future_by_position / current_close - 1.0) * 100.0

    return df


def outcome_summary(frame: pd.DataFrame, pattern_column: str, horizon: int) -> dict:
    col = f"forward_return_{int(horizon)}"
    if pattern_column not in frame or col not in frame:
        raise ValueError("Pattern or forward-return column is missing.")
    values = frame.loc[frame[pattern_column].fillna(False).astype(bool), col].dropna()
    if values.empty:
        return {"observations": 0, "mean_return_pct": None, "positive_rate_pct": None}
    return {
        "observations": int(values.size),
        "mean_return_pct": float(values.mean()),
        "positive_rate_pct": float((values > 0).mean() * 100),
        "median_return_pct": float(values.median()),
    }
