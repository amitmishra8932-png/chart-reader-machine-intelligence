"""Strict as-of alignment of completed higher-timeframe candles."""
from __future__ import annotations

from typing import Literal

import pandas as pd


TimestampLabel = Literal["start", "end"]


def align_closed_higher_timeframe(
    base: pd.DataFrame,
    higher: pd.DataFrame,
    *,
    base_interval: str = "5min",
    higher_interval: str = "15min",
    base_timestamp_label: TimestampLabel,
    higher_timestamp_label: TimestampLabel,
    tolerance: str = "15min",
    higher_prefix: str = "context_15m_",
) -> pd.DataFrame:
    """Attach only higher-timeframe candles already closed by each base close.

    The timestamp-label convention is mandatory because provider documentation
    does not always say whether candle timestamps denote interval start or end.
    Matching is restricted to the same symbol and calendar date. No session
    carry-forward is permitted across dates.
    """
    if base_timestamp_label not in ("start", "end"):
        raise ValueError("base_timestamp_label must be 'start' or 'end'")
    if higher_timestamp_label not in ("start", "end"):
        raise ValueError("higher_timestamp_label must be 'start' or 'end'")
    if not {"timestamp", "symbol"}.issubset(base.columns):
        raise ValueError("base data requires timestamp and symbol columns")
    if not {"timestamp", "symbol"}.issubset(higher.columns):
        raise ValueError("higher-timeframe data requires timestamp and symbol columns")
    if base.empty:
        return base.copy()

    base_data = base.copy()
    higher_data = higher.copy()
    base_data["__position"] = range(len(base_data))
    base_data["__parsed_timestamp"] = pd.to_datetime(
        base_data["timestamp"], errors="coerce"
    )
    higher_data["__parsed_timestamp"] = pd.to_datetime(
        higher_data["timestamp"], errors="coerce"
    )
    if base_data["__parsed_timestamp"].isna().any() or higher_data["__parsed_timestamp"].isna().any():
        raise ValueError("Invalid timestamps must be resolved before timeframe alignment")

    base_data["__session_date"] = base_data["__parsed_timestamp"].dt.strftime("%Y-%m-%d")
    higher_data["__session_date"] = higher_data["__parsed_timestamp"].dt.strftime("%Y-%m-%d")

    base_delta = pd.Timedelta(base_interval)
    higher_delta = pd.Timedelta(higher_interval)
    allowed_age = pd.Timedelta(tolerance)
    base_data["__decision_time"] = base_data["__parsed_timestamp"]
    higher_data["__available_at"] = higher_data["__parsed_timestamp"]
    if base_timestamp_label == "start":
        base_data["__decision_time"] += base_delta
    if higher_timestamp_label == "start":
        higher_data["__available_at"] += higher_delta

    if higher_data.duplicated(
        ["symbol", "__session_date", "__available_at"]
    ).any():
        raise ValueError("Duplicate higher-timeframe candle availability times found")

    context_columns = [
        column for column in higher.columns if column not in ("symbol", "timestamp")
    ]
    rename = {
        column: f"{higher_prefix}{column}" for column in context_columns
    }
    higher_data = higher_data.rename(columns=rename)
    higher_data = higher_data.rename(
        columns={"timestamp": f"{higher_prefix}timestamp"}
    )
    output_parts = []

    for (symbol, session_date), base_group in base_data.groupby(
        ["symbol", "__session_date"], dropna=False, sort=False
    ):
        context_group = higher_data.loc[
            (higher_data["symbol"] == symbol)
            & (higher_data["__session_date"] == session_date)
        ].copy()
        left = base_group.sort_values("__decision_time", kind="mergesort")
        if context_group.empty:
            merged = left.copy()
            for column in [f"{higher_prefix}timestamp", *rename.values()]:
                merged[column] = pd.NA
            merged[f"{higher_prefix}available_at"] = pd.NaT
        else:
            right_columns = [
                "__available_at", f"{higher_prefix}timestamp", *rename.values()
            ]
            right = context_group[right_columns].sort_values(
                "__available_at", kind="mergesort"
            )
            merged = pd.merge_asof(
                left,
                right,
                left_on="__decision_time",
                right_on="__available_at",
                direction="backward",
                tolerance=allowed_age,
                allow_exact_matches=True,
            )
        merged[f"{higher_prefix}age_minutes"] = (
            merged["__decision_time"] - merged["__available_at"]
        ).dt.total_seconds() / 60.0
        output_parts.append(merged)

    result = (
        pd.concat(output_parts, axis=0)
        .sort_values("__position", kind="mergesort")
    )
    internal = [
        "__position", "__parsed_timestamp", "__session_date", "__decision_time",
        "__available_at",
    ]
    result = result.drop(columns=[column for column in internal if column in result])
    result.index = base.index
    return result
