"""Forward-return labelling for research; labels are not trade recommendations."""
from __future__ import annotations
import pandas as pd


def label_forward_returns(frame: pd.DataFrame, horizons=(3, 6, 12)) -> pd.DataFrame:
    df = frame.copy()
    for horizon in horizons:
        if int(horizon) <= 0:
            raise ValueError("Horizons must be positive integers.")
        future = df["close"].shift(-int(horizon))
        df[f"forward_return_{horizon}"] = (future / df["close"] - 1.0) * 100.0
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
