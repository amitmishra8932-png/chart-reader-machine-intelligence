"""Fixed-horizon, cost-aware event simulation for research predictions.

This module does not train a model or create live signals. It converts a supplied
{-1, 0, +1} prediction stream into non-overlapping, session-bounded trades.
Entry is next-bar open; exit is the close after exactly horizon_bars full bars
from entry. Costs are an explicit all-in round-trip basis-point assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp", "symbol", "session_date", "open", "close", "prediction"
}


def simulate_fixed_horizon_trades(
    predictions: pd.DataFrame,
    *,
    horizon_bars: int,
    interval_minutes: int = 5,
    round_trip_cost_bps: float = 0.0,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Convert predictions to non-overlapping, session-bounded trades.

    Prediction at candle t is assumed available only after t closes. Entry is
    candle t+1 open; exit is candle t+horizon_bars close. Thus the holding
    interval contains exactly horizon_bars full candles, including entry.
    """
    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    columns = [
        "symbol", "session_date", "signal_time", "entry_time", "exit_time",
        "side", "entry_price", "exit_price", "gross_return", "cost_return",
        "net_return",
    ]
    audit_columns = [
        "symbol", "session_date", "signal_time", "prediction",
        "status", "reason", "entry_time", "exit_time",
    ]
    if predictions.empty:
        empty_trades = pd.DataFrame(columns=columns)
        empty_audit = pd.DataFrame(columns=audit_columns)
        return (empty_trades, empty_audit) if return_audit else empty_trades
    if (
        isinstance(horizon_bars, bool)
        or int(horizon_bars) != horizon_bars
        or horizon_bars <= 0
    ):
        raise ValueError("horizon_bars must be a positive integer.")
    if (
        isinstance(interval_minutes, bool)
        or int(interval_minutes) != interval_minutes
        or interval_minutes <= 0
    ):
        raise ValueError("interval_minutes must be a positive integer.")
    try:
        cost_bps = float(round_trip_cost_bps)
    except (TypeError, ValueError) as exc:
        raise ValueError("round_trip_cost_bps must be a finite non-negative number.") from exc
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("round_trip_cost_bps must be finite and non-negative.")

    data = predictions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("All timestamps must be valid.")
    for col in ("open", "close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if not np.isfinite(data[col]).all() or (data[col] <= 0).any():
            raise ValueError(f"{col} must contain finite positive prices.")
    data["prediction"] = pd.to_numeric(data["prediction"], errors="coerce")
    if data["prediction"].isna().any() or not data["prediction"].isin([-1, 0, 1]).all():
        raise ValueError("prediction must contain only -1, 0, +1.")
    if data[["symbol", "session_date"]].isna().any().any():
        raise ValueError("symbol and session_date cannot be missing.")
    timestamp_dates = data["timestamp"].dt.strftime("%Y-%m-%d")
    if not timestamp_dates.eq(data["session_date"].astype(str)).all():
        raise ValueError("session_date must match each timestamp's calendar date.")
    if data.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate symbol/timestamp rows are not allowed.")

    trades = []
    audit = []
    cost = cost_bps / 10000.0
    step = pd.Timedelta(minutes=int(interval_minutes))

    for (symbol, session_date), group in data.groupby(
        ["symbol", "session_date"], sort=False
    ):
        group = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        last_exit_index = -1
        for signal_index, row in group.iterrows():
            side = int(row["prediction"])
            audit_row = {
                "symbol": str(symbol),
                "session_date": str(session_date),
                "signal_time": row["timestamp"],
                "prediction": side,
                "status": "skipped",
                "reason": None,
                "entry_time": pd.NaT,
                "exit_time": pd.NaT,
            }
            if side == 0:
                audit_row["reason"] = "neutral_prediction"
                audit.append(audit_row)
                continue
            if signal_index <= last_exit_index:
                audit_row["reason"] = "overlap_or_exit_bar_cooldown"
                audit.append(audit_row)
                continue

            entry_index = signal_index + 1
            exit_index = signal_index + int(horizon_bars)
            if entry_index >= len(group) or exit_index >= len(group):
                audit_row["reason"] = "insufficient_horizon_bars"
                audit.append(audit_row)
                continue

            entry = group.iloc[entry_index]
            exit_row = group.iloc[exit_index]
            audit_row["entry_time"] = entry["timestamp"]
            audit_row["exit_time"] = exit_row["timestamp"]
            if entry["timestamp"] - row["timestamp"] != step:
                audit_row["reason"] = "entry_timestamp_gap"
                audit.append(audit_row)
                continue

            expected_times = pd.date_range(
                start=entry["timestamp"],
                periods=int(horizon_bars),
                freq=step,
            )
            observed_times = group.loc[
                entry_index:exit_index, "timestamp"
            ].reset_index(drop=True)
            if not observed_times.equals(pd.Series(expected_times)):
                audit_row["reason"] = "holding_window_gap"
                audit.append(audit_row)
                continue

            entry_price = float(entry["open"])
            exit_price = float(exit_row["close"])
            gross = (
                exit_price / entry_price - 1.0
                if side == 1
                else (entry_price - exit_price) / entry_price
            )
            trades.append({
                "symbol": str(symbol),
                "session_date": str(session_date),
                "signal_time": row["timestamp"],
                "entry_time": entry["timestamp"],
                "exit_time": exit_row["timestamp"],
                "side": "LONG" if side == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "cost_return": cost,
                "net_return": gross - cost,
            })
            last_exit_index = exit_index
            audit_row["status"] = "accepted"
            audit_row["reason"] = "executed"
            audit.append(audit_row)

    trade_frame = pd.DataFrame(trades, columns=columns)
    audit_frame = pd.DataFrame(audit, columns=audit_columns)
    return (trade_frame, audit_frame) if return_audit else trade_frame


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Return trade-level diagnostics without assuming portfolio sizing."""
    if trades.empty:
        return {
            "trade_count": 0,
            "gross_mean_bps": None,
            "gross_median_bps": None,
            "gross_win_rate_pct": None,
            "net_mean_bps": None,
            "net_median_bps": None,
            "net_win_rate_pct": None,
            "profit_factor": None,
            "total_net_bps": 0.0,
        }
    gross = trades["gross_return"].astype(float)
    net = trades["net_return"].astype(float)
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    return {
        "trade_count": int(len(trades)),
        "gross_mean_bps": float(gross.mean() * 10000),
        "gross_median_bps": float(gross.median() * 10000),
        "gross_win_rate_pct": float((gross > 0).mean() * 100),
        "net_mean_bps": float(net.mean() * 10000),
        "net_median_bps": float(net.median() * 10000),
        "net_win_rate_pct": float((net > 0).mean() * 100),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "total_net_bps": float(net.sum() * 10000),
    }
