"""Phase 8.21: gap-safe forward excursion diagnostics for quality research."""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_INTERVAL_MINUTES = {"5m": 5, "15m": 15}


def add_forward_excursions(
    frame: pd.DataFrame, *, horizons: tuple[int, ...] = (3, 6, 12)
) -> pd.DataFrame:
    """Add long/short MFE and MAE percentages over contiguous future bars.

    Window starts with the bar after the source candle and ends at exactly H
    bars ahead. Source close is the reference. Missing bars/session crossings
    invalidate the window. These are path diagnostics, not simulated fills.
    """
    required = {"timestamp", "symbol", "timeframe", "close", "high", "low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required excursion columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot calculate excursions on an empty dataset.")
    if not horizons or any(
        isinstance(h, (bool, np.bool_)) or int(h) != h or int(h) <= 0
        for h in horizons
    ):
        raise ValueError("Horizons must be a non-empty sequence of positive integers.")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Every timestamp must be parseable.")
    if frame[["symbol", "timeframe"]].isna().any().any():
        raise ValueError("symbol and timeframe cannot be missing.")
    timeframe = frame["timeframe"].astype(str).str.strip().str.lower()
    unsupported = sorted(set(timeframe) - set(SUPPORTED_INTERVAL_MINUTES))
    if unsupported:
        raise ValueError("Unsupported timeframe(s): " + ", ".join(unsupported))
    prices = frame[["close", "high", "low"]].apply(pd.to_numeric, errors="coerce")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close, high and low must be finite positive prices.")
    if (prices["high"] < prices["low"]).any():
        raise ValueError("high cannot be below low.")
    if (prices["close"] > prices["high"]).any() or (prices["close"] < prices["low"]).any():
        raise ValueError("close must lie within each candle's high/low range.")
    work = pd.DataFrame(
        {
            "pos": np.arange(len(frame), dtype=np.int64),
            "symbol": frame["symbol"].astype(str).to_numpy(),
            "tf": timeframe.to_numpy(),
            "date": timestamps.dt.strftime("%Y-%m-%d").to_numpy(),
            "ts": timestamps.to_numpy(),
            "close": prices["close"].to_numpy(dtype=float),
            "high": prices["high"].to_numpy(dtype=float),
            "low": prices["low"].to_numpy(dtype=float),
            "minutes": timeframe.map(SUPPORTED_INTERVAL_MINUTES).to_numpy(),
        }
    )
    if work.duplicated(["symbol", "tf", "ts"]).any():
        raise ValueError("Duplicate symbol/timeframe/timestamp rows are not allowed.")
    result = frame.copy()
    for horizon in sorted(set(int(h) for h in horizons)):
        columns = {
            f"forward_mfe_long_{horizon}": np.full(len(frame), np.nan),
            f"forward_mae_long_{horizon}": np.full(len(frame), np.nan),
            f"forward_mfe_short_{horizon}": np.full(len(frame), np.nan),
            f"forward_mae_short_{horizon}": np.full(len(frame), np.nan),
        }
        for _, group in work.groupby(["symbol", "tf", "date"], sort=False):
            group = group.sort_values("ts", kind="mergesort")
            pos = group["pos"].to_numpy(dtype=int)
            ts = pd.DatetimeIndex(group["ts"])
            close = group["close"].to_numpy(dtype=float)
            high = group["high"].to_numpy(dtype=float)
            low = group["low"].to_numpy(dtype=float)
            delta = pd.Timedelta(minutes=int(group["minutes"].iloc[0]))
            for i in range(len(group) - horizon):
                future_ts = ts[i + 1 : i + horizon + 1]
                if len(future_ts) != horizon:
                    continue
                # Compare Timedelta values directly. DatetimeIndex.asi8 uses
                # the index's native resolution (s/us/ns), while Timedelta.value
                # is nanoseconds; mixing them breaks continuity checks on pandas 3.
                if future_ts[0] - ts[i] != delta:
                    continue
                if horizon > 1 and not (future_ts[1:] - future_ts[:-1] == delta).all():
                    continue
                entry = close[i]
                max_high = float(np.max(high[i + 1 : i + horizon + 1]))
                min_low = float(np.min(low[i + 1 : i + horizon + 1]))
                p = pos[i]
                columns[f"forward_mfe_long_{horizon}"][p] = (max_high / entry - 1) * 100
                columns[f"forward_mae_long_{horizon}"][p] = (min_low / entry - 1) * 100
                columns[f"forward_mfe_short_{horizon}"][p] = (entry / min_low - 1) * 100
                columns[f"forward_mae_short_{horizon}"][p] = (entry / max_high - 1) * 100
        for name, values in columns.items():
            result[name] = values
    return result
