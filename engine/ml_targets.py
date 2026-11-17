"""Explicit, gap-safe directional targets for historical ML research."""
from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_INTERVAL_MINUTES = {"5m": 5, "15m": 15}


def add_directional_targets(
    frame: pd.DataFrame,
    *,
    neutral_bands_pct: dict[int, float],
) -> pd.DataFrame:
    """Add forward returns and {-1, 0, +1} targets without crossing gaps/sessions.

    A target is +1 when the close-to-close forward return exceeds the supplied
    positive neutral band, -1 when it is below the negative band, and 0
    otherwise. The band is expressed in percentage points (e.g. 0.20 means
    0.20%). Bands are mandatory.

    A horizon is labelled only when its endpoint is exactly horizon * the
    timeframe interval after the source candle. Since duplicate timestamps are
    rejected and rows are sorted, this prevents labels from jumping over
    missing candles. Supported intervals are 5m and 15m.
    """
    required = {"timestamp", "symbol", "timeframe", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required target columns: {sorted(missing)}")
    if not neutral_bands_pct:
        raise ValueError("Provide an explicit neutral band for each horizon.")

    bands: dict[int, float] = {}
    for raw_horizon, raw_band in neutral_bands_pct.items():
        if isinstance(raw_horizon, (bool, np.bool_)):
            raise ValueError("Horizons must be positive integers.")
        try:
            horizon = int(raw_horizon)
            band = float(raw_band)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid horizon or neutral band.") from exc
        if horizon <= 0 or horizon != raw_horizon:
            raise ValueError("Horizons must be positive integers.")
        if not np.isfinite(band) or band <= 0:
            raise ValueError("Neutral bands must be finite positive percentages.")
        bands[horizon] = band

    if frame.empty:
        raise ValueError("Cannot create targets from an empty dataset.")

    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("Every timestamp must be parseable before target creation.")
    if frame[["symbol", "timeframe"]].isna().any().any():
        raise ValueError("symbol and timeframe cannot be missing.")
    timeframe = frame["timeframe"].astype(str).str.strip().str.lower()
    unsupported = sorted(set(timeframe) - set(SUPPORTED_INTERVAL_MINUTES))
    if unsupported:
        raise ValueError(
            "Unsupported target timeframe(s); expected 5m or 15m: "
            + ", ".join(unsupported)
        )
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("close must contain finite positive prices.")

    work = pd.DataFrame({
        "__position": np.arange(len(frame), dtype=np.int64),
        "__symbol": frame["symbol"].astype(str).to_numpy(),
        "__timeframe": timeframe.to_numpy(),
        "__interval_minutes": timeframe.map(SUPPORTED_INTERVAL_MINUTES).to_numpy(),
        "__session_date": parsed.dt.strftime("%Y-%m-%d").to_numpy(),
        "__timestamp": parsed.to_numpy(),
        "__close": close,
    })
    if work.duplicated(["__symbol", "__timeframe", "__timestamp"]).any():
        raise ValueError("Duplicate symbol/timeframe/timestamp rows must be resolved first.")

    ordered = work.sort_values(
        ["__symbol", "__timeframe", "__session_date", "__timestamp"],
        kind="mergesort",
    )
    group_keys = ["__symbol", "__timeframe", "__session_date"]
    result = frame.copy()

    for horizon, band in sorted(bands.items()):
        grouped = ordered.groupby(group_keys, sort=False)
        future = grouped["__close"].shift(-horizon)
        future_timestamp = grouped["__timestamp"].shift(-horizon)
        expected_delta = pd.to_timedelta(
            ordered["__interval_minutes"].to_numpy(dtype=float) * horizon,
            unit="m",
        )
        continuous = (future_timestamp - ordered["__timestamp"]).eq(expected_delta)
        future = future.where(continuous)

        future_by_position = np.full(len(frame), np.nan, dtype=float)
        future_by_position[ordered["__position"].to_numpy()] = future.to_numpy(dtype=float)
        forward_return = (future_by_position / close - 1.0) * 100.0

        target = np.full(len(frame), np.nan, dtype=float)
        valid = np.isfinite(forward_return)
        target[valid & (forward_return > band)] = 1
        target[valid & (forward_return < -band)] = -1
        target[valid & (forward_return >= -band) & (forward_return <= band)] = 0

        result[f"forward_return_{horizon}"] = forward_return
        result[f"target_{horizon}"] = pd.array(target, dtype="Int8")

    return result
