"""Conservative, session-bounded historical replay for AI trade decisions.

Signals are calculated only from completed bars available at the decision time.
Entries use the following 5m bar open; intrabar stop/target ambiguity resolves
against the strategy (stop first). This is a research simulator, not execution.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.ai_decision import analyse_trade_setup


def _prepare(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{timeframe} replay data missing columns: {sorted(missing)}")
    data = frame.copy()
    parsed = pd.to_datetime(data["timestamp"], errors="coerce")
    if parsed.isna().any() or parsed.dt.tz is None:
        raise ValueError("Replay timestamps must be valid and timezone-aware.")
    data["timestamp"] = parsed.dt.tz_convert("UTC")
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if not np.isfinite(data[["open", "high", "low", "close", "volume"]].to_numpy(float)).all():
        raise ValueError("Replay OHLCV values must be finite.")
    if (data[["open", "high", "low", "close"]] <= 0).any().any() or (data["volume"] < 0).any():
        raise ValueError("Replay prices must be positive and volume non-negative.")
    invalid_geometry = (
        (data["high"] < data[["open", "close", "low"]].max(axis=1))
        | (data["low"] > data[["open", "close", "high"]].min(axis=1))
    )
    if invalid_geometry.any():
        raise ValueError("Replay contains invalid OHLC candle geometry.")
    if data["timestamp"].duplicated().any():
        raise ValueError("Replay contains duplicate timestamps.")
    data = data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    local = data["timestamp"].dt.tz_convert("Asia/Kolkata")
    if timeframe == "5m":
        valid = (local.dt.minute % 5 == 0) & (local.dt.second == 0)
    else:
        valid = (local.dt.minute % 15 == 0) & (local.dt.second == 0)
    if not valid.all():
        raise ValueError(f"Replay contains timestamps not aligned to {timeframe} boundaries.")
    data["__session"] = local.dt.strftime("%Y-%m-%d")
    return data


def _fill_price(reference: float, side: str, action: str, slippage_bps: float) -> float:
    adverse = slippage_bps / 10_000.0
    if action == "entry":
        return reference * (1 + adverse if side == "LONG" else 1 - adverse)
    return reference * (1 - adverse if side == "LONG" else 1 + adverse)


def replay_ai_decisions(
    candles_5m: pd.DataFrame,
    candles_15m: pd.DataFrame,
    *,
    symbol: str,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    min_score: int = 7,
    min_reward_risk: float = 1.8,
) -> dict[str, Any]:
    """Replay one symbol. Costs are round-trip bps of entry notional.

    Trades are closed by stop, target, or the final available bar of the same
    NSE session. If stop and target are both touched in one bar, stop wins.
    """
    for name, value in (("transaction_cost_bps", transaction_cost_bps),
                        ("slippage_bps", slippage_bps)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    five = _prepare(candles_5m, "5m")
    fifteen = _prepare(candles_15m, "15m")
    if five.empty or fifteen.empty:
        raise ValueError("Replay requires non-empty 5m and 15m candles.")
    if "symbol" in five and five["symbol"].astype(str).str.upper().nunique() != 1:
        raise ValueError("Replay accepts exactly one 5m symbol per call.")
    if "symbol" in fifteen and fifteen["symbol"].astype(str).str.upper().nunique() != 1:
        raise ValueError("Replay accepts exactly one 15m symbol per call.")

    trades: list[dict[str, Any]] = []
    skipped = {"no_setup": 0, "no_next_bar": 0, "invalid_next_open": 0}
    i = 20
    while i < len(five) - 1:
        signal_bar = five.iloc[i]
        session = signal_bar["__session"]
        if five.iloc[i + 1]["__session"] != session:
            i += 1
            continue
        five_history = five.iloc[:i + 1].drop(columns=["__session"])
        five_end = signal_bar["timestamp"] + pd.Timedelta(minutes=5)
        eligible_15 = fifteen[
            (fifteen["__session"] == session)
            & (fifteen["timestamp"] + pd.Timedelta(minutes=15) <= five_end)
        ]
        if len(eligible_15) < 21:
            i += 1
            continue
        fifteen_history = eligible_15.drop(columns=["__session"])
        try:
            decision = analyse_trade_setup(
                five_history, fifteen_history, symbol=symbol,
                min_score=min_score, min_reward_risk=min_reward_risk,
            )
        except (ValueError, KeyError, TypeError):
            skipped["no_setup"] += 1
            i += 1
            continue
        if decision.get("decision") not in {"LONG", "SHORT"}:
            skipped["no_setup"] += 1
            i += 1
            continue

        entry_bar = five.iloc[i + 1]
        raw_entry = float(entry_bar["open"])
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            skipped["invalid_next_open"] += 1
            i += 1
            continue
        side = decision["decision"]
        stop = float(decision["stop_loss"])
        target = float(decision["target"])
        entry = _fill_price(raw_entry, side, "entry", slippage_bps)
        # Do not simulate a fill if the next open has already invalidated the setup.
        if (side == "LONG" and (entry <= stop or entry >= target)) or (
            side == "SHORT" and (entry >= stop or entry <= target)
        ):
            skipped["invalid_next_open"] += 1
            i += 1
            continue

        exit_raw = None
        exit_reason = None
        exit_index = i + 1
        j = i + 1
        while j < len(five) and five.iloc[j]["__session"] == session:
            bar = five.iloc[j]
            high, low = float(bar["high"]), float(bar["low"])
            if side == "LONG":
                stop_hit, target_hit = low <= stop, high >= target
            else:
                stop_hit, target_hit = high >= stop, low <= target
            if stop_hit:
                # Stop first when both barriers trade; honour worse opening gaps.
                opening = float(bar["open"])
                if side == "LONG" and opening < stop:
                    exit_raw = opening
                elif side == "SHORT" and opening > stop:
                    exit_raw = opening
                else:
                    exit_raw = stop
                exit_reason, exit_index = "STOP", j
                break
            if target_hit:
                exit_raw, exit_reason, exit_index = target, "TARGET", j
                break
            j += 1
        if exit_raw is None:
            exit_index = min(j - 1, len(five) - 1)
            exit_raw = float(five.iloc[exit_index]["close"])
            exit_reason = "SESSION_END"
        exit_price = _fill_price(exit_raw, side, "exit", slippage_bps)
        gross_pct = ((exit_price / entry) - 1) * 100
        if side == "SHORT":
            gross_pct = -gross_pct
        cost_pct = transaction_cost_bps / 100.0
        net_pct = gross_pct - cost_pct
        trades.append({
            "symbol": str(symbol).upper(), "side": side,
            "signal_time": signal_bar["timestamp"].isoformat(),
            "entry_time": entry_bar["timestamp"].isoformat(),
            "exit_time": five.iloc[exit_index]["timestamp"].isoformat(),
            "signal_close": float(signal_bar["close"]),
            "entry_price": entry, "stop_loss": stop, "target": target,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "score": int(decision["score"]), "gross_return_pct": gross_pct,
            "cost_pct": cost_pct, "net_return_pct": net_pct,
            "bars_held": int(exit_index - (i + 1) + 1),
        })
        # No overlapping positions for the same symbol.
        i = exit_index + 1

    frame = pd.DataFrame(trades)
    if frame.empty:
        return {
            "symbol": str(symbol).upper(), "trades": [], "trade_count": 0,
            "win_rate_pct": None, "mean_net_return_pct": None,
            "profit_factor": None, "max_drawdown_pct": None,
            "skipped": skipped,
            "limitations": ["No trades qualified in this sample; this is not proof of a strategy failure."],
        }
    net = frame["net_return_pct"].astype(float)
    wins = net[net > 0].sum()
    losses = abs(net[net < 0].sum())
    equity = (1 + net / 100).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    return {
        "symbol": str(symbol).upper(), "trades": trades,
        "trade_count": int(len(frame)),
        "win_rate_pct": float((net > 0).mean() * 100),
        "mean_net_return_pct": float(net.mean()),
        "median_net_return_pct": float(net.median()),
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else None),
        "max_drawdown_pct": float(drawdown.min()),
        "stop_exits": int((frame["exit_reason"] == "STOP").sum()),
        "target_exits": int((frame["exit_reason"] == "TARGET").sum()),
        "session_end_exits": int((frame["exit_reason"] == "SESSION_END").sum()),
        "skipped": skipped,
        "limitations": [
            "Uses next-bar open fills with fixed adverse slippage and round-trip transaction-cost assumptions.",
            "If stop and target both touch in one candle, stop is assumed first.",
            "No market impact, queue position, taxes by turnover, circuit limits, or partial fills are modelled.",
            "Single-symbol, one-position-at-a-time replay; results need walk-forward out-of-sample validation.",
        ],
    }
