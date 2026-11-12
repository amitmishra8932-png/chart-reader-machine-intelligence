"""Manual paper-trading journal and risk controls.

This module never connects to a broker or submits orders. Prices are entered
by the user; all results are simulated and friction is an estimate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Iterable

import pandas as pd

TRADE_COLUMNS = [
    "trade_id", "opened_at", "closed_at", "symbol", "timeframe", "side",
    "entry_price", "stop_price", "target_price", "quantity", "exit_price",
    "status", "estimated_cost_bps", "notes",
]


def suggest_quantity(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    max_exposure_pct: float = 100.0,
) -> int:
    """Size a cash-only simulated position using stop risk and notional cap."""
    values = (capital, risk_pct, entry_price, stop_price, max_exposure_pct)
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Sizing inputs must be finite numbers.")
    if capital <= 0 or entry_price <= 0 or stop_price <= 0:
        raise ValueError("Capital, entry and stop must be positive.")
    if not 0 < risk_pct <= 100 or not 0 < max_exposure_pct <= 100:
        raise ValueError("Risk and exposure percentages must be between 0 and 100.")
    distance = abs(entry_price - stop_price)
    if distance <= 0:
        raise ValueError("Entry and stop prices must be different.")
    risk_budget = capital * risk_pct / 100.0
    exposure_budget = capital * max_exposure_pct / 100.0
    by_risk = math.floor(risk_budget / distance)
    by_exposure = math.floor(exposure_budget / entry_price)
    quantity = min(by_risk, by_exposure)
    if quantity < 1:
        raise ValueError("Risk budget is too small for one share at this stop distance.")
    return int(quantity)


def open_paper_trade(
    trades: Iterable[dict],
    *,
    symbol: str,
    timeframe: str,
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    quantity: int,
    capital: float,
    max_open_trades: int = 3,
    max_total_open_risk_pct: float = 3.0,
    estimated_cost_bps: float = 10.0,
    notes: str = "",
    opened_at: str | None = None,
    trade_id: str | None = None,
) -> dict:
    """Validate and create one manual paper trade; does not mutate input trades."""
    side = str(side).upper()
    symbol = str(symbol).strip().upper()
    timeframe = str(timeframe).strip().lower()
    try:
        entry, stop, target = map(float, (entry_price, stop_price, target_price))
        qty, capital = int(quantity), float(capital)
        cost_bps = float(estimated_cost_bps)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prices, quantity and capital must be valid numbers.") from exc
    if not symbol:
        raise ValueError("Select a stock symbol.")
    if timeframe not in {"5m", "15m"}:
        raise ValueError("Timeframe must be 5m or 15m.")
    if side not in {"LONG", "SHORT"}:
        raise ValueError("Side must be LONG or SHORT.")
    if any(not math.isfinite(v) for v in (entry, stop, target, capital, cost_bps)):
        raise ValueError("Trade values must be finite numbers.")
    if min(entry, stop, target, capital) <= 0 or qty <= 0:
        raise ValueError("Prices, capital and quantity must be positive.")
    if cost_bps < 0:
        raise ValueError("Estimated costs cannot be negative.")
    if side == "LONG" and not stop < entry < target:
        raise ValueError("For a LONG, stop must be below entry and target above entry.")
    if side == "SHORT" and not target < entry < stop:
        raise ValueError("For a SHORT, target must be below entry and stop above entry.")
    risk_amount = abs(entry - stop) * qty
    if risk_amount > capital * max_total_open_risk_pct / 100.0 + 1e-9:
        raise ValueError("This trade would exceed the total open-risk limit.")
    current = list(trades)
    open_trades = [t for t in current if t.get("status") == "OPEN"]
    if len(open_trades) >= max_open_trades:
        raise ValueError(f"Maximum of {max_open_trades} simultaneous open paper trades reached.")
    existing_risk = sum(
        abs(float(t["entry_price"]) - float(t["stop_price"])) * int(t["quantity"])
        for t in open_trades
    )
    if existing_risk + risk_amount > capital * max_total_open_risk_pct / 100.0 + 1e-9:
        raise ValueError("This trade would exceed the portfolio's total open-risk limit.")
    if any(t.get("status") == "OPEN" and t.get("symbol") == symbol for t in open_trades):
        raise ValueError(f"{symbol} already has an open paper trade.")
    now = opened_at or datetime.now(timezone.utc).isoformat()
    return {
        "trade_id": trade_id or f"PT-{len(current) + 1:05d}",
        "opened_at": now,
        "closed_at": "",
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "quantity": qty,
        "exit_price": None,
        "status": "OPEN",
        "estimated_cost_bps": cost_bps,
        "notes": str(notes).strip(),
    }


def close_paper_trade(trade: dict, exit_price: float, closed_at: str | None = None) -> dict:
    """Return a closed copy of an open paper trade at the user-entered exit price."""
    try:
        price = float(exit_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("Exit price must be a valid number.") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Exit price must be positive and finite.")
    if trade.get("status") != "OPEN":
        raise ValueError("Only OPEN paper trades can be closed.")
    result = dict(trade)
    result["exit_price"] = price
    result["closed_at"] = closed_at or datetime.now(timezone.utc).isoformat()
    result["status"] = "CLOSED"
    return result


def trade_result(trade: dict) -> dict:
    """Calculate realised P&L and R multiple for a closed trade."""
    if trade.get("status") != "CLOSED" or trade.get("exit_price") is None:
        raise ValueError("Trade must be closed before calculating realised P&L.")
    entry = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    quantity = int(trade["quantity"])
    direction = 1.0 if trade["side"] == "LONG" else -1.0
    gross = (exit_price - entry) * quantity * direction
    costs = (entry + exit_price) * quantity * float(trade.get("estimated_cost_bps", 0)) / 10000.0
    net = gross - costs
    initial_risk = abs(entry - float(trade["stop_price"])) * quantity
    return {
        "gross_pnl": gross,
        "estimated_costs": costs,
        "net_pnl": net,
        "r_multiple": net / initial_risk if initial_risk else None,
    }


def paper_trading_summary(trades: Iterable[dict], starting_capital: float) -> dict:
    """Summarise closed paper trades; open positions are excluded from realised P&L."""
    trades = list(trades)
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    results = [trade_result(t) for t in closed]
    pnls = [r["net_pnl"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    equity = float(starting_capital)
    peak = equity
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)
    return {
        "closed_trades": len(closed),
        "open_trades": sum(t.get("status") == "OPEN" for t in trades),
        "net_pnl": sum(pnls),
        "win_rate_pct": len(wins) / len(closed) * 100.0 if closed else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else None),
        "max_drawdown_pct": max_drawdown,
        "ending_realised_equity": float(starting_capital) + sum(pnls),
        "average_r": sum(r["r_multiple"] for r in results if r["r_multiple"] is not None) / len(results) if results else None,
    }


def trades_frame(trades: Iterable[dict]) -> pd.DataFrame:
    """Create a stable CSV-ready trade ledger with realised metrics."""
    rows = []
    for trade in trades:
        row = {column: trade.get(column) for column in TRADE_COLUMNS}
        if trade.get("status") == "CLOSED":
            row.update(trade_result(trade))
        else:
            row.update({"gross_pnl": None, "estimated_costs": None, "net_pnl": None, "r_multiple": None})
        rows.append(row)
    return pd.DataFrame(rows, columns=TRADE_COLUMNS + ["gross_pnl", "estimated_costs", "net_pnl", "r_multiple"])


def load_trade_ledger(uploaded_file) -> list[dict]:
    """Validate and restore a previously exported CSV paper-trade ledger."""
    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read paper ledger CSV: {exc}") from exc
    missing = [column for column in TRADE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Ledger is missing required columns: " + ", ".join(missing))
    records = []
    seen_ids = set()
    for raw in frame[TRADE_COLUMNS].to_dict("records"):
        row = {key: (None if pd.isna(value) else value) for key, value in raw.items()}
        trade_id = str(row["trade_id"]).strip()
        symbol = str(row["symbol"]).strip().upper()
        side = str(row["side"]).upper()
        timeframe = str(row["timeframe"]).lower()
        status = str(row["status"]).upper()
        if not trade_id or trade_id == "None" or trade_id in seen_ids:
            raise ValueError("Ledger contains a blank or duplicate trade ID.")
        if not symbol or symbol == "NONE" or side not in {"LONG", "SHORT"}:
            raise ValueError("Ledger contains an invalid symbol or trade direction.")
        if timeframe not in {"5m", "15m"} or status not in {"OPEN", "CLOSED"}:
            raise ValueError("Ledger contains an invalid timeframe or status.")
        try:
            entry = float(row["entry_price"])
            stop = float(row["stop_price"])
            target = float(row["target_price"])
            quantity = int(row["quantity"])
            costs = float(row["estimated_cost_bps"] or 0)
            exit_price = None if row["exit_price"] is None else float(row["exit_price"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Ledger contains invalid numeric trade fields.") from exc
        if not all(math.isfinite(v) and v > 0 for v in (entry, stop, target)) or quantity <= 0 or costs < 0:
            raise ValueError("Ledger contains non-positive prices/quantity or invalid costs.")
        if side == "LONG" and not stop < entry < target:
            raise ValueError("Ledger contains invalid LONG stop/target levels.")
        if side == "SHORT" and not target < entry < stop:
            raise ValueError("Ledger contains invalid SHORT stop/target levels.")
        if status == "CLOSED" and (exit_price is None or not math.isfinite(exit_price) or exit_price <= 0):
            raise ValueError("Closed ledger trades must contain a positive exit price.")
        if status == "OPEN" and exit_price is not None:
            raise ValueError("Open ledger trades must not contain an exit price.")
        row.update({
            "trade_id": trade_id, "symbol": symbol, "side": side,
            "timeframe": timeframe, "status": status,
            "entry_price": entry, "stop_price": stop, "target_price": target,
            "quantity": quantity, "estimated_cost_bps": costs,
            "exit_price": exit_price, "notes": "" if row["notes"] is None else str(row["notes"]),
            "opened_at": "" if row["opened_at"] is None else str(row["opened_at"]),
            "closed_at": "" if row["closed_at"] is None else str(row["closed_at"]),
        })
        records.append(row)
        seen_ids.add(trade_id)
    open_records = [t for t in records if t["status"] == "OPEN"]
    if len(open_records) > 3:
        raise ValueError("Ledger has more than 3 open positions; review it before restoring.")
    open_symbols = [t["symbol"] for t in open_records]
    if len(open_symbols) != len(set(open_symbols)):
        raise ValueError("Ledger contains multiple open trades for the same stock.")
    return records
