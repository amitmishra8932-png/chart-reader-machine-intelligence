"""Tests for manual paper-trading risk controls and ledger maths."""
import unittest
from io import BytesIO

from engine.paper_trading import (
    close_paper_trade, load_trade_ledger, open_paper_trade, paper_trading_summary,
    suggest_quantity, trade_result, trades_frame,
)


class PaperTradingTests(unittest.TestCase):
    def test_quantity_respects_stop_risk_and_cash_exposure(self):
        qty = suggest_quantity(100000, 1.0, 100, 95)
        self.assertEqual(qty, 200)
        self.assertLessEqual(qty * 5, 1000)
        self.assertLessEqual(qty * 100, 100000)

    def test_quantity_rejects_invalid_stop(self):
        with self.assertRaisesRegex(ValueError, "different"):
            suggest_quantity(100000, 1.0, 100, 100)

    def test_open_long_trade(self):
        trade = open_paper_trade(
            [], symbol="reliance", timeframe="5m", side="LONG",
            entry_price=100, stop_price=98, target_price=104,
            quantity=100, capital=100000, opened_at="2026-11-13T09:20:00+05:30",
        )
        self.assertEqual(trade["symbol"], "RELIANCE")
        self.assertEqual(trade["status"], "OPEN")

    def test_open_trade_rejects_wrong_long_levels(self):
        with self.assertRaisesRegex(ValueError, "LONG"):
            open_paper_trade(
                [], symbol="RELIANCE", timeframe="5m", side="LONG",
                entry_price=100, stop_price=102, target_price=104,
                quantity=10, capital=100000,
            )

    def test_open_trade_enforces_aggregate_risk(self):
        first = open_paper_trade(
            [], symbol="RELIANCE", timeframe="5m", side="LONG",
            entry_price=100, stop_price=95, target_price=110,
            quantity=200, capital=100000,
        )
        with self.assertRaisesRegex(ValueError, "total open-risk"):
            open_paper_trade(
                [first], symbol="HDFCBANK", timeframe="5m", side="LONG",
                entry_price=100, stop_price=95, target_price=110,
                quantity=500, capital=100000,
            )

    def test_open_trade_enforces_total_cash_exposure(self):
        first = open_paper_trade(
            [], symbol="RELIANCE", timeframe="5m", side="LONG",
            entry_price=100, stop_price=99, target_price=105,
            quantity=500, capital=100000,
        )
        with self.assertRaisesRegex(ValueError, "cash-exposure"):
            open_paper_trade(
                [first], symbol="HDFCBANK", timeframe="5m", side="LONG",
                entry_price=100, stop_price=99, target_price=105,
                quantity=600, capital=100000,
            )

    def test_only_one_open_trade_per_symbol(self):
        first = open_paper_trade(
            [], symbol="RELIANCE", timeframe="5m", side="LONG",
            entry_price=100, stop_price=95, target_price=110,
            quantity=100, capital=100000,
        )
        with self.assertRaisesRegex(ValueError, "already has"):
            open_paper_trade(
                [first], symbol="RELIANCE", timeframe="15m", side="SHORT",
                entry_price=100, stop_price=105, target_price=95,
                quantity=100, capital=100000,
            )

    def test_close_and_calculate_long_pnl_after_costs(self):
        trade = open_paper_trade(
            [], symbol="RELIANCE", timeframe="5m", side="LONG",
            entry_price=100, stop_price=95, target_price=110,
            quantity=100, capital=100000, estimated_cost_bps=10,
        )
        closed = close_paper_trade(trade, 105, closed_at="2026-11-13T10:00:00+05:30")
        result = trade_result(closed)
        self.assertAlmostEqual(result["gross_pnl"], 500)
        self.assertAlmostEqual(result["estimated_costs"], 2.05)
        self.assertAlmostEqual(result["net_pnl"], 497.95)

    def test_short_pnl_direction(self):
        trade = open_paper_trade(
            [], symbol="ICICIBANK", timeframe="15m", side="SHORT",
            entry_price=100, stop_price=105, target_price=90,
            quantity=10, capital=100000, estimated_cost_bps=0,
        )
        closed = close_paper_trade(trade, 95)
        self.assertAlmostEqual(trade_result(closed)["net_pnl"], 50)


    def test_exported_ledger_can_be_restored(self):
        trade = open_paper_trade(
            [], symbol="RELIANCE", timeframe="5m", side="LONG",
            entry_price=100, stop_price=95, target_price=110,
            quantity=100, capital=100000, opened_at="2026-11-13T09:20:00+05:30",
        )
        exported = trades_frame([trade]).to_csv(index=False).encode("utf-8")
        restored = load_trade_ledger(BytesIO(exported))
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["trade_id"], trade["trade_id"])
        self.assertEqual(restored[0]["status"], "OPEN")

    def test_restore_rejects_invalid_ledger(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            load_trade_ledger(BytesIO(b"symbol,close\\nRELIANCE,100\\n"))

    def test_summary_uses_closed_trades_only(self):
        trade = open_paper_trade(
            [], symbol="RELIANCE", timeframe="5m", side="LONG",
            entry_price=100, stop_price=95, target_price=110,
            quantity=100, capital=100000, estimated_cost_bps=0,
        )
        closed = close_paper_trade(trade, 105)
        summary = paper_trading_summary([closed], 100000)
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(summary["open_trades"], 0)
        self.assertAlmostEqual(summary["net_pnl"], 500)
        self.assertEqual(summary["win_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
