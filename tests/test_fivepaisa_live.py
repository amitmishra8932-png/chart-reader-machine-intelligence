import json
import unittest

import pandas as pd

from providers.fivepaisa_live import (
    LiveCandleBuilder,
    parse_live_tick_message,
)


class FivePaisaLiveFeedTests(unittest.TestCase):
    def test_parse_marketfeed_v3_tick(self):
        payload = [{
            "Exch": "N",
            "ExchType": "C",
            "Token": 2885,
            "LastRate": 2915.4,
            "LastQty": 10,
            "TotalQty": 1200,
            "OpenRate": 2900.0,
            "High": 2920.0,
            "Low": 2890.0,
            "PClose": 2853.3,
            "TickDt": "/Date(1706956047000+0530)/",
        }]
        ticks = parse_live_tick_message(json.dumps(payload), {2885: "RELIANCE"})
        self.assertEqual(len(ticks), 1)
        tick = ticks[0]
        self.assertEqual(tick["symbol"], "RELIANCE")
        self.assertEqual(tick["scrip_code"], 2885)
        self.assertEqual(tick["last_price"], 2915.4)
        self.assertEqual(tick["total_qty"], 1200)
        self.assertEqual(str(tick["timestamp"].tz), "Asia/Kolkata")

    def test_parse_rejects_bad_json_and_skips_invalid_ticks(self):
        with self.assertRaises(ValueError):
            parse_live_tick_message("{bad-json", {})
        ticks = parse_live_tick_message(
            json.dumps([{"Token": 1, "LastRate": "not-a-price", "TickDt": "bad"}]),
            {1: "ABC"},
        )
        self.assertEqual(ticks, [])

    def test_aggregates_ticks_into_completed_five_minute_candle(self):
        builder = LiveCandleBuilder(5)
        base = pd.Timestamp("2026-11-18 09:15", tz="Asia/Kolkata")
        def tick(minute, price, total, last=5):
            return {
                "timestamp": base + pd.Timedelta(minutes=minute),
                "symbol": "RELIANCE",
                "last_price": price,
                "total_qty": total,
                "last_qty": last,
            }

        self.assertIsNone(builder.add_tick(tick(0, 100, 100)))
        self.assertIsNone(builder.add_tick(tick(1, 102, 150)))
        self.assertIsNone(builder.add_tick(tick(4, 99, 200)))
        completed = builder.add_tick(tick(5, 101, 230))
        self.assertIsNotNone(completed)
        self.assertEqual(completed["timestamp"], base)
        self.assertEqual(completed["open"], 100)
        self.assertEqual(completed["high"], 102)
        self.assertEqual(completed["low"], 99)
        self.assertEqual(completed["close"], 99)
        self.assertEqual(completed["volume"], 105)
        self.assertTrue(completed["is_complete"])

    def test_ignores_out_of_order_tick_without_corrupting_volume_counter(self):
        builder = LiveCandleBuilder(5)
        base = pd.Timestamp("2026-11-18 09:15", tz="Asia/Kolkata")
        def tick(timestamp, price, total):
            return {
                "timestamp": timestamp, "symbol": "ABC",
                "last_price": price, "total_qty": total, "last_qty": 1,
            }

        builder.add_tick(tick(base, 100, 100))
        builder.add_tick(tick(base + pd.Timedelta(minutes=5), 101, 120))
        stale = tick(base + pd.Timedelta(minutes=2), 99, 500)
        self.assertIsNone(builder.add_tick(stale))
        completed = builder.add_tick(
            tick(base + pd.Timedelta(minutes=10), 102, 130)
        )
        self.assertEqual(completed["volume"], 20)

    def test_ignores_out_of_order_tick_inside_current_candle(self):
        builder = LiveCandleBuilder(5)
        base = pd.Timestamp("2026-11-18 09:15", tz="Asia/Kolkata")
        def tick(minute, price, total):
            return {
                "timestamp": base + pd.Timedelta(minutes=minute),
                "symbol": "ABC", "last_price": price,
                "total_qty": total, "last_qty": 1,
            }

        builder.add_tick(tick(0, 100, 100))
        builder.add_tick(tick(2, 102, 150))
        builder.add_tick(tick(1, 99, 180))
        completed = builder.add_tick(tick(5, 101, 200))
        self.assertEqual(completed["close"], 102)
        self.assertEqual(completed["volume"], 51)

    def test_rejects_unsupported_candle_interval(self):
        with self.assertRaises(ValueError):
            LiveCandleBuilder(3)


if __name__ == "__main__":
    unittest.main()
