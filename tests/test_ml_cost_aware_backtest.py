import pandas as pd
import pytest

from engine.ml_cost_aware_backtest import (
    simulate_fixed_horizon_trades,
    summarize_trades,
)


def sample_rows():
    times = pd.date_range("2026-01-05 09:15", periods=7, freq="5min")
    return pd.DataFrame({
        "timestamp": times,
        "symbol": ["ABC"] * 7,
        "session_date": ["2026-01-05"] * 7,
        "open": [100, 101, 102, 103, 104, 105, 106],
        "close": [100, 102, 103, 104, 105, 106, 107],
        "prediction": [1, 1, 1, -1, 0, 0, 0],
    })


def test_entry_next_open_exit_after_full_horizon_and_cost():
    trades = simulate_fixed_horizon_trades(
        sample_rows(), horizon_bars=2, round_trip_cost_bps=10
    )
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_time"] == pd.Timestamp("2026-01-05 09:20")
    assert trade["exit_time"] == pd.Timestamp("2026-01-05 09:30")
    assert trade["side"] == "LONG"
    assert trade["entry_price"] == 101
    assert trade["exit_price"] == 104
    assert trade["net_return"] == pytest.approx((104 / 101 - 1) - 0.001)


def test_does_not_cross_timestamp_gap_or_session():
    frame = sample_rows()
    frame.loc[2, "timestamp"] = pd.Timestamp("2026-01-05 09:26")
    trades = simulate_fixed_horizon_trades(frame, horizon_bars=2)
    assert trades.empty


def test_non_overlapping_and_neutral_signals_skipped():
    frame = sample_rows()
    trades = simulate_fixed_horizon_trades(frame, horizon_bars=2)
    assert len(trades) == 1
    assert trades.iloc[0]["signal_time"] == pd.Timestamp("2026-01-05 09:15")


def test_cost_and_prediction_validation():
    frame = sample_rows()
    with pytest.raises(ValueError):
        simulate_fixed_horizon_trades(frame, horizon_bars=2, round_trip_cost_bps=-1)
    frame.loc[0, "prediction"] = 2
    with pytest.raises(ValueError):
        simulate_fixed_horizon_trades(frame, horizon_bars=2)


def test_summary_reports_trade_level_metrics():
    trades = simulate_fixed_horizon_trades(sample_rows(), horizon_bars=2)
    summary = summarize_trades(trades)
    assert summary["trade_count"] == 1
    assert summary["net_win_rate_pct"] == 100.0
