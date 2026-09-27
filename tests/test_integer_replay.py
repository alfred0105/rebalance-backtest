import pandas as pd

from rebalance_backtest.backtest import BacktestResult
from rebalance_backtest.integer_replay import (
    replay_fractional_targets_as_whole_shares,
)


def test_whole_share_replay_uses_integer_shares_and_nonnegative_cash():
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "A.KS": [100_000, 101_000, 102_000, 103_000, 104_000],
            "SAFE.KS": [50_000, 50_000, 50_000, 50_000, 50_000],
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {
            "A.KS": [0.0] * 5,
            "SAFE.KS": [1.0] * 5,
            "CASH": [0.0] * 5,
        },
        index=dates,
    )
    trades = pd.DataFrame(
        [
            {
                "date": dates[2],
                "turnover": 0.5,
                "cost": 0.0,
                "equity_after_cost": 300_000.0,
                "target_A.KS": 0.5,
                "target_SAFE.KS": 0.5,
                "target_CASH": 0.0,
            }
        ]
    ).set_index("date")
    fractional = BacktestResult(
        equity_curve=pd.Series([300_000.0] * 5, index=dates),
        daily_returns=pd.Series([0.0] * 5, index=dates),
        weights=weights,
        trades=trades,
        metrics={},
    )

    result = replay_fractional_targets_as_whole_shares(
        prices,
        fractional,
        initial_capital=300_000.0,
        transaction_cost_bps=5.0,
    )

    risky = result.shares.drop(columns=["CASH"])
    assert (risky >= 0).all().all()
    assert all(float(x).is_integer() for x in risky.to_numpy().ravel())
    assert (result.shares["CASH"] >= -1e-9).all()
    assert result.equity_curve.iloc[-1] > 0
    assert result.metrics["ending_value"] > 0


def test_whole_share_replay_adds_cash_distributions():
    dates = pd.date_range("2026-01-02", periods=3, freq="B")
    prices = pd.DataFrame({"A.KS": [100_000, 100_000, 100_000]}, index=dates)
    weights = pd.DataFrame(
        {"A.KS": [1.0, 1.0, 1.0], "CASH": [0.0, 0.0, 0.0]},
        index=dates,
    )
    fractional = BacktestResult(
        equity_curve=pd.Series([300_000.0] * 3, index=dates),
        daily_returns=pd.Series([0.0] * 3, index=dates),
        weights=weights,
        trades=pd.DataFrame(),
        metrics={},
    )
    dividends = pd.DataFrame({"A.KS": [0.0, 1_000.0, 0.0]}, index=dates)

    result = replay_fractional_targets_as_whole_shares(
        prices,
        fractional,
        initial_capital=300_000.0,
        transaction_cost_bps=0.0,
        dividends=dividends,
    )

    assert result.metrics["total_distributions"] == 3_000.0
    assert result.equity_curve.iloc[-1] == 303_000.0
    assert not result.cashflows.empty
