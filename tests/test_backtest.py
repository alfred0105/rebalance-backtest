import numpy as np
import pandas as pd

from rebalance_backtest.backtest import run_backtest
from rebalance_backtest.strategy import FixedWeightStrategy


def _prices():
    dates = pd.bdate_range("2024-01-01", periods=100)
    t = np.arange(len(dates))
    return pd.DataFrame(
        {
            "A": 100 * np.exp(0.001 * t),
            "B": 100 * np.exp(-0.0002 * t),
        },
        index=dates,
    )


def test_costs_reduce_final_equity_when_rebalancing_occurs():
    prices = _prices()
    strategy = FixedWeightStrategy({"A": 0.5, "B": 0.5})
    free = run_backtest(prices, strategy, schedule="W-FRI", transaction_cost_bps=0)
    costly = run_backtest(prices, strategy, schedule="W-FRI", transaction_cost_bps=25)

    assert len(costly.trades) > 0
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]


def test_buy_and_hold_has_no_rebalance_trades():
    prices = _prices()
    strategy = FixedWeightStrategy({"A": 0.6, "B": 0.4})
    result = run_backtest(prices, strategy, schedule=None, transaction_cost_bps=10)
    assert result.trades.empty
    assert result.equity_curve.iloc[-1] > 0


class _NoOpDailyStrategy:
    def initial_weights(self, columns):
        return pd.Series({"A": 0.6, "B": 0.4}).reindex(columns).fillna(0.0)

    def target_weights(self, price_history, current_weights):
        return current_weights.copy()


def test_daily_noop_signal_does_not_create_phantom_rebalances():
    prices = _prices()
    result = run_backtest(
        prices,
        _NoOpDailyStrategy(),
        schedule="D",
        transaction_cost_bps=10,
    )
    assert result.trades.empty
    assert result.metrics["total_turnover"] == 0.0
    assert result.metrics["total_transaction_cost"] == 0.0
