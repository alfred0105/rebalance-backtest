"""Small example: four-ETF portfolio with weekly momentum overlay."""

from rebalance_backtest.backtest import run_backtest
from rebalance_backtest.data import fetch_prices
from rebalance_backtest.strategy import FixedWeightStrategy, MomentumTiltStrategy

prices = fetch_prices(["SPY", "QQQ", "TLT", "GLD"], "2012-01-01")
base = {"SPY": 0.35, "QQQ": 0.25, "TLT": 0.20, "GLD": 0.20}

monthly = run_backtest(
    prices,
    FixedWeightStrategy(base),
    schedule="M",
    transaction_cost_bps=5,
)
weekly_momentum = run_backtest(
    prices,
    MomentumTiltStrategy(base),
    schedule="W-FRI",
    transaction_cost_bps=5,
)

print("Monthly fixed:", monthly.metrics)
print("Weekly momentum:", weekly_momentum.metrics)
