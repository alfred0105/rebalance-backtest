import numpy as np
import pandas as pd

from rebalance_backtest.strategy import MomentumTiltStrategy


def test_momentum_tilt_overweights_stronger_asset():
    dates = pd.bdate_range("2024-01-01", periods=180)
    # A trends strongly upward; B trends mildly downward. Add small oscillation
    # so realized volatility is non-zero.
    t = np.arange(len(dates))
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(0.0025 * t) * (1 + 0.005 * np.sin(t / 3)),
            "B": 100 * np.exp(-0.0007 * t) * (1 + 0.005 * np.cos(t / 4)),
        },
        index=dates,
    )
    strategy = MomentumTiltStrategy(
        {"A": 0.5, "B": 0.5},
        no_trade_band=0.0,
        entry_speed=1.0,
        exit_speed=1.0,
    )
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = strategy.target_weights(prices, current)

    assert target["A"] > target["B"]
    assert 0 <= target.sum() <= 1.0 + 1e-12


def test_insufficient_history_keeps_current_weights():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame({"A": np.linspace(100, 110, 30)}, index=dates)
    strategy = MomentumTiltStrategy({"A": 0.8})
    current = pd.Series({"A": 0.73})
    target = strategy.target_weights(prices, current)
    pd.testing.assert_series_equal(target, current, check_names=False)
