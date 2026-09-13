import numpy as np
import pandas as pd

from rebalance_backtest.rotation import AdaptiveRotationStrategy


def _prices(stock_growth: float, bond_growth: float, safe_growth: float) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=320)
    t = np.arange(len(dates), dtype=float)
    wiggle = 1.0 + 0.003 * np.sin(t / 5.0)
    return pd.DataFrame(
        {
            "STOCK": 100.0 * np.exp(stock_growth * t) * wiggle,
            "BOND": 100.0 * np.exp(bond_growth * t) * (1.0 + 0.0015 * np.cos(t / 7.0)),
            "SAFE": 100.0 * np.exp(safe_growth * t),
        },
        index=dates,
    )


def _strategy() -> AdaptiveRotationStrategy:
    return AdaptiveRotationStrategy(
        stock_tickers=["STOCK"],
        bond_tickers=["BOND"],
        safe_ticker="SAFE",
        no_trade_band=0.05,
        max_weekly_shift=0.10,
    )


def test_rotation_prefers_stock_when_stock_has_best_excess_trend():
    prices = _prices(stock_growth=0.0012, bond_growth=0.0003, safe_growth=0.0001)
    rec = _strategy().recommend(prices)
    assert rec.regime == "STOCK"
    assert rec.target_weights["STOCK"] > 0.9


def test_rotation_moves_to_bond_when_stock_weakens():
    prices = _prices(stock_growth=-0.0004, bond_growth=0.0007, safe_growth=0.0001)
    rec = _strategy().recommend(prices)
    assert rec.regime == "BOND"
    assert rec.target_weights["BOND"] > 0.9


def test_rotation_moves_to_safe_when_risky_assets_lag_safe():
    prices = _prices(stock_growth=-0.0005, bond_growth=0.00002, safe_growth=0.00015)
    rec = _strategy().recommend(prices)
    assert rec.regime == "SAFE"
    assert rec.target_weights["SAFE"] == 1.0


def test_rotation_caps_one_week_shift_to_ten_percentage_points():
    prices = _prices(stock_growth=0.0012, bond_growth=0.0003, safe_growth=0.0001)
    current = pd.Series({"STOCK": 0.0, "BOND": 0.0, "SAFE": 1.0})
    rec = _strategy().recommend(prices, current, apply_no_trade_band=True)
    assert np.isclose(rec.target_weights["STOCK"], 0.10)
    assert np.isclose(rec.target_weights["SAFE"], 0.90)


def test_rotation_does_nothing_inside_five_percentage_point_band():
    prices = _prices(stock_growth=0.0012, bond_growth=0.0003, safe_growth=0.0001)
    current = pd.Series({"STOCK": 0.92, "BOND": 0.0, "SAFE": 0.08})
    rec = _strategy().recommend(prices, current, apply_no_trade_band=True)
    pd.testing.assert_series_equal(rec.target_weights, current.astype(float))
