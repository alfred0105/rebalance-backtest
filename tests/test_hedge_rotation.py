import pandas as pd

from rebalance_backtest.hedge_rotation import PeakHedgeRotationStrategy


def _strategy(**kwargs):
    params = dict(
        stock_tickers=["MKT", "SEC1", "SEC2"],
        bond_tickers=["KR3Y"],
        hedge_tickers=["GOLD", "USD", "UST30"],
        safe_ticker="SAFE",
        market_ticker="MKT",
        market_weight=0.75,
        sector_breadth_weight=0.25,
    )
    params.update(kwargs)
    return PeakHedgeRotationStrategy(**params)


def test_peak_cap_ladder():
    strategy = _strategy()
    assert strategy._peak_cap(-0.03) == 1.0
    assert strategy._peak_cap(-0.04) == 0.80
    assert strategy._peak_cap(-0.06) == 0.65
    assert strategy._peak_cap(-0.08) == 0.50
    assert strategy._peak_cap(-0.10) == 0.35
    assert strategy._peak_cap(-0.15) == 0.35


def test_peak_protection_needs_confirmation():
    strategy = _strategy()
    stock, _, _, emergency, peak = strategy._stock_and_defensive_targets(
        market_score=0.60,
        defensive_score=0.30,
        short_return=0.02,
        peak_drawdown=-0.08,
        breadth_score=0.10,
    )
    assert not emergency
    assert not peak
    assert stock == 0.90

    stock, _, _, emergency, peak = strategy._stock_and_defensive_targets(
        market_score=0.60,
        defensive_score=0.30,
        short_return=-0.01,
        peak_drawdown=-0.08,
        breadth_score=0.10,
    )
    assert not emergency
    assert peak
    assert stock == 0.50


def test_defensive_rotation_prefers_positive_leaders():
    strategy = _strategy()
    columns = ["MKT", "SEC1", "SEC2", "KR3Y", "GOLD", "USD", "UST30", "SAFE"]
    scores = pd.Series(
        {
            "MKT": 0.50,
            "SEC1": 0.40,
            "SEC2": 0.20,
            "KR3Y": -0.10,
            "GOLD": 0.60,
            "USD": 0.30,
            "UST30": -0.20,
            "SAFE": 0.0,
        }
    )
    target = strategy._ideal_peak_hedge_target(
        scores=scores,
        columns=columns,
        date=pd.Timestamp("2026-09-22"),
        stock_target=0.50,
        defensive_target=0.40,
    )

    assert target["GOLD"] > 0
    assert target["USD"] > 0
    assert target["KR3Y"] == 0
    assert target["UST30"] == 0
    assert abs(float(target.sum()) - 1.0) < 1e-12
