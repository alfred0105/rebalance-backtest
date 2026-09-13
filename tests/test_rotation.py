import pandas as pd

from rebalance_backtest.rotation import AdaptiveRotationStrategy


def _strategy() -> AdaptiveRotationStrategy:
    return AdaptiveRotationStrategy(
        stock_tickers=["MKT", "A", "B", "C"],
        bond_tickers=["BOND"],
        safe_ticker="SAFE",
        market_ticker="MKT",
    )


def _current(**kwargs) -> pd.Series:
    s = pd.Series(0.0, index=["MKT", "A", "B", "C", "BOND", "SAFE"])
    for key, value in kwargs.items():
        s[key] = value
    if s.sum() < 1.0:
        s["SAFE"] += 1.0 - s.sum()
    return s


def test_hot_sector_cannot_force_stock_regime_when_market_breadth_is_weak():
    st = _strategy()
    scores = pd.Series(
        {"MKT": -0.30, "A": 2.00, "B": -0.50, "C": -0.40, "BOND": -0.20, "SAFE": 0.0}
    )
    regime, market_score, _, _ = st._choose_regime(scores, _current(SAFE=1.0))
    assert market_score < 0
    assert regime == "SAFE"


def test_stock_hysteresis_requires_more_to_enter_than_to_stay():
    st = _strategy()
    scores = pd.Series(
        {"MKT": 0.05, "A": 0.06, "B": 0.04, "C": 0.03, "BOND": -0.20, "SAFE": 0.0}
    )
    assert st._choose_regime(
        scores, _current(MKT=0.30, A=0.275, B=0.275, SAFE=0.15)
    )[0] == "STOCK"
    assert st._choose_regime(scores, _current(SAFE=1.0))[0] == "SAFE"


def test_risk_on_adds_only_five_percentage_points_per_week():
    st = _strategy()
    scores = pd.Series(
        {"MKT": 1.0, "A": 0.9, "B": 0.8, "C": 0.1, "BOND": -0.2, "SAFE": 0.0}
    )
    ideal = st._ideal_target("STOCK", scores, _current().index)
    nxt = st._apply_staged_transition(ideal, _current(SAFE=1.0), "STOCK")
    assert abs(nxt[st.stock_tickers].sum() - 0.05) < 1e-9
    assert abs(nxt["SAFE"] - 0.95) < 1e-9


def test_risk_off_cuts_fifteen_percentage_points_per_week():
    st = _strategy()
    scores = pd.Series(
        {"MKT": -1.0, "A": -0.8, "B": -0.7, "C": -0.6, "BOND": -0.2, "SAFE": 0.0}
    )
    ideal = st._ideal_target("SAFE", scores, _current().index)
    cur = _current(MKT=0.30, A=0.275, B=0.275, SAFE=0.15)
    nxt = st._apply_staged_transition(ideal, cur, "SAFE")
    assert abs(nxt[st.stock_tickers].sum() - 0.70) < 1e-9
    assert abs(nxt["SAFE"] - 0.30) < 1e-9


def test_stock_ideal_has_core_and_sector_caps():
    st = _strategy()
    scores = pd.Series(
        {"MKT": 0.7, "A": 1.0, "B": 0.8, "C": 0.2, "BOND": -0.1, "SAFE": 0.0}
    )
    ideal = st._ideal_target("STOCK", scores, _current().index)
    assert abs(ideal["MKT"] - 0.30) < 1e-9
    assert ideal[["A", "B", "C"]].max() <= 0.30 + 1e-9
    assert abs(ideal[st.stock_tickers].sum() - 0.85) < 1e-9
    assert abs(ideal["SAFE"] - 0.15) < 1e-9
