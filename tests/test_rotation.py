import numpy as np
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


def test_stock_target_changes_continuously_with_market_score():
    st = _strategy()
    assert np.isclose(st._stock_target_from_score(-0.30), 0.00)
    assert np.isclose(st._stock_target_from_score(0.00), 0.35)
    assert np.isclose(st._stock_target_from_score(0.30), 0.75)
    assert np.isclose(st._stock_target_from_score(0.60), 0.90)


def test_bond_only_uses_residual_when_bond_score_is_positive():
    st = _strategy()
    stock, bond, safe, brake = st._allocation_targets(0.30, 0.30, 0.02)
    assert not brake
    assert np.isclose(stock, 0.75)
    assert np.isclose(bond, (1.0 - stock) * 0.90)
    assert np.isclose(stock + bond + safe, 1.0)

    stock2, bond2, safe2, _ = st._allocation_targets(0.30, -0.10, 0.02)
    assert np.isclose(stock2, 0.75)
    assert np.isclose(bond2, 0.0)
    assert np.isclose(safe2, 0.25)


def test_crash_brake_caps_stock_target():
    st = _strategy()
    stock, _, _, brake = st._allocation_targets(0.60, -0.10, -0.10)
    assert brake
    assert np.isclose(stock, st.brake_stock_cap)


def test_daily_risk_on_adds_five_percentage_points():
    st = _strategy()
    ideal = pd.Series(
        {"MKT": 0.40, "A": 0.20, "B": 0.20, "C": 0.0, "BOND": 0.0, "SAFE": 0.20}
    )
    nxt = st._apply_staged_transition(ideal, _current(SAFE=1.0), emergency=False)
    assert np.isclose(nxt[st.stock_tickers].sum(), 0.05)
    assert np.isclose(nxt["SAFE"], 0.95)


def test_normal_daily_risk_off_cuts_twenty_percentage_points():
    st = _strategy()
    ideal = pd.Series(
        {"MKT": 0.10, "A": 0.05, "B": 0.05, "C": 0.0, "BOND": 0.0, "SAFE": 0.80}
    )
    cur = _current(MKT=0.40, A=0.20, B=0.20, SAFE=0.20)
    nxt = st._apply_staged_transition(ideal, cur, emergency=False)
    assert np.isclose(nxt[st.stock_tickers].sum(), 0.60)


def test_emergency_brake_can_cut_thirty_five_percentage_points_in_one_day():
    st = _strategy()
    ideal = pd.Series(
        {"MKT": 0.15, "A": 0.10, "B": 0.10, "C": 0.0, "BOND": 0.0, "SAFE": 0.65}
    )
    cur = _current(MKT=0.40, A=0.20, B=0.20, SAFE=0.20)
    nxt = st._apply_staged_transition(ideal, cur, emergency=True)
    assert np.isclose(nxt[st.stock_tickers].sum(), 0.45)


def test_sector_selection_only_refreshes_when_month_changes():
    st = _strategy()
    jan_scores = pd.Series({"MKT": 0.5, "A": 1.0, "B": 0.8, "C": 0.1, "BOND": 0.0, "SAFE": 0.0})
    jan = st._refresh_monthly_sectors(jan_scores, pd.Timestamp("2026-01-05"))
    assert jan == ("A", "B")

    changed_scores = pd.Series({"MKT": 0.5, "A": 0.1, "B": 0.2, "C": 2.0, "BOND": 0.0, "SAFE": 0.0})
    still_jan = st._refresh_monthly_sectors(changed_scores, pd.Timestamp("2026-01-20"))
    assert still_jan == ("A", "B")

    feb = st._refresh_monthly_sectors(changed_scores, pd.Timestamp("2026-02-02"))
    assert feb == ("C", "B")
