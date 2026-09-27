import pandas as pd

from rebalance_backtest.report_cli import _integer_plan


def test_integer_plan_respects_capital_and_whole_shares():
    weights = {
        "A.KS": 0.45,
        "B.KS": 0.30,
        "C.KS": 0.25,
    }
    prices = {
        "A.KS": 113_000.0,
        "B.KS": 61_000.0,
        "C.KS": 14_500.0,
    }
    names = {ticker: ticker for ticker in weights}

    plan, cash = _integer_plan(weights, prices, names, 300_000.0)

    assert not plan.empty
    assert (plan["shares"] >= 0).all()
    assert all(float(x).is_integer() for x in plan["shares"].astype(float))
    assert plan["actual_value"].sum() + cash <= 300_000.0 + 1e-9
    assert cash >= -1e-9


def test_integer_plan_keeps_unaffordable_target_as_zero_shares():
    weights = {"EXPENSIVE.KS": 1.0}
    prices = {"EXPENSIVE.KS": 500_000.0}
    names = {"EXPENSIVE.KS": "Expensive"}

    plan, cash = _integer_plan(weights, prices, names, 300_000.0)

    assert int(plan.loc[0, "shares"]) == 0
    assert cash == 300_000.0
