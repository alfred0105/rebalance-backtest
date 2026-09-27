import pandas as pd

from rebalance_backtest.execution_plan import build_execution_plan


def test_execution_plan_blocks_excessive_first_share_overweight():
    plan = build_execution_plan(
        {"EXPENSIVE.KS": 0.15, "CHEAP.KS": 0.15},
        {"EXPENSIVE.KS": 61_290.0, "CHEAP.KS": 14_500.0},
        {"EXPENSIVE.KS": "Expensive", "CHEAP.KS": "Cheap"},
        300_000.0,
        max_overweight_pp=0.04,
    )

    expensive = plan.positions.set_index("ticker").loc["EXPENSIVE.KS"]
    assert int(expensive["shares"]) == 0
    assert bool(expensive["blocked_by_size"])
    assert plan.cash >= 0


def test_execution_plan_charges_toss_commission_and_integer_prices():
    plan = build_execution_plan(
        {"A.KS": 0.50, "B.KS": 0.50},
        {"A.KS": 50_000.0, "B.KS": 25_000.0},
        {"A.KS": "A", "B.KS": "B"},
        300_000.0,
    )

    assert plan.total_commission >= 0
    assert plan.cash >= 0
    assert pd.api.types.is_integer_dtype(plan.positions["price"])
    assert pd.api.types.is_integer_dtype(plan.positions["shares"])
