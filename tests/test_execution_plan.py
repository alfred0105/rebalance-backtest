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


def test_large_core_sleeve_is_not_blocked_by_small_account_rounding():
    plan = build_execution_plan(
        {"CORE.KS": 0.325, "SAFE.KS": 0.30, "SAT.KS": 0.108},
        {"CORE.KS": 111_400.0, "SAFE.KS": 113_207.0, "SAT.KS": 44_685.0},
        {"CORE.KS": "Core", "SAFE.KS": "Safe", "SAT.KS": "Satellite"},
        300_000.0,
        max_overweight_pp=0.04,
    )

    positions = plan.positions.set_index("ticker")
    assert int(positions.loc["CORE.KS", "shares"]) >= 1
    assert int(positions.loc["SAFE.KS", "shares"]) >= 1
    assert not bool(positions.loc["CORE.KS", "blocked_by_size"])
    assert not bool(positions.loc["SAFE.KS", "blocked_by_size"])
    assert bool(positions.loc["SAT.KS", "blocked_by_size"])
    assert plan.cash < 100_000.0
