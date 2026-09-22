import pandas as pd

from rebalance_backtest.dynamic_selector import (
    DynamicSelectionConfig,
    select_dynamic_universe,
)


def _row(
    ticker,
    name,
    bucket,
    score,
    cluster,
    *,
    momentum=0.5,
    r1=0.0,
    r5=0.0,
    r20=0.0,
    peak=-0.02,
    corr=0.2,
    vol=0.2,
):
    return {
        "ticker": ticker,
        "name": name,
        "bucket": bucket,
        "screen_score": score,
        "momentum_score": momentum,
        "return_1d": r1,
        "return_5d": r5,
        "return_20d": r20,
        "return_63d": 0.05,
        "peak_drawdown_60d": peak,
        "annual_vol_63d": vol,
        "corr_market_126d": corr,
        "max_drawdown_252d": -0.15,
        "cluster_id": cluster,
    }


def _scored(*, crash_market=False, shocked_a=False):
    rows = [
        _row(
            "069500.KS",
            "KODEX 200",
            "EQUITY",
            0.60,
            0,
            r5=-0.07 if crash_market else 0.01,
            r20=-0.02,
        ),
        _row(
            "AAA.KS",
            "Alpha",
            "EQUITY",
            0.90,
            1,
            r1=-0.06 if shocked_a else 0.0,
        ),
        _row("BBB.KS", "Beta", "EQUITY", 0.80, 2),
        _row("CCC.KS", "Gamma", "EQUITY", 0.70, 3),
        _row("AAA2.KS", "Alpha clone", "EQUITY", 0.85, 1),
        _row("GOLD.KS", "Gold", "REAL_ASSET", 0.75, 10, corr=-0.1),
        _row("BOND.KS", "Bond", "DEFENSIVE", 0.65, 11, corr=0.1, vol=0.08),
        _row("FAKE.KS", "Stock bond mix", "DEFENSIVE", 0.95, 12, corr=0.95),
    ]
    return pd.DataFrame(rows)


def test_initial_selection_uses_unique_clusters_and_reserves_market_core():
    config = DynamicSelectionConfig(
        aggressive_slots=2,
        defensive_slots=2,
    )
    state, active = select_dynamic_universe(
        _scored(),
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    assert active["aggressive"] == ["AAA.KS", "BBB.KS"]
    assert "069500.KS" not in active["aggressive"]
    assert "AAA2.KS" not in active["aggressive"]
    assert active["defensive"] == ["GOLD.KS", "BOND.KS"]
    assert "FAKE.KS" not in active["defensive"]
    assert not active["market_emergency"]
    assert len(state["aggressive"]) == 2


def test_minimum_hold_blocks_normal_monthly_replacement():
    config = DynamicSelectionConfig(
        aggressive_slots=1,
        defensive_slots=0,
        min_hold_business_days=20,
        replacement_score_margin=0.15,
    )
    previous = {
        "version": 1,
        "last_run_date": "2026-08-31",
        "last_selection_month": "2026-08",
        "aggressive": [
            {
                "ticker": "CCC.KS",
                "name": "Gamma",
                "entered": "2026-09-01",
                "cluster_id": 3,
                "last_score": 0.70,
            }
        ],
        "defensive": [],
        "cooldowns": {},
    }

    _, active = select_dynamic_universe(
        _scored(),
        previous_state=previous,
        as_of=pd.Timestamp("2026-09-10"),
        config=config,
    )

    assert active["aggressive"] == ["CCC.KS"]
    assert not any(
        event["action"] == "MONTHLY_REPLACE"
        for event in active["events"]
    )


def test_monthly_replacement_occurs_after_minimum_hold_and_margin():
    config = DynamicSelectionConfig(
        aggressive_slots=1,
        defensive_slots=0,
        min_hold_business_days=20,
        replacement_score_margin=0.15,
    )
    previous = {
        "version": 1,
        "last_run_date": "2026-08-31",
        "last_selection_month": "2026-08",
        "aggressive": [
            {
                "ticker": "CCC.KS",
                "name": "Gamma",
                "entered": "2026-07-01",
                "cluster_id": 3,
                "last_score": 0.70,
            }
        ],
        "defensive": [],
        "cooldowns": {},
    }

    _, active = select_dynamic_universe(
        _scored(),
        previous_state=previous,
        as_of=pd.Timestamp("2026-09-10"),
        config=config,
    )

    assert active["aggressive"] == ["AAA.KS"]
    assert any(
        event["action"] == "MONTHLY_REPLACE"
        and event["out"] == "CCC.KS"
        and event["in"] == "AAA.KS"
        for event in active["events"]
    )


def test_individual_shock_overrides_minimum_hold_and_starts_cooldown():
    config = DynamicSelectionConfig(
        aggressive_slots=1,
        defensive_slots=0,
        min_hold_business_days=20,
        cooldown_business_days=5,
    )
    previous = {
        "version": 1,
        "last_run_date": "2026-09-21",
        "last_selection_month": "2026-09",
        "aggressive": [
            {
                "ticker": "AAA.KS",
                "name": "Alpha",
                "entered": "2026-09-21",
                "cluster_id": 1,
                "last_score": 0.90,
            }
        ],
        "defensive": [],
        "cooldowns": {},
    }

    state, active = select_dynamic_universe(
        _scored(shocked_a=True),
        previous_state=previous,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    assert active["aggressive"] == ["BBB.KS"]
    assert "AAA.KS" in state["cooldowns"]
    assert any(
        event["action"] == "EMERGENCY_EXIT"
        and event["ticker"] == "AAA.KS"
        for event in active["events"]
    )


def test_market_shock_exits_aggressive_and_prevents_refill():
    config = DynamicSelectionConfig(
        aggressive_slots=2,
        defensive_slots=1,
    )
    previous = {
        "version": 1,
        "last_run_date": "2026-09-21",
        "last_selection_month": "2026-09",
        "aggressive": [
            {
                "ticker": "AAA.KS",
                "name": "Alpha",
                "entered": "2026-09-01",
                "cluster_id": 1,
                "last_score": 0.90,
            },
            {
                "ticker": "BBB.KS",
                "name": "Beta",
                "entered": "2026-09-01",
                "cluster_id": 2,
                "last_score": 0.80,
            },
        ],
        "defensive": [],
        "cooldowns": {},
    }

    state, active = select_dynamic_universe(
        _scored(crash_market=True),
        previous_state=previous,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    assert active["market_emergency"]
    assert active["aggressive"] == []
    assert len(active["defensive"]) == 1
    assert "AAA.KS" in state["cooldowns"]
    assert "BBB.KS" in state["cooldowns"]
