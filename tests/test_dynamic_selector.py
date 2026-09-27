import pandas as pd

from rebalance_backtest.dynamic_selector import (
    DynamicSelectionConfig,
    build_dynamic_target_allocation,
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
    theme="OTHER",
    role="RISK",
    price=10_000.0,
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
        "theme": theme,
        "defensive_role": role,
        "price_listing": price,
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
            "Alpha Semiconductor",
            "EQUITY",
            0.90,
            1,
            r1=-0.06 if shocked_a else 0.0,
            theme="SEMICONDUCTOR",
        ),
        _row("BBB.KS", "Beta Dividend", "EQUITY", 0.80, 2, theme="DIVIDEND"),
        _row("CCC.KS", "Gamma Bank", "EQUITY", 0.70, 3, theme="BANK_FINANCE"),
        _row("AAA2.KS", "Alpha clone", "EQUITY", 0.85, 1, theme="SEMICONDUCTOR"),
        _row("GOLD.KS", "Gold", "REAL_ASSET", 0.75, 10, corr=-0.1, theme="GOLD", role="HEDGE"),
        _row("BOND.KS", "Bond", "DEFENSIVE", 0.65, 11, corr=0.1, vol=0.08, theme="BOND", role="HEDGE"),
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


def test_integrated_allocation_sums_to_one_and_uses_selected_satellites():
    config = DynamicSelectionConfig(
        aggressive_slots=2,
        defensive_slots=2,
    )
    scored = _scored()
    _, active = select_dynamic_universe(
        scored,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )
    allocation = build_dynamic_target_allocation(
        scored,
        active,
        config=config,
    )

    weights = allocation["ideal_target_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert "069500.KS" in weights
    assert "AAA.KS" in weights
    assert "BBB.KS" in weights
    assert allocation["stock_target"] <= 0.90


def test_peak_lock_caps_stock_target():
    config = DynamicSelectionConfig(
        aggressive_slots=1,
        defensive_slots=1,
    )
    scored = _scored()
    scored.loc[
        scored["ticker"] == "069500.KS",
        ["peak_drawdown_60d", "return_20d"],
    ] = [-0.08, -0.02]

    _, active = select_dynamic_universe(
        scored,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )
    allocation = build_dynamic_target_allocation(
        scored,
        active,
        config=config,
    )

    assert allocation["peak_lock"]
    assert allocation["stock_target"] <= 0.50
    assert abs(sum(allocation["ideal_target_weights"].values()) - 1.0) < 1e-12


def test_new_candidate_in_emergency_state_is_not_entered():
    config = DynamicSelectionConfig(
        aggressive_slots=1,
        defensive_slots=0,
    )
    scored = _scored(shocked_a=True)

    _, active = select_dynamic_universe(
        scored,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    assert active["aggressive"] == ["BBB.KS"]
    assert "AAA.KS" not in active["aggressive"]


def test_same_theme_is_limited_even_when_correlation_clusters_differ():
    scored = _scored()
    extra = _row(
        "SEMIX.KS",
        "Another Semiconductor",
        "EQUITY",
        0.88,
        99,
        theme="SEMICONDUCTOR",
    )
    scored = pd.concat([scored, pd.DataFrame([extra])], ignore_index=True)
    config = DynamicSelectionConfig(aggressive_slots=3, defensive_slots=0)

    _, active = select_dynamic_universe(
        scored,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    semis = [
        ticker
        for ticker in active["aggressive"]
        if scored.set_index("ticker").loc[ticker, "theme"] == "SEMICONDUCTOR"
    ]
    assert len(semis) == 1


def test_small_capital_blocks_first_share_with_excessive_overweight():
    scored = _scored()
    scored.loc[scored["ticker"] == "AAA.KS", "price_listing"] = 61_290.0
    config = DynamicSelectionConfig(
        aggressive_slots=1,
        defensive_slots=0,
        execution_capital=300_000.0,
        aggressive_slot_target_weight=0.15,
        max_initial_overweight_pp=0.04,
    )

    _, active = select_dynamic_universe(
        scored,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    assert active["aggressive"] == ["BBB.KS"]


def test_expired_cooldown_still_requires_recovery_before_reentry():
    scored = _scored()
    scored.loc[scored["ticker"] == "AAA.KS", ["momentum_score", "return_5d"]] = [-0.1, -0.01]
    previous = {
        "version": 2,
        "last_run_date": "2026-09-10",
        "last_selection_month": "2026-09",
        "aggressive": [],
        "defensive": [],
        "cooldowns": {"AAA.KS": "2026-09-15"},
        "risk_state": "NORMAL",
        "risk_state_counter": 0,
    }
    config = DynamicSelectionConfig(aggressive_slots=1, defensive_slots=0)

    state, active = select_dynamic_universe(
        scored,
        previous_state=previous,
        as_of=pd.Timestamp("2026-09-22"),
        config=config,
    )

    assert active["aggressive"] == ["BBB.KS"]
    assert "AAA.KS" in state["cooldowns"]


def test_deep_drawdown_old_state_enters_recovery_instead_of_full_risk():
    scored = _scored()
    scored.loc[
        scored["ticker"] == "069500.KS",
        ["peak_drawdown_60d", "return_20d", "momentum_score"],
    ] = [-0.18, 0.05, 0.8]
    _, active = select_dynamic_universe(
        scored,
        as_of=pd.Timestamp("2026-09-22"),
        config=DynamicSelectionConfig(aggressive_slots=2, defensive_slots=1),
    )
    allocation = build_dynamic_target_allocation(
        scored,
        active,
        config=DynamicSelectionConfig(aggressive_slots=2, defensive_slots=1),
        previous_state={"version": 1},
    )

    assert allocation["risk_state"] == "RECOVERY_2"
    assert allocation["stock_target"] <= 0.65
