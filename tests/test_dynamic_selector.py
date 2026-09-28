import pandas as pd

from rebalance_backtest.dynamic_selector import (
    DynamicSelectionConfig,
    build_dynamic_target_allocation,
    reconcile_active_universe_for_execution,
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


def test_post_risk_reconciliation_removes_zero_share_philadelphia_semiconductor():
    scored = pd.DataFrame(
        [
            _row(
                "069500.KS",
                "KODEX 200",
                "EQUITY",
                0.60,
                0,
                momentum=0.59,
                r5=0.01,
                r20=0.03,
                peak=-0.16,
                theme="BROAD_MARKET",
                price=111_400.0,
            ),
            _row(
                "091170.KS",
                "KODEX 은행",
                "EQUITY",
                0.90,
                1,
                momentum=0.88,
                theme="BANK_FINANCE",
                price=16_485.0,
            ),
            _row(
                "381180.KS",
                "TIGER 미국필라델피아반도체나스닥",
                "EQUITY",
                0.85,
                2,
                momentum=0.81,
                theme="SEMICONDUCTOR",
                price=44_685.0,
            ),
            _row(
                "458730.KS",
                "TIGER 미국배당다우존스",
                "EQUITY",
                0.80,
                3,
                momentum=0.48,
                theme="DIVIDEND",
                price=14_375.0,
            ),
            _row(
                "CHEAP.KS",
                "Cheap Healthcare",
                "EQUITY",
                0.70,
                4,
                momentum=0.40,
                theme="BIO_HEALTHCARE",
                price=12_000.0,
            ),
            _row(
                "153130.KS",
                "KODEX 단기채권",
                "DEFENSIVE",
                0.20,
                20,
                momentum=0.0,
                corr=0.0,
                vol=0.02,
                theme="BOND",
                role="SAFE",
                price=113_207.0,
            ),
        ]
    )
    config = DynamicSelectionConfig(
        aggressive_slots=3,
        defensive_slots=0,
        execution_capital=300_000.0,
        max_initial_overweight_pp=0.04,
    )
    previous = {
        "version": 2,
        "last_run_date": "2026-09-27",
        "last_selection_month": "2026-08",
        "aggressive": [],
        "defensive": [],
        "cooldowns": {},
        "risk_state": "RECOVERY_2",
        "risk_state_counter": 0,
    }
    as_of = pd.Timestamp("2026-09-28")

    state, active = select_dynamic_universe(
        scored,
        previous_state=previous,
        as_of=as_of,
        config=config,
    )
    assert active["aggressive"] == [
        "091170.KS",
        "381180.KS",
        "458730.KS",
    ]

    allocation = build_dynamic_target_allocation(
        scored,
        active,
        config=config,
        previous_state=previous,
    )
    assert allocation["risk_state"] == "RECOVERY_2"
    assert abs(allocation["ideal_target_weights"]["381180.KS"] - (0.325 / 3)) < 1e-12

    state, active, allocation = reconcile_active_universe_for_execution(
        scored,
        state,
        active,
        allocation,
        as_of=as_of,
        config=config,
        previous_state=previous,
    )

    assert "381180.KS" not in active["aggressive"]
    assert "CHEAP.KS" in active["aggressive"]
    assert any(
        event["action"] == "EXECUTION_CONSTRAINT_EXIT"
        and event["ticker"] == "381180.KS"
        for event in active["events"]
    )
    assert any(
        event["action"] == "EXECUTION_CONSTRAINT_REPLACE"
        and event["ticker"] == "CHEAP.KS"
        for event in active["events"]
    )

    from rebalance_backtest.execution_plan import build_execution_plan

    prices = dict(
        zip(scored["ticker"], scored["price_listing"])
    )
    names = dict(zip(scored["ticker"], scored["name"]))
    plan = build_execution_plan(
        allocation["ideal_target_weights"],
        prices,
        names,
        300_000.0,
        max_overweight_pp=0.04,
    ).positions.set_index("ticker")

    for ticker in active["aggressive"]:
        assert int(plan.loc[ticker, "shares"]) >= 1


def test_execution_reconciliation_allows_fewer_slots_when_no_replacement_works():
    scored = pd.DataFrame(
        [
            _row(
                "069500.KS",
                "KODEX 200",
                "EQUITY",
                0.60,
                0,
                momentum=0.59,
                r20=0.03,
                peak=-0.16,
                theme="BROAD_MARKET",
                price=111_400.0,
            ),
            _row(
                "091170.KS",
                "KODEX 은행",
                "EQUITY",
                0.90,
                1,
                theme="BANK_FINANCE",
                price=16_485.0,
            ),
            _row(
                "381180.KS",
                "TIGER 미국필라델피아반도체나스닥",
                "EQUITY",
                0.85,
                2,
                theme="SEMICONDUCTOR",
                price=44_685.0,
            ),
            _row(
                "153130.KS",
                "KODEX 단기채권",
                "DEFENSIVE",
                0.20,
                20,
                momentum=0.0,
                corr=0.0,
                vol=0.02,
                theme="BOND",
                role="SAFE",
                price=113_207.0,
            ),
        ]
    )
    config = DynamicSelectionConfig(
        aggressive_slots=2,
        defensive_slots=0,
        execution_capital=300_000.0,
    )
    previous = {
        "version": 2,
        "last_selection_month": "2026-08",
        "aggressive": [],
        "defensive": [],
        "cooldowns": {},
        "risk_state": "RECOVERY_1",
        "risk_state_counter": 0,
    }
    as_of = pd.Timestamp("2026-09-28")
    state, active = select_dynamic_universe(
        scored,
        previous_state=previous,
        as_of=as_of,
        config=config,
    )
    allocation = build_dynamic_target_allocation(
        scored,
        active,
        config=config,
        previous_state=previous,
    )
    state, active, allocation = reconcile_active_universe_for_execution(
        scored,
        state,
        active,
        allocation,
        as_of=as_of,
        config=config,
        previous_state=previous,
    )

    assert len(active["aggressive"]) <= config.aggressive_slots
    assert "381180.KS" not in active["aggressive"]
