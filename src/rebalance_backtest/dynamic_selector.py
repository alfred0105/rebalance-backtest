from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicSelectionConfig:
    aggressive_slots: int = 3
    defensive_slots: int = 2
    min_hold_business_days: int = 20
    cooldown_business_days: int = 5
    replacement_score_margin: float = 0.15

    # Emergency logic: absolute floors plus volatility-aware thresholds.
    emergency_return_1d: float = -0.05
    emergency_return_5d: float = -0.08
    emergency_peak_drawdown_60d: float = -0.10
    emergency_momentum_floor: float = 0.0
    emergency_vol_sigma_1d: float = 3.0
    emergency_vol_sigma_5d: float = 2.5
    catastrophic_return_1d: float = -0.10
    catastrophic_return_5d: float = -0.15

    # A cooldown is a minimum wait. Re-entry also requires recovery.
    reentry_min_momentum: float = 0.0
    reentry_min_return_5d: float = 0.0

    market_emergency_return_5d: float = -0.06
    market_emergency_return_20d: float = -0.08

    defensive_market_corr_max: float = 0.60
    defensive_min_annual_vol: float = 0.01

    # Concentration and small-capital implementation constraints.
    max_same_theme: int = 1
    execution_capital: float = 300_000.0
    aggressive_slot_target_weight: float = 0.15
    defensive_slot_target_weight: float = 0.05
    max_initial_overweight_pp: float = 0.04

    # Risk-state recovery after a peak lock / market emergency.
    recovery_stage_days: int = 3
    recovery_1_cap: float = 0.50
    recovery_2_cap: float = 0.65
    recovery_3_cap: float = 0.80

    # Defensive sleeve roles.
    safe_min_weight: float = 0.03
    diversifier_max_total: float = 0.05
    hedge_max_total: float = 0.15
    defensive_single_max: float = 0.35

    market_ticker: str = "069500.KS"
    safe_ticker: str = "153130.KS"


def empty_state() -> dict[str, Any]:
    return {
        "version": 2,
        "last_run_date": None,
        "last_selection_month": None,
        "aggressive": [],
        "defensive": [],
        "cooldowns": {},
        "risk_state": "NORMAL",
        "risk_state_counter": 0,
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return empty_state()
    base = empty_state()
    base.update(state)
    base["aggressive"] = list(base.get("aggressive") or [])
    base["defensive"] = list(base.get("defensive") or [])
    base["cooldowns"] = dict(base.get("cooldowns") or {})
    base["risk_state"] = str(base.get("risk_state") or "NORMAL")
    base["risk_state_counter"] = int(base.get("risk_state_counter") or 0)
    return base


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _business_days_held(entered: str, as_of: pd.Timestamp) -> int:
    start = np.datetime64(pd.Timestamp(entered).date(), "D")
    end = np.datetime64(as_of.date(), "D")
    if end <= start:
        return 0
    return int(np.busday_count(start, end))


def _cooldown_until(as_of: pd.Timestamp, business_days: int) -> str:
    return (as_of + pd.offsets.BDay(business_days)).date().isoformat()


def _row_map(scored: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        str(row["ticker"]): row
        for _, row in scored.iterrows()
    }


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _market_emergency(
    rows: dict[str, pd.Series],
    config: DynamicSelectionConfig,
) -> tuple[bool, str | None]:
    row = rows.get(config.market_ticker)
    if row is None:
        return False, None

    r5 = _finite(row.get("return_5d"))
    r20 = _finite(row.get("return_20d"))
    if np.isfinite(r5) and r5 <= config.market_emergency_return_5d:
        return True, f"MARKET_5D_{r5:.1%}"
    if np.isfinite(r20) and r20 <= config.market_emergency_return_20d:
        return True, f"MARKET_20D_{r20:.1%}"
    return False, None


def _adaptive_emergency_thresholds(
    row: pd.Series,
    config: DynamicSelectionConfig,
) -> tuple[float, float]:
    annual_vol = _finite(row.get("annual_vol_63d"))
    if not np.isfinite(annual_vol) or annual_vol <= 0:
        return config.emergency_return_1d, config.emergency_return_5d

    daily_vol = annual_vol / np.sqrt(252.0)
    one_day = -max(
        abs(config.emergency_return_1d),
        config.emergency_vol_sigma_1d * daily_vol,
    )
    five_day = -max(
        abs(config.emergency_return_5d),
        config.emergency_vol_sigma_5d * daily_vol * np.sqrt(5.0),
    )
    return float(one_day), float(five_day)


def _individual_emergency_reason(
    row: pd.Series | None,
    config: DynamicSelectionConfig,
) -> str | None:
    if row is None:
        return None

    r1 = _finite(row.get("return_1d"))
    r5 = _finite(row.get("return_5d"))
    peak_dd = _finite(row.get("peak_drawdown_60d"))
    momentum = _finite(row.get("momentum_score"))

    if np.isfinite(r1) and r1 <= config.catastrophic_return_1d:
        return f"1D_HARD_STOP_{r1:.1%}"
    if np.isfinite(r5) and r5 <= config.catastrophic_return_5d:
        return f"5D_HARD_STOP_{r5:.1%}"

    threshold_1d, threshold_5d = _adaptive_emergency_thresholds(row, config)
    if np.isfinite(r1) and r1 <= threshold_1d:
        return f"1D_VOL_SHOCK_{r1:.1%}_TH_{threshold_1d:.1%}"
    if np.isfinite(r5) and r5 <= threshold_5d:
        return f"5D_VOL_SHOCK_{r5:.1%}_TH_{threshold_5d:.1%}"
    if (
        np.isfinite(peak_dd)
        and peak_dd <= config.emergency_peak_drawdown_60d
        and np.isfinite(momentum)
        and momentum < config.emergency_momentum_floor
    ):
        return f"PEAK_BREAK_{peak_dd:.1%}_MOM_{momentum:.2f}"
    return None


def _recovery_confirmed(
    row: pd.Series | None,
    config: DynamicSelectionConfig,
) -> bool:
    if row is None:
        return False
    momentum = _finite(row.get("momentum_score"))
    r5 = _finite(row.get("return_5d"))
    if not np.isfinite(momentum) or not np.isfinite(r5):
        return False
    if momentum <= config.reentry_min_momentum:
        return False
    if r5 <= config.reentry_min_return_5d:
        return False
    return _individual_emergency_reason(row, config) is None


def _cooldown_blocked(
    ticker: str,
    row: pd.Series | None,
    cooldowns: dict[str, str],
    as_of: pd.Timestamp,
    config: DynamicSelectionConfig,
) -> bool:
    until = cooldowns.get(ticker)
    if not until:
        return False
    if as_of.normalize() <= pd.Timestamp(until):
        return True
    # Cooldown expiry is only the minimum wait. Re-entry still requires
    # positive short-term recovery.
    return not _recovery_confirmed(row, config)


def _candidate_affordable(
    row: pd.Series,
    bucket: str,
    config: DynamicSelectionConfig,
) -> bool:
    if config.execution_capital <= 0:
        return True
    price = _finite(row.get("price_listing"), 0.0)
    if price <= 0:
        return True
    target = (
        config.aggressive_slot_target_weight
        if bucket == "aggressive"
        else config.defensive_slot_target_weight
    )
    max_one_share_weight = target + config.max_initial_overweight_pp
    return price / config.execution_capital <= max_one_share_weight + 1e-12


def _deduped_pool(
    scored: pd.DataFrame,
    bucket: str,
    config: DynamicSelectionConfig,
) -> pd.DataFrame:
    if bucket == "aggressive":
        pool = scored.loc[
            (scored["bucket"] == "EQUITY")
            & (scored["ticker"] != config.market_ticker)
        ].copy()
    else:
        pool = scored.loc[
            scored["bucket"].isin(["DEFENSIVE", "REAL_ASSET"])
        ].copy()
        corr = pd.to_numeric(pool["corr_market_126d"], errors="coerce")
        vol = pd.to_numeric(pool["annual_vol_63d"], errors="coerce")
        role = (
            pool["defensive_role"].astype(str)
            if "defensive_role" in pool.columns
            else pd.Series("HEDGE", index=pool.index)
        )
        pool = pool.loc[
            (pool["ticker"] != config.safe_ticker)
            & (role != "SAFE")
            & ((corr <= config.defensive_market_corr_max) | corr.isna())
            & ((vol >= config.defensive_min_annual_vol) | vol.isna())
        ].copy()

    pool["screen_score"] = pd.to_numeric(
        pool["screen_score"], errors="coerce"
    )
    pool = pool.dropna(subset=["screen_score"]).sort_values(
        ["screen_score", "momentum_score"],
        ascending=False,
    )

    if "cluster_id" in pool.columns:
        pool = pool.drop_duplicates("cluster_id", keep="first")
    return pool.reset_index(drop=True)


def _holding_from_row(row: pd.Series, as_of: pd.Timestamp) -> dict[str, Any]:
    return {
        "ticker": str(row["ticker"]),
        "name": str(row.get("name", row["ticker"])),
        "entered": as_of.date().isoformat(),
        "cluster_id": int(row.get("cluster_id", -1)),
        "theme": str(row.get("theme", "OTHER")),
        "defensive_role": str(row.get("defensive_role", "RISK")),
        "last_score": _finite(row.get("screen_score"), -999.0),
        "price_listing": _finite(row.get("price_listing"), 0.0),
    }


def _refresh_holding_metadata(
    holding: dict[str, Any],
    row: pd.Series | None,
) -> dict[str, Any]:
    updated = dict(holding)
    if row is not None:
        updated["name"] = str(row.get("name", updated.get("name", updated["ticker"])))
        updated["cluster_id"] = int(row.get("cluster_id", updated.get("cluster_id", -1)))
        updated["theme"] = str(row.get("theme", updated.get("theme", "OTHER")))
        updated["defensive_role"] = str(
            row.get("defensive_role", updated.get("defensive_role", "RISK"))
        )
        updated["price_listing"] = _finite(
            row.get("price_listing"),
            float(updated.get("price_listing", 0.0)),
        )
        updated["last_score"] = _finite(
            row.get("screen_score"),
            float(updated.get("last_score", -999.0)),
        )
    return updated


def _exit_emergencies(
    holdings: list[dict[str, Any]],
    *,
    rows: dict[str, pd.Series],
    as_of: pd.Timestamp,
    cooldowns: dict[str, str],
    config: DynamicSelectionConfig,
    market_emergency: bool,
    market_reason: str | None,
    aggressive: bool,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for holding in holdings:
        ticker = str(holding["ticker"])
        row = rows.get(ticker)

        reason = None
        if aggressive and market_emergency:
            reason = market_reason or "MARKET_EMERGENCY"
        else:
            reason = _individual_emergency_reason(row, config)

        if reason is None:
            kept.append(_refresh_holding_metadata(holding, row))
            continue

        cooldowns[ticker] = _cooldown_until(
            as_of,
            config.cooldown_business_days,
        )
        events.append(
            {
                "action": "EMERGENCY_EXIT",
                "ticker": ticker,
                "name": holding.get("name", ticker),
                "reason": reason,
                "cooldown_until": cooldowns[ticker],
            }
        )
    return kept


def _eligible_candidates(
    pool: pd.DataFrame,
    *,
    holdings: list[dict[str, Any]],
    cooldowns: dict[str, str],
    as_of: pd.Timestamp,
    config: DynamicSelectionConfig,
    bucket: str,
) -> list[pd.Series]:
    held_tickers = {str(item["ticker"]) for item in holdings}
    held_clusters = {
        int(item.get("cluster_id", -1))
        for item in holdings
        if int(item.get("cluster_id", -1)) >= 0
    }
    theme_counts: dict[str, int] = {}
    for item in holdings:
        theme = str(item.get("theme", "OTHER"))
        if theme not in {"OTHER", "BROAD_MARKET"}:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    candidates: list[pd.Series] = []
    for _, row in pool.iterrows():
        ticker = str(row["ticker"])
        cluster_id = int(row.get("cluster_id", -1))
        theme = str(row.get("theme", "OTHER"))

        if ticker in held_tickers:
            continue
        if cluster_id >= 0 and cluster_id in held_clusters:
            continue
        if (
            theme not in {"OTHER", "BROAD_MARKET"}
            and theme_counts.get(theme, 0) >= config.max_same_theme
        ):
            continue
        if not _candidate_affordable(row, bucket, config):
            continue
        if _cooldown_blocked(
            ticker,
            row,
            cooldowns,
            as_of,
            config,
        ):
            continue
        if _individual_emergency_reason(row, config) is not None:
            continue
        candidates.append(row)
    return candidates


def _fill_empty_slots(
    holdings: list[dict[str, Any]],
    *,
    pool: pd.DataFrame,
    slots: int,
    cooldowns: dict[str, str],
    as_of: pd.Timestamp,
    config: DynamicSelectionConfig,
    events: list[dict[str, Any]],
    bucket: str,
) -> list[dict[str, Any]]:
    while len(holdings) < slots:
        candidates = _eligible_candidates(
            pool,
            holdings=holdings,
            cooldowns=cooldowns,
            as_of=as_of,
            config=config,
            bucket=bucket,
        )
        if not candidates:
            break
        row = candidates[0]
        holding = _holding_from_row(row, as_of)
        holdings.append(holding)
        events.append(
            {
                "action": "FILL_SLOT",
                "bucket": bucket,
                "ticker": holding["ticker"],
                "name": holding["name"],
                "theme": holding["theme"],
                "defensive_role": holding["defensive_role"],
                "score": holding["last_score"],
            }
        )
    return holdings


def _monthly_replace(
    holdings: list[dict[str, Any]],
    *,
    pool: pd.DataFrame,
    cooldowns: dict[str, str],
    as_of: pd.Timestamp,
    config: DynamicSelectionConfig,
    events: list[dict[str, Any]],
    bucket: str,
) -> list[dict[str, Any]]:
    rows = _row_map(pool)
    holdings = [
        _refresh_holding_metadata(item, rows.get(str(item["ticker"])))
        for item in holdings
    ]

    while holdings:
        candidates = _eligible_candidates(
            pool,
            holdings=holdings,
            cooldowns=cooldowns,
            as_of=as_of,
            config=config,
            bucket=bucket,
        )
        if not candidates:
            break
        challenger = candidates[0]
        challenger_score = _finite(challenger.get("screen_score"), -999.0)

        replaceable = [
            item
            for item in holdings
            if _business_days_held(str(item["entered"]), as_of)
            >= config.min_hold_business_days
        ]
        if not replaceable:
            break

        incumbent = min(
            replaceable,
            key=lambda item: _finite(item.get("last_score"), -999.0),
        )
        incumbent_score = _finite(incumbent.get("last_score"), -999.0)
        incumbent_missing = str(incumbent["ticker"]) not in rows

        if (
            not incumbent_missing
            and challenger_score
            < incumbent_score + config.replacement_score_margin
        ):
            break

        holdings.remove(incumbent)
        # Re-check after removing the incumbent because its theme/cluster no
        # longer consumes a slot.
        refreshed = _eligible_candidates(
            pool,
            holdings=holdings,
            cooldowns=cooldowns,
            as_of=as_of,
            config=config,
            bucket=bucket,
        )
        if not refreshed:
            holdings.append(incumbent)
            break
        challenger = refreshed[0]
        challenger_score = _finite(challenger.get("screen_score"), -999.0)
        if (
            not incumbent_missing
            and challenger_score
            < incumbent_score + config.replacement_score_margin
        ):
            holdings.append(incumbent)
            break

        new_holding = _holding_from_row(challenger, as_of)
        holdings.append(new_holding)
        events.append(
            {
                "action": "MONTHLY_REPLACE",
                "bucket": bucket,
                "out": incumbent["ticker"],
                "out_name": incumbent.get("name", incumbent["ticker"]),
                "out_score": incumbent_score,
                "in": new_holding["ticker"],
                "in_name": new_holding["name"],
                "in_score": challenger_score,
                "theme": new_holding["theme"],
                "margin": challenger_score - incumbent_score,
            }
        )
    return holdings


def _clean_recovered_cooldowns(
    cooldowns: dict[str, str],
    rows: dict[str, pd.Series],
    as_of: pd.Timestamp,
    config: DynamicSelectionConfig,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for ticker, until in cooldowns.items():
        if as_of.normalize() <= pd.Timestamp(until):
            output[ticker] = until
            continue
        if not _recovery_confirmed(rows.get(ticker), config):
            output[ticker] = until
    return output


def select_dynamic_universe(
    scored: pd.DataFrame,
    *,
    previous_state: dict[str, Any] | None = None,
    as_of: pd.Timestamp | None = None,
    config: DynamicSelectionConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or DynamicSelectionConfig()
    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    state = empty_state()
    if previous_state:
        state.update(previous_state)
        state["aggressive"] = list(previous_state.get("aggressive") or [])
        state["defensive"] = list(previous_state.get("defensive") or [])
        state["cooldowns"] = dict(previous_state.get("cooldowns") or {})

    rows = _row_map(scored)
    events: list[dict[str, Any]] = []
    cooldowns = dict(state.get("cooldowns") or {})

    market_emergency, market_reason = _market_emergency(rows, config)

    aggressive = _exit_emergencies(
        list(state.get("aggressive") or []),
        rows=rows,
        as_of=as_of,
        cooldowns=cooldowns,
        config=config,
        market_emergency=market_emergency,
        market_reason=market_reason,
        aggressive=True,
        events=events,
    )
    defensive = _exit_emergencies(
        list(state.get("defensive") or []),
        rows=rows,
        as_of=as_of,
        cooldowns=cooldowns,
        config=config,
        market_emergency=False,
        market_reason=None,
        aggressive=False,
        events=events,
    )

    aggressive_pool = _deduped_pool(scored, "aggressive", config)
    defensive_pool = _deduped_pool(scored, "defensive", config)

    month_key = as_of.strftime("%Y-%m")
    monthly_review = state.get("last_selection_month") != month_key

    if monthly_review and not market_emergency:
        aggressive = _monthly_replace(
            aggressive,
            pool=aggressive_pool,
            cooldowns=cooldowns,
            as_of=as_of,
            config=config,
            events=events,
            bucket="aggressive",
        )
    if monthly_review:
        defensive = _monthly_replace(
            defensive,
            pool=defensive_pool,
            cooldowns=cooldowns,
            as_of=as_of,
            config=config,
            events=events,
            bucket="defensive",
        )

    if not market_emergency:
        aggressive = _fill_empty_slots(
            aggressive,
            pool=aggressive_pool,
            slots=config.aggressive_slots,
            cooldowns=cooldowns,
            as_of=as_of,
            config=config,
            events=events,
            bucket="aggressive",
        )
    defensive = _fill_empty_slots(
        defensive,
        pool=defensive_pool,
        slots=config.defensive_slots,
        cooldowns=cooldowns,
        as_of=as_of,
        config=config,
        events=events,
        bucket="defensive",
    )

    cooldowns = _clean_recovered_cooldowns(
        cooldowns,
        rows,
        as_of,
        config,
    )

    new_state = {
        "version": 2,
        "last_run_date": as_of.date().isoformat(),
        "last_selection_month": month_key,
        "aggressive": aggressive,
        "defensive": defensive,
        "cooldowns": cooldowns,
        "risk_state": str(state.get("risk_state") or "NORMAL"),
        "risk_state_counter": int(state.get("risk_state_counter") or 0),
    }

    active = {
        "as_of": as_of.date().isoformat(),
        "market_emergency": market_emergency,
        "market_emergency_reason": market_reason,
        "aggressive": [item["ticker"] for item in aggressive],
        "defensive": [item["ticker"] for item in defensive],
        "market_ticker": config.market_ticker,
        "safe_ticker": config.safe_ticker,
        "rules": asdict(config),
        "events": events,
    }
    return new_state, active


def _execution_plan_for_allocation(
    scored: pd.DataFrame,
    allocation: dict[str, Any],
    config: DynamicSelectionConfig,
):
    from .execution_plan import build_execution_plan

    rows = _row_map(scored)
    weights = {
        str(ticker): float(weight)
        for ticker, weight in allocation.get("ideal_target_weights", {}).items()
        if float(weight) > 0
    }
    prices = {
        ticker: _finite(rows[ticker].get("price_listing"), 0.0)
        for ticker in weights
        if ticker in rows
    }
    names = {
        ticker: str(rows[ticker].get("name", ticker))
        for ticker in weights
        if ticker in rows
    }
    return build_execution_plan(
        weights,
        prices,
        names,
        config.execution_capital,
        max_overweight_pp=config.max_initial_overweight_pp,
    )


def _nonexecuted_satellites(
    active: dict[str, Any],
    allocation: dict[str, Any],
    plan,
    config: DynamicSelectionConfig,
) -> list[tuple[str, str]]:
    if plan.positions.empty:
        return []
    positions = plan.positions.set_index("ticker")
    targets = allocation.get("ideal_target_weights", {})
    output: list[tuple[str, str]] = []
    for bucket in ("aggressive", "defensive"):
        for ticker in active.get(bucket, []):
            if ticker in {config.market_ticker, config.safe_ticker}:
                continue
            target = float(targets.get(ticker, 0.0))
            if target <= 0:
                output.append((bucket, ticker))
                continue
            if ticker not in positions.index:
                output.append((bucket, ticker))
                continue
            if int(positions.loc[ticker, "shares"]) <= 0:
                output.append((bucket, ticker))
    return output


def _remove_holding(
    holdings: list[dict[str, Any]],
    ticker: str,
) -> list[dict[str, Any]]:
    return [item for item in holdings if str(item.get("ticker")) != ticker]


def _candidate_trial_is_fully_executable(
    scored: pd.DataFrame,
    active: dict[str, Any],
    *,
    config: DynamicSelectionConfig,
    previous_state: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    allocation = build_dynamic_target_allocation(
        scored,
        active,
        config=config,
        previous_state=previous_state,
    )
    plan = _execution_plan_for_allocation(scored, allocation, config)
    failures = _nonexecuted_satellites(active, allocation, plan, config)
    return not failures, allocation


def reconcile_active_universe_for_execution(
    scored: pd.DataFrame,
    state: dict[str, Any],
    active: dict[str, Any],
    allocation: dict[str, Any],
    *,
    as_of: pd.Timestamp,
    config: DynamicSelectionConfig,
    previous_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Align selected satellites with what a small whole-share account can hold.

    Core market and SAFE sleeves are never removed here. Satellites that receive
    zero shares under the post-risk allocation are removed immediately. Vacant
    slots are refilled only when a trial candidate leaves *all* selected
    satellites executable. If no candidate works, the slot stays empty.
    """
    rows = _row_map(scored)
    aggressive_pool = _deduped_pool(scored, "aggressive", config)
    defensive_pool = _deduped_pool(scored, "defensive", config)
    blocked_this_run: set[str] = set()
    events = list(active.get("events") or [])

    max_passes = config.aggressive_slots + config.defensive_slots + 3
    for _ in range(max_passes):
        plan = _execution_plan_for_allocation(scored, allocation, config)
        failures = _nonexecuted_satellites(active, allocation, plan, config)
        if not failures:
            break

        removed_by_bucket: dict[str, list[str]] = {
            "aggressive": [],
            "defensive": [],
        }
        targets = allocation.get("ideal_target_weights", {})
        position_map = (
            plan.positions.set_index("ticker")
            if not plan.positions.empty
            else pd.DataFrame()
        )

        for bucket, ticker in failures:
            blocked_this_run.add(ticker)
            removed_by_bucket[bucket].append(ticker)
            holding = next(
                (
                    item
                    for item in state.get(bucket, [])
                    if str(item.get("ticker")) == ticker
                ),
                None,
            )
            target_weight = float(targets.get(ticker, 0.0))
            price = _finite(rows.get(ticker, pd.Series(dtype=float)).get("price_listing"), 0.0)
            one_share_weight = (
                price / config.execution_capital
                if config.execution_capital > 0 and price > 0
                else float("nan")
            )
            blocked_flag = False
            if (
                not position_map.empty
                and ticker in position_map.index
                and "blocked_by_size" in position_map.columns
            ):
                blocked_flag = bool(position_map.loc[ticker, "blocked_by_size"])

            state[bucket] = _remove_holding(
                list(state.get(bucket) or []),
                ticker,
            )
            active[bucket] = [
                item for item in active.get(bucket, []) if str(item) != ticker
            ]
            events.append(
                {
                    "action": "EXECUTION_CONSTRAINT_EXIT",
                    "bucket": bucket,
                    "ticker": ticker,
                    "name": (
                        holding.get("name", ticker)
                        if holding is not None
                        else str(rows.get(ticker, pd.Series(dtype=float)).get("name", ticker))
                    ),
                    "target_weight": target_weight,
                    "one_share_weight": one_share_weight,
                    "blocked_by_size": blocked_flag,
                    "reason": "ZERO_WHOLE_SHARES_AFTER_RISK_ALLOCATION",
                }
            )

        # Recompute after all zero-share exits. Fewer satellites receive larger
        # per-slot weights, which can make the remaining holdings executable.
        allocation = build_dynamic_target_allocation(
            scored,
            active,
            config=config,
            previous_state=previous_state,
        )

        for bucket in ("aggressive", "defensive"):
            slot_limit = (
                config.aggressive_slots
                if bucket == "aggressive"
                else config.defensive_slots
            )
            pool = aggressive_pool if bucket == "aggressive" else defensive_pool

            while len(active.get(bucket, [])) < slot_limit:
                filtered_pool = pool.loc[
                    ~pool["ticker"].astype(str).isin(blocked_this_run)
                ].copy()
                candidates = _eligible_candidates(
                    filtered_pool,
                    holdings=list(state.get(bucket) or []),
                    cooldowns=dict(state.get("cooldowns") or {}),
                    as_of=as_of,
                    config=config,
                    bucket=bucket,
                )
                if not candidates:
                    break

                accepted = False
                for row in candidates:
                    ticker = str(row["ticker"])
                    trial_active = {
                        **active,
                        "aggressive": list(active.get("aggressive") or []),
                        "defensive": list(active.get("defensive") or []),
                        "events": list(events),
                    }
                    trial_active[bucket].append(ticker)
                    executable, trial_allocation = _candidate_trial_is_fully_executable(
                        scored,
                        trial_active,
                        config=config,
                        previous_state=previous_state,
                    )
                    if not executable:
                        blocked_this_run.add(ticker)
                        continue

                    holding = _holding_from_row(row, as_of)
                    state[bucket] = list(state.get(bucket) or []) + [holding]
                    active[bucket] = list(active.get(bucket) or []) + [ticker]
                    allocation = trial_allocation
                    events.append(
                        {
                            "action": "EXECUTION_CONSTRAINT_REPLACE",
                            "bucket": bucket,
                            "ticker": ticker,
                            "name": holding["name"],
                            "theme": holding["theme"],
                            "defensive_role": holding["defensive_role"],
                            "score": holding["last_score"],
                            "reason": "WHOLE_SHARE_EXECUTABLE_REPLACEMENT",
                        }
                    )
                    accepted = True
                    break

                if not accepted:
                    break

        # Re-evaluate the full set after refill attempts.
        allocation = build_dynamic_target_allocation(
            scored,
            active,
            config=config,
            previous_state=previous_state,
        )

    active["events"] = events
    active["aggressive"] = [
        str(item["ticker"]) for item in state.get("aggressive", [])
    ]
    active["defensive"] = [
        str(item["ticker"]) for item in state.get("defensive", [])
    ]
    return state, active, allocation


def _peak_cap(drawdown: float) -> float:
    if drawdown <= -0.10:
        return 0.35
    if drawdown <= -0.08:
        return 0.50
    if drawdown <= -0.06:
        return 0.65
    return 0.80


def _next_risk_state(
    *,
    previous_state: dict[str, Any] | None,
    market_emergency: bool,
    peak_condition: bool,
    healthy_recovery: bool,
    deep_drawdown: bool,
    config: DynamicSelectionConfig,
) -> tuple[str, int]:
    previous_state = previous_state or {}
    previous_version = int(previous_state.get("version") or 1)
    state = str(previous_state.get("risk_state") or "NORMAL")
    counter = int(previous_state.get("risk_state_counter") or 0)

    if market_emergency:
        return "EMERGENCY", 0
    if peak_condition:
        return "PEAK_LOCK", 0

    # Migrating an old state while the market is still far below its 60d peak
    # starts cautiously rather than jumping straight to 90% equity.
    if previous_version < 2 and deep_drawdown:
        return ("RECOVERY_2", 0) if healthy_recovery else ("PEAK_LOCK", 0)

    recovery_states = {
        "EMERGENCY",
        "PEAK_LOCK",
        "RECOVERY_1",
        "RECOVERY_2",
        "RECOVERY_3",
    }
    if state not in recovery_states:
        return "NORMAL", 0

    if not healthy_recovery:
        return state, 0

    if state in {"EMERGENCY", "PEAK_LOCK"}:
        return "RECOVERY_1", 1

    counter += 1
    if counter < config.recovery_stage_days:
        return state, counter

    if state == "RECOVERY_1":
        return "RECOVERY_2", 0
    if state == "RECOVERY_2":
        return "RECOVERY_3", 0
    if state == "RECOVERY_3":
        return "NORMAL", 0
    return "NORMAL", 0


def _role_capped_defensive_weights(
    ranked: list[pd.Series],
    total_target: float,
    config: DynamicSelectionConfig,
) -> dict[str, float]:
    if not ranked or total_target <= 0:
        return {}

    scores = np.asarray(
        [max(0.0, _finite(row.get("momentum_score"), 0.0)) for row in ranked],
        dtype=float,
    )
    if scores.sum() <= 0:
        return {}

    raw = total_target * scores / scores.sum()
    weights = {
        str(row["ticker"]): min(float(weight), config.defensive_single_max)
        for row, weight in zip(ranked, raw)
    }

    def role_of(row: pd.Series) -> str:
        return str(row.get("defensive_role", "HEDGE"))

    rows_by_ticker = {str(row["ticker"]): row for row in ranked}

    for role, cap in (
        ("DIVERSIFIER", config.diversifier_max_total),
        ("HEDGE", config.hedge_max_total),
    ):
        tickers = [
            ticker
            for ticker, row in rows_by_ticker.items()
            if role_of(row) == role
        ]
        role_total = sum(weights.get(ticker, 0.0) for ticker in tickers)
        if role_total > cap and role_total > 0:
            scale = cap / role_total
            for ticker in tickers:
                weights[ticker] *= scale

    # Redistribute any leftover only into names with role/single-name room.
    for _ in range(3):
        assigned = sum(weights.values())
        leftover = total_target - assigned
        if leftover <= 1e-12:
            break

        room: dict[str, float] = {}
        role_totals: dict[str, float] = {}
        for ticker, weight in weights.items():
            role = role_of(rows_by_ticker[ticker])
            role_totals[role] = role_totals.get(role, 0.0) + weight

        for ticker, row in rows_by_ticker.items():
            role = role_of(row)
            role_cap = (
                config.diversifier_max_total
                if role == "DIVERSIFIER"
                else config.hedge_max_total
            )
            role_room = max(0.0, role_cap - role_totals.get(role, 0.0))
            single_room = max(
                0.0,
                config.defensive_single_max - weights.get(ticker, 0.0),
            )
            room[ticker] = min(role_room, single_room)

        total_room = sum(room.values())
        if total_room <= 1e-12:
            break
        for ticker, available in room.items():
            if available <= 0:
                continue
            addition = min(leftover * available / total_room, available)
            weights[ticker] = weights.get(ticker, 0.0) + addition

    return {
        ticker: float(weight)
        for ticker, weight in weights.items()
        if weight > 1e-12
    }


def build_dynamic_target_allocation(
    scored: pd.DataFrame,
    active: dict[str, Any],
    *,
    config: DynamicSelectionConfig | None = None,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the live ideal allocation for the dynamically selected universe.

    Historical validation still requires dated universes. This function is the
    forward/live allocation engine and records the risk-state machine so the
    portfolio cannot jump directly from a crash state to full risk.
    """
    config = config or DynamicSelectionConfig()
    rows = _row_map(scored)
    market = rows.get(config.market_ticker)
    if market is None:
        raise ValueError(f"Market ticker {config.market_ticker} is missing.")

    market_momentum = _finite(market.get("momentum_score"), 0.0)
    breadth_series = pd.to_numeric(
        scored.loc[
            (scored["bucket"] == "EQUITY")
            & (scored["ticker"] != config.market_ticker),
            "momentum_score",
        ],
        errors="coerce",
    ).dropna()
    breadth = float(breadth_series.median()) if len(breadth_series) else market_momentum
    market_score = 0.75 * market_momentum + 0.25 * breadth

    stock_score_points = np.asarray(
        [-0.30, -0.15, 0.00, 0.15, 0.30, 0.60],
        dtype=float,
    )
    stock_target_points = np.asarray(
        [0.00, 0.15, 0.35, 0.60, 0.75, 0.90],
        dtype=float,
    )
    raw_stock_target = float(
        np.interp(market_score, stock_score_points, stock_target_points)
    )
    raw_stock_target = float(np.clip(raw_stock_target, 0.0, 0.90))

    peak_dd = _finite(market.get("peak_drawdown_60d"), 0.0)
    short_return = _finite(market.get("return_20d"), 0.0)
    market_emergency = bool(active.get("market_emergency"))

    peak_condition = (
        peak_dd <= -0.04
        and (short_return <= 0.0 or breadth <= 0.0)
    )
    healthy_recovery = (
        short_return > 0.0
        and breadth > 0.0
        and market_score > 0.0
        and not market_emergency
    )
    risk_state, risk_state_counter = _next_risk_state(
        previous_state=previous_state,
        market_emergency=market_emergency,
        peak_condition=peak_condition,
        healthy_recovery=healthy_recovery,
        deep_drawdown=peak_dd <= -0.10,
        config=config,
    )

    stock_target = raw_stock_target
    if market_emergency or risk_state == "EMERGENCY":
        stock_target = min(stock_target, 0.35)
    elif risk_state == "PEAK_LOCK":
        stock_target = min(stock_target, _peak_cap(peak_dd))
    elif risk_state == "RECOVERY_1":
        stock_target = min(stock_target, config.recovery_1_cap)
    elif risk_state == "RECOVERY_2":
        stock_target = min(stock_target, config.recovery_2_cap)
    elif risk_state == "RECOVERY_3":
        stock_target = min(stock_target, config.recovery_3_cap)

    defensive_rows = [
        rows[ticker]
        for ticker in active.get("defensive", [])
        if ticker in rows
    ]
    positive_defensive = [
        row
        for row in defensive_rows
        if _finite(row.get("momentum_score"), -999.0) > 0
    ]
    positive_defensive = sorted(
        positive_defensive,
        key=lambda row: _finite(row.get("momentum_score"), 0.0),
        reverse=True,
    )[: config.defensive_slots]

    best_defensive = max(
        [_finite(row.get("momentum_score"), 0.0) for row in positive_defensive],
        default=0.0,
    )

    residual = max(0.0, 1.0 - stock_target)
    safe_floor = min(residual, config.safe_min_weight)
    defensive_budget = max(0.0, residual - safe_floor)
    defensive_strength = float(np.clip(best_defensive / 0.30, 0.0, 1.0))
    requested_defensive = defensive_budget * defensive_strength

    weights: dict[str, float] = {}

    aggressive_tickers = [
        ticker
        for ticker in active.get("aggressive", [])
        if ticker in rows
    ]
    if stock_target > 0:
        core = stock_target * 0.50
        satellite_total = stock_target - core
        if aggressive_tickers and satellite_total > 0:
            per_satellite = min(0.25, satellite_total / len(aggressive_tickers))
            assigned = per_satellite * len(aggressive_tickers)
            weights[config.market_ticker] = core + (satellite_total - assigned)
            for ticker in aggressive_tickers:
                weights[ticker] = per_satellite
        else:
            weights[config.market_ticker] = stock_target

    defensive_weights = _role_capped_defensive_weights(
        positive_defensive,
        requested_defensive,
        config,
    )
    weights.update(defensive_weights)

    assigned_total = float(sum(weights.values()))
    actual_safe = max(0.0, 1.0 - assigned_total)
    weights[config.safe_ticker] = actual_safe
    actual_defensive = float(sum(defensive_weights.values()))

    return {
        "market_score": market_score,
        "market_momentum": market_momentum,
        "breadth": breadth,
        "peak_drawdown_60d": peak_dd,
        "return_20d": short_return,
        "peak_lock": risk_state == "PEAK_LOCK",
        "market_emergency": market_emergency,
        "risk_state": risk_state,
        "risk_state_counter": risk_state_counter,
        "raw_stock_target": raw_stock_target,
        "stock_target": stock_target,
        "defensive_target": actual_defensive,
        "safe_target": actual_safe,
        "ideal_target_weights": weights,
    }
