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

    emergency_return_1d: float = -0.05
    emergency_return_5d: float = -0.08
    emergency_peak_drawdown_60d: float = -0.10
    emergency_momentum_floor: float = 0.0

    market_emergency_return_5d: float = -0.06
    market_emergency_return_20d: float = -0.08

    defensive_market_corr_max: float = 0.60
    defensive_min_annual_vol: float = 0.01

    market_ticker: str = "069500.KS"
    safe_ticker: str = "153130.KS"


def empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        "last_run_date": None,
        "last_selection_month": None,
        "aggressive": [],
        "defensive": [],
        "cooldowns": {},
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


def _cooldown_active(
    ticker: str,
    cooldowns: dict[str, str],
    as_of: pd.Timestamp,
) -> bool:
    until = cooldowns.get(ticker)
    if not until:
        return False
    return as_of.normalize() <= pd.Timestamp(until)


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

    if np.isfinite(r1) and r1 <= config.emergency_return_1d:
        return f"1D_DROP_{r1:.1%}"
    if np.isfinite(r5) and r5 <= config.emergency_return_5d:
        return f"5D_DROP_{r5:.1%}"
    if (
        np.isfinite(peak_dd)
        and peak_dd <= config.emergency_peak_drawdown_60d
        and np.isfinite(momentum)
        and momentum < config.emergency_momentum_floor
    ):
        return f"PEAK_BREAK_{peak_dd:.1%}_MOM_{momentum:.2f}"
    return None


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
        pool = pool.loc[
            (pool["ticker"] != config.safe_ticker)
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
        "last_score": _finite(row.get("screen_score"), -999.0),
    }


def _refresh_holding_metadata(
    holding: dict[str, Any],
    row: pd.Series | None,
) -> dict[str, Any]:
    updated = dict(holding)
    if row is not None:
        updated["name"] = str(row.get("name", updated.get("name", updated["ticker"])))
        updated["cluster_id"] = int(row.get("cluster_id", updated.get("cluster_id", -1)))
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
) -> list[pd.Series]:
    held_tickers = {str(item["ticker"]) for item in holdings}
    held_clusters = {
        int(item.get("cluster_id", -1))
        for item in holdings
        if int(item.get("cluster_id", -1)) >= 0
    }
    candidates: list[pd.Series] = []
    for _, row in pool.iterrows():
        ticker = str(row["ticker"])
        cluster_id = int(row.get("cluster_id", -1))
        if ticker in held_tickers:
            continue
        if cluster_id >= 0 and cluster_id in held_clusters:
            continue
        if _cooldown_active(ticker, cooldowns, as_of):
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
    events: list[dict[str, Any]],
    bucket: str,
) -> list[dict[str, Any]]:
    while len(holdings) < slots:
        candidates = _eligible_candidates(
            pool,
            holdings=holdings,
            cooldowns=cooldowns,
            as_of=as_of,
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
                "margin": challenger_score - incumbent_score,
            }
        )
    return holdings


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

    # Remove expired cooldowns so state remains compact.
    cooldowns = {
        ticker: until
        for ticker, until in cooldowns.items()
        if as_of.normalize() <= pd.Timestamp(until)
    }

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
            events=events,
            bucket="aggressive",
        )
    defensive = _fill_empty_slots(
        defensive,
        pool=defensive_pool,
        slots=config.defensive_slots,
        cooldowns=cooldowns,
        as_of=as_of,
        events=events,
        bucket="defensive",
    )

    new_state = {
        "version": 1,
        "last_run_date": as_of.date().isoformat(),
        "last_selection_month": month_key,
        "aggressive": aggressive,
        "defensive": defensive,
        "cooldowns": cooldowns,
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


def build_dynamic_target_allocation(
    scored: pd.DataFrame,
    active: dict[str, Any],
    *,
    config: DynamicSelectionConfig | None = None,
) -> dict[str, Any]:
    """Combine dynamic security selection with the existing risk-allocation logic.

    This is an ideal research allocation, not a brokerage execution state.
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
    stock_target = float(
        np.interp(market_score, stock_score_points, stock_target_points)
    )
    stock_target = float(np.clip(stock_target, 0.0, 0.90))

    peak_dd = _finite(market.get("peak_drawdown_60d"), 0.0)
    short_return = _finite(market.get("return_20d"), 0.0)
    peak_lock = False
    if peak_dd <= -0.04 and (short_return <= 0.0 or breadth <= 0.0):
        peak_lock = True
        if peak_dd <= -0.10:
            stock_target = min(stock_target, 0.35)
        elif peak_dd <= -0.08:
            stock_target = min(stock_target, 0.50)
        elif peak_dd <= -0.06:
            stock_target = min(stock_target, 0.65)
        else:
            stock_target = min(stock_target, 0.80)

    market_emergency = bool(active.get("market_emergency"))
    if market_emergency:
        stock_target = min(stock_target, 0.35)

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
    best_defensive = max(
        [_finite(row.get("momentum_score"), 0.0) for row in positive_defensive],
        default=0.0,
    )

    residual = max(0.0, 1.0 - stock_target)
    defensive_strength = float(np.clip(best_defensive / 0.30, 0.0, 1.0))
    defensive_target = residual * 0.90 * defensive_strength
    safe_target = 1.0 - stock_target - defensive_target

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

    if defensive_target > 0 and positive_defensive:
        ranked = sorted(
            positive_defensive,
            key=lambda row: _finite(row.get("momentum_score"), 0.0),
            reverse=True,
        )[:2]
        scores = np.asarray(
            [_finite(row.get("momentum_score"), 0.0) for row in ranked],
            dtype=float,
        )
        if scores.sum() > 0:
            raw = defensive_target * scores / scores.sum()
            raw = np.minimum(raw, 0.35)
            assigned = float(raw.sum())
            leftover = max(0.0, defensive_target - assigned)
            if leftover > 1e-12:
                room = np.maximum(0.0, 0.35 - raw)
                if room.sum() > 0:
                    extra = np.minimum(leftover * room / room.sum(), room)
                    raw += extra
            for row, weight in zip(ranked, raw):
                if weight > 1e-12:
                    weights[str(row["ticker"])] = float(weight)

    assigned_total = float(sum(weights.values()))
    weights[config.safe_ticker] = max(0.0, 1.0 - assigned_total)

    return {
        "market_score": market_score,
        "market_momentum": market_momentum,
        "breadth": breadth,
        "peak_drawdown_60d": peak_dd,
        "return_20d": short_return,
        "peak_lock": peak_lock,
        "market_emergency": market_emergency,
        "stock_target": stock_target,
        "defensive_target": defensive_target,
        "safe_target": safe_target,
        "ideal_target_weights": weights,
    }
