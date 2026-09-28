from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import TOSS_KRX_COMMISSION_BPS, toss_krx_commissions_by_ticker


@dataclass(frozen=True)
class ExecutionPlan:
    positions: pd.DataFrame
    cash: float
    total_commission: int
    tracking_error: float


def build_execution_plan(
    weights: dict[str, float] | pd.Series,
    prices: dict[str, float] | pd.Series,
    names: dict[str, str] | None,
    capital: float,
    *,
    commission_bps: float = TOSS_KRX_COMMISSION_BPS,
    max_overweight_pp: float = 0.04,
) -> ExecutionPlan:
    """Approximate ideal weights with whole KRX ETF shares.

    A position whose first share would overshoot its ideal weight by more than
    the allowed overweight is not forced into a small account. Residual capital
    remains cash. Toss commission is charged per ticker order.
    """
    if capital <= 0:
        raise ValueError("capital must be positive")
    if commission_bps < 0:
        raise ValueError("commission_bps cannot be negative")

    target = pd.Series(weights, dtype=float).clip(lower=0.0)
    px = pd.Series(prices, dtype=float).reindex(target.index)
    names = names or {}

    valid = px.notna() & np.isfinite(px) & (px > 0) & (target > 0)
    target = target.loc[valid]
    px = px.loc[valid]
    if target.empty:
        return ExecutionPlan(pd.DataFrame(), float(capital), 0, 0.0)

    shares = np.floor(capital * target / px).astype(int)

    def overweight_limit(target_weight: float) -> float:
        # Small accounts need a wider tolerance for large core/safe sleeves:
        # one KRX ETF share can easily represent 35-40% of a 300k account.
        # Keep the strict 4%p guard for satellite sleeves, but allow up to
        # 10%p for sleeves that intentionally target at least 25%.
        if target_weight >= 0.25:
            return max(max_overweight_pp, 0.10)
        return max_overweight_pp

    blocked: set[str] = set()
    for ticker in shares.index:
        first_share_weight = float(px[ticker] / capital)
        allowed_overweight = overweight_limit(float(target[ticker]))
        if (
            shares[ticker] == 0
            and first_share_weight
            > float(target[ticker]) + allowed_overweight
        ):
            blocked.add(str(ticker))

    def state(candidate: pd.Series) -> tuple[pd.Series, pd.Series, float]:
        notionals = candidate.astype(float) * px
        commissions = toss_krx_commissions_by_ticker(
            notionals,
            commission_bps=commission_bps,
        )
        cash = capital - float(notionals.sum()) - float(commissions.sum())
        return notionals, commissions, cash

    def objective(candidate: pd.Series) -> float:
        notionals, _, cash = state(candidate)
        if cash < -1e-9:
            return float("inf")
        actual = notionals / capital
        weight_error = float(np.square(actual - target).sum())
        cash_error = (max(0.0, cash) / capital) ** 2
        return weight_error + cash_error

    while True:
        _, _, cash = state(shares)
        if cash >= -1e-9:
            break
        owned = shares[shares > 0]
        if owned.empty:
            break
        best_remove = None
        best_error = float("inf")
        for ticker in owned.index:
            shares[ticker] -= 1
            err = objective(shares)
            shares[ticker] += 1
            if err < best_error:
                best_error = err
                best_remove = ticker
        if best_remove is None:
            break
        shares[best_remove] -= 1

    while True:
        base_error = objective(shares)
        best_ticker = None
        best_error = base_error

        for ticker in shares.index:
            if str(ticker) in blocked and shares[ticker] == 0:
                continue
            trial_weight = float((shares[ticker] + 1) * px[ticker] / capital)
            allowed_overweight = overweight_limit(float(target[ticker]))
            if trial_weight > float(target[ticker]) + allowed_overweight:
                continue
            shares[ticker] += 1
            trial_error = objective(shares)
            shares[ticker] -= 1
            if trial_error + 1e-12 < best_error:
                best_error = trial_error
                best_ticker = ticker

        if best_ticker is None:
            break
        shares[best_ticker] += 1

    notionals, commissions, cash = state(shares)
    actual_weights = notionals / capital
    positions = pd.DataFrame(
        {
            "ticker": shares.index.astype(str),
            "name": [names.get(str(t), str(t)) for t in shares.index],
            "target_weight": target.reindex(shares.index).to_numpy(dtype=float),
            "target_value": (
                capital * target.reindex(shares.index)
            ).to_numpy(dtype=float),
            "price": px.reindex(shares.index).round().astype(int).to_numpy(),
            "shares": shares.to_numpy(dtype=int),
            "notional": notionals.round().astype(int).to_numpy(),
            "actual_value": notionals.round().astype(int).to_numpy(),
            "commission": commissions.to_numpy(dtype=int),
            "actual_weight": actual_weights.to_numpy(dtype=float),
        }
    )
    positions["weight_gap"] = (
        positions["actual_weight"] - positions["target_weight"]
    )
    positions["blocked_by_size"] = positions["ticker"].isin(blocked)
    positions = positions.sort_values(
        ["target_weight", "ticker"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return ExecutionPlan(
        positions=positions,
        cash=max(0.0, float(cash)),
        total_commission=int(commissions.sum()),
        tracking_error=float(objective(shares)),
    )
