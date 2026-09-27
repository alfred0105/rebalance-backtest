from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .costs import TOSS_KRX_COMMISSION_BPS, toss_krx_commissions_by_ticker
from .execution_plan import ExecutionPlan, build_execution_plan


@dataclass(frozen=True)
class PaperAccountUpdate:
    state: dict[str, Any]
    plan: ExecutionPlan
    trades: pd.DataFrame
    holdings_table: pd.DataFrame


def empty_paper_state(initial_capital: float) -> dict[str, Any]:
    return {
        "version": 1,
        "initial_capital": float(initial_capital),
        "cash": float(initial_capital),
        "holdings": {},
        "last_date": None,
        "total_commission": 0.0,
    }


def update_paper_account(
    *,
    target_weights: dict[str, float],
    prices: dict[str, float],
    names: dict[str, str],
    as_of: str,
    previous_state: dict[str, Any] | None,
    initial_capital: float,
    commission_bps: float = TOSS_KRX_COMMISSION_BPS,
    max_overweight_pp: float = 0.04,
) -> PaperAccountUpdate:
    state = empty_paper_state(initial_capital)
    if previous_state:
        state.update(previous_state)
    cash = float(state.get("cash", initial_capital))
    current = {
        str(ticker): int(shares)
        for ticker, shares in dict(state.get("holdings") or {}).items()
        if int(shares) > 0
    }

    all_tickers = set(current) | set(target_weights)
    mark_prices = {
        ticker: float(prices[ticker])
        for ticker in all_tickers
        if ticker in prices and float(prices[ticker]) > 0
    }

    # Holdings lacking a current quote stay untouched rather than being valued
    # at zero or force-sold on missing data.
    frozen = {
        ticker: shares
        for ticker, shares in current.items()
        if ticker not in mark_prices
    }
    tradable_current = {
        ticker: shares
        for ticker, shares in current.items()
        if ticker in mark_prices
    }

    frozen_value = 0.0
    equity_before = cash + sum(
        shares * mark_prices[ticker]
        for ticker, shares in tradable_current.items()
    )
    # With no current quote, preserve their last known notional if available.
    last_values = dict(state.get("last_values") or {})
    frozen_value = sum(
        float(last_values.get(ticker, 0.0))
        for ticker in frozen
    )
    equity_before += frozen_value

    executable_weights = {
        ticker: weight
        for ticker, weight in target_weights.items()
        if ticker in mark_prices
    }
    plan = build_execution_plan(
        executable_weights,
        mark_prices,
        names,
        equity_before,
        commission_bps=commission_bps,
        max_overweight_pp=max_overweight_pp,
    )

    desired = {
        str(row["ticker"]): int(row["shares"])
        for _, row in plan.positions.iterrows()
    }
    for ticker in tradable_current:
        desired.setdefault(ticker, 0)

    deltas = pd.Series(
        {
            ticker: desired.get(ticker, 0) - tradable_current.get(ticker, 0)
            for ticker in set(desired) | set(tradable_current)
        },
        dtype=int,
    )
    notionals = pd.Series(
        {
            ticker: abs(int(delta)) * mark_prices[ticker]
            for ticker, delta in deltas.items()
            if ticker in mark_prices
        },
        dtype=float,
    )
    commissions = toss_krx_commissions_by_ticker(
        notionals,
        commission_bps=commission_bps,
    )

    trade_rows: list[dict[str, Any]] = []

    # Sell first to release cash.
    for ticker, delta in deltas[deltas < 0].items():
        shares = abs(int(delta))
        notional = shares * mark_prices[ticker]
        fee = int(commissions.get(ticker, 0))
        cash += notional - fee
        trade_rows.append(
            {
                "date": as_of,
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "action": "SELL",
                "shares": shares,
                "price": int(round(mark_prices[ticker])),
                "notional": int(round(notional)),
                "commission": fee,
            }
        )

    # Then buy target shares. The zero-based planner is conservative on fees,
    # so delta execution should normally have at least as much cash available.
    executed_desired = dict(desired)
    for ticker, delta in deltas[deltas > 0].items():
        shares = int(delta)
        unit_price = mark_prices[ticker]
        notional = shares * unit_price
        fee = int(commissions.get(ticker, 0))
        required = notional + fee
        if required > cash + 1e-9:
            affordable = max(0, int(cash // unit_price))
            while affordable > 0:
                trial_notional = affordable * unit_price
                trial_fee = int(
                    toss_krx_commissions_by_ticker(
                        pd.Series({ticker: trial_notional}),
                        commission_bps=commission_bps,
                    ).iloc[0]
                )
                if trial_notional + trial_fee <= cash + 1e-9:
                    break
                affordable -= 1
            shares = affordable
            executed_desired[ticker] = tradable_current.get(ticker, 0) + shares
            notional = shares * unit_price
            fee = int(
                toss_krx_commissions_by_ticker(
                    pd.Series({ticker: notional}),
                    commission_bps=commission_bps,
                ).iloc[0]
            ) if shares > 0 else 0
            required = notional + fee
        if shares <= 0:
            continue
        cash -= required
        trade_rows.append(
            {
                "date": as_of,
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "action": "BUY",
                "shares": shares,
                "price": int(round(unit_price)),
                "notional": int(round(notional)),
                "commission": fee,
            }
        )

    final_holdings = {
        **frozen,
        **{
            ticker: int(shares)
            for ticker, shares in executed_desired.items()
            if int(shares) > 0
        },
    }
    total_fee = int(sum(int(row["commission"]) for row in trade_rows))

    holdings_rows: list[dict[str, Any]] = []
    last_values_out: dict[str, float] = {}
    marked_value = 0.0
    for ticker, shares in sorted(final_holdings.items()):
        price = mark_prices.get(ticker)
        if price is None:
            value = float(last_values.get(ticker, 0.0))
        else:
            value = float(shares * price)
        last_values_out[ticker] = value
        marked_value += value
        holdings_rows.append(
            {
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "shares": int(shares),
                "price": int(round(price)) if price is not None else None,
                "value": int(round(value)),
            }
        )

    equity_after = cash + marked_value
    for row in holdings_rows:
        row["weight"] = float(row["value"] / equity_after) if equity_after > 0 else 0.0

    new_state = {
        "version": 1,
        "initial_capital": float(state.get("initial_capital", initial_capital)),
        "cash": float(cash),
        "holdings": final_holdings,
        "last_values": last_values_out,
        "last_date": as_of,
        "total_commission": float(state.get("total_commission", 0.0)) + total_fee,
        "equity": float(equity_after),
        "total_return": (
            float(equity_after / float(state.get("initial_capital", initial_capital)) - 1.0)
            if float(state.get("initial_capital", initial_capital)) > 0
            else 0.0
        ),
    }

    return PaperAccountUpdate(
        state=new_state,
        plan=plan,
        trades=pd.DataFrame(trade_rows),
        holdings_table=pd.DataFrame(holdings_rows),
    )
