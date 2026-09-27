from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .costs import TOSS_KRX_COMMISSION_BPS, toss_krx_commissions_by_ticker
from .metrics import performance_metrics


@dataclass
class IntegerReplayResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    shares: pd.DataFrame
    portfolio: pd.DataFrame
    trades: pd.DataFrame
    cashflows: pd.DataFrame
    metrics: dict[str, float]


def _allocate_whole_shares(
    capital: float,
    target_weights: pd.Series,
    prices: pd.Series,
) -> tuple[pd.Series, float]:
    target = target_weights.reindex(prices.index).fillna(0.0).clip(lower=0.0)
    px = prices.astype(float)
    valid = (px > 0) & np.isfinite(px)
    target = target.where(valid, 0.0)

    shares = np.floor((capital * target) / px).fillna(0.0).astype(int)

    def values_and_cash(candidate: pd.Series) -> tuple[pd.Series, float]:
        values = candidate.astype(float) * px
        cash = float(capital - values.sum())
        return values, cash

    def objective(candidate: pd.Series) -> float:
        values, cash = values_and_cash(candidate)
        actual = values / capital if capital > 0 else values * 0.0
        return float(((actual - target) ** 2).sum() + (cash / capital) ** 2)

    while True:
        _, cash = values_and_cash(shares)
        base = objective(shares)
        best_ticker = None
        best = base
        for ticker in shares.index:
            price = float(px[ticker])
            if price > cash + 1e-9:
                continue
            shares[ticker] += 1
            trial = objective(shares)
            shares[ticker] -= 1
            if trial + 1e-12 < best:
                best = trial
                best_ticker = ticker
        if best_ticker is None:
            break
        shares[best_ticker] += 1

    values, cash = values_and_cash(shares)
    return shares, cash


def replay_fractional_targets_as_whole_shares(
    prices: pd.DataFrame,
    fractional: BacktestResult,
    *,
    initial_capital: float,
    transaction_cost_bps: float = TOSS_KRX_COMMISSION_BPS,
    dividends: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
) -> IntegerReplayResult:
    """Replay an existing strategy's target changes using whole ETF shares.

    Strategy decisions remain identical to the fractional-weight backtest. Only
    execution is changed: target weights are approximated with integer shares
    and residual capital remains cash.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")

    prices = prices.copy().sort_index().dropna(how="any")
    columns = prices.columns
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    if dividends is None:
        dividends = pd.DataFrame(0.0, index=prices.index, columns=columns)
    else:
        dividends = (
            dividends.copy()
            .reindex(index=prices.index, columns=columns)
            .fillna(0.0)
            .astype(float)
        )
    if splits is None:
        splits = pd.DataFrame(0.0, index=prices.index, columns=columns)
    else:
        splits = (
            splits.copy()
            .reindex(index=prices.index, columns=columns)
            .fillna(0.0)
            .astype(float)
        )

    initial_target = (
        fractional.weights.iloc[0]
        .drop(labels=["CASH"], errors="ignore")
        .reindex(columns)
        .fillna(0.0)
    )
    shares, cash = _allocate_whole_shares(
        initial_capital,
        initial_target,
        prices.iloc[0],
    )

    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    if not fractional.trades.empty:
        for date, row in fractional.trades.iterrows():
            target = pd.Series(
                {
                    ticker: float(row.get(f"target_{ticker}", 0.0))
                    for ticker in columns
                },
                dtype=float,
            )
            target_by_date[pd.Timestamp(date)] = target

    equity_records: list[tuple[pd.Timestamp, float]] = []
    return_records: list[tuple[pd.Timestamp, float]] = []
    share_records: list[dict[str, float | int | pd.Timestamp]] = []
    portfolio_records: list[dict[str, float | int | pd.Timestamp]] = []
    trade_records: list[dict[str, float | int | pd.Timestamp | str]] = []
    rebalance_records: list[dict[str, float | pd.Timestamp]] = []
    cashflow_records: list[dict[str, float | pd.Timestamp | str]] = []
    total_distributions = 0.0

    first_equity = float((shares.astype(float) * prices.iloc[0]).sum() + cash)
    equity_records.append((prices.index[0], first_equity))
    return_records.append((prices.index[0], 0.0))
    share_records.append(
        {"date": prices.index[0], **shares.to_dict(), "CASH": cash}
    )
    first_values = shares.astype(float) * prices.iloc[0].astype(float)
    first_portfolio = {"date": prices.index[0], "equity": first_equity}
    for ticker in columns:
        first_portfolio[f"shares_{ticker}"] = int(shares[ticker])
        first_portfolio[f"value_{ticker}"] = float(first_values[ticker])
        first_portfolio[f"weight_{ticker}"] = (
            float(first_values[ticker] / first_equity) if first_equity > 0 else 0.0
        )
    first_portfolio["CASH"] = cash
    first_portfolio["weight_CASH"] = cash / first_equity if first_equity > 0 else 0.0
    portfolio_records.append(first_portfolio)
    prev_equity = first_equity

    for date, px in prices.iloc[1:].iterrows():
        split_row = splits.loc[date]
        for ticker in columns:
            factor = float(split_row.get(ticker, 0.0))
            if np.isfinite(factor) and factor > 0 and abs(factor - 1.0) > 1e-12:
                before = int(shares[ticker])
                shares[ticker] = int(round(before * factor))
                cashflow_records.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "type": "SPLIT",
                        "amount": factor,
                    }
                )

        dividend_row = dividends.loc[date]
        distribution_cash = float(
            (
                shares.astype(float)
                * dividend_row.reindex(columns).fillna(0.0).astype(float)
            ).sum()
        )
        if distribution_cash > 0:
            cash += distribution_cash
            total_distributions += distribution_cash
            for ticker in columns:
                per_share = float(dividend_row.get(ticker, 0.0))
                if per_share <= 0 or int(shares[ticker]) <= 0:
                    continue
                cashflow_records.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "type": "DISTRIBUTION",
                        "amount": float(int(shares[ticker]) * per_share),
                    }
                )

        equity_before_trade = float((shares.astype(float) * px).sum() + cash)

        target = target_by_date.get(pd.Timestamp(date))
        if target is not None:
            desired, desired_cash = _allocate_whole_shares(
                equity_before_trade,
                target,
                px,
            )
            delta = desired - shares

            current_values = shares.astype(float) * px
            desired_values = desired.astype(float) * px
            current_weights = current_values / equity_before_trade
            desired_weights = desired_values / equity_before_trade
            current_cash_weight = cash / equity_before_trade
            desired_cash_weight = desired_cash / equity_before_trade
            turnover = 0.5 * (
                float((desired_weights - current_weights).abs().sum())
                + abs(desired_cash_weight - current_cash_weight)
            )

            def execution_costs(
                proposed: pd.Series,
            ) -> tuple[pd.Series, pd.Series, float, float, float]:
                proposed_delta = proposed - shares
                notionals = proposed_delta.abs().astype(float) * px.astype(float)
                commissions = toss_krx_commissions_by_ticker(
                    notionals,
                    commission_bps=transaction_cost_bps,
                )
                buy_notional = float(notionals.loc[proposed_delta > 0].sum())
                sell_notional = float(notionals.loc[proposed_delta < 0].sum())
                return (
                    proposed_delta,
                    commissions,
                    float(commissions.sum()),
                    buy_notional,
                    sell_notional,
                )

            delta, commissions, cost, buy_notional, sell_notional = execution_costs(
                desired
            )
            post_cash = desired_cash - cost

            # If commissions push cash slightly negative, remove purchases
            # until the account remains fully funded.
            while post_cash < -1e-9:
                bought = delta[delta > 0]
                if bought.empty:
                    break
                ticker = max(
                    bought.index,
                    key=lambda t: float(px[t]),
                )
                desired[ticker] -= 1
                desired_values = desired.astype(float) * px
                desired_cash = equity_before_trade - float(desired_values.sum())
                desired_weights = desired_values / equity_before_trade
                desired_cash_weight = desired_cash / equity_before_trade
                turnover = 0.5 * (
                    float((desired_weights - current_weights).abs().sum())
                    + abs(desired_cash_weight - current_cash_weight)
                )
                (
                    delta,
                    commissions,
                    cost,
                    buy_notional,
                    sell_notional,
                ) = execution_costs(desired)
                post_cash = desired_cash - cost

            if (delta != 0).any():
                rebalance_records.append(
                    {
                        "date": date,
                        "turnover": turnover,
                        "cost": cost,
                        "buy_notional": buy_notional,
                        "sell_notional": sell_notional,
                    }
                )
                for ticker in columns:
                    change = int(delta[ticker])
                    if change == 0:
                        continue
                    trade_records.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "action": "BUY" if change > 0 else "SELL",
                            "shares": abs(change),
                            "price": int(round(float(px[ticker]))),
                            "notional": int(round(abs(change) * float(px[ticker]))),
                            "commission": int(commissions[ticker]),
                            "turnover_event": turnover,
                            "cost_event": cost,
                        }
                    )
            shares = desired
            cash = max(0.0, float(post_cash))

        equity = float((shares.astype(float) * px).sum() + cash)
        net_return = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        equity_records.append((date, equity))
        return_records.append((date, net_return))
        share_records.append({"date": date, **shares.to_dict(), "CASH": cash})
        values = shares.astype(float) * px.astype(float)
        portfolio_record = {"date": date, "equity": equity}
        for ticker in columns:
            portfolio_record[f"shares_{ticker}"] = int(shares[ticker])
            portfolio_record[f"value_{ticker}"] = float(values[ticker])
            portfolio_record[f"weight_{ticker}"] = (
                float(values[ticker] / equity) if equity > 0 else 0.0
            )
        portfolio_record["CASH"] = cash
        portfolio_record["weight_CASH"] = cash / equity if equity > 0 else 0.0
        portfolio_records.append(portfolio_record)
        prev_equity = equity

    equity_curve = pd.Series(
        [value for _, value in equity_records],
        index=pd.DatetimeIndex([date for date, _ in equity_records]),
        name="equity",
        dtype=float,
    )
    daily_returns = pd.Series(
        [value for _, value in return_records],
        index=pd.DatetimeIndex([date for date, _ in return_records]),
        name="return",
        dtype=float,
    )
    shares_df = pd.DataFrame(share_records).set_index("date")
    portfolio_df = pd.DataFrame(portfolio_records).set_index("date")
    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df = trades_df.set_index("date")

    rebalance_df = pd.DataFrame(rebalance_records)
    if not rebalance_df.empty:
        rebalance_df = rebalance_df.set_index("date")
    cashflows_df = pd.DataFrame(cashflow_records)
    if not cashflows_df.empty:
        cashflows_df = cashflows_df.set_index("date")
    metrics = performance_metrics(
        equity_curve,
        daily_returns,
        rebalance_df,
        risk_free_rate=0.0,
    )
    metrics["ending_value"] = float(equity_curve.iloc[-1])
    metrics["residual_cash"] = float(cash)
    metrics["total_distributions"] = float(total_distributions)
    return IntegerReplayResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        shares=shares_df,
        portfolio=portfolio_df,
        trades=trades_df,
        cashflows=cashflows_df,
        metrics=metrics,
    )
