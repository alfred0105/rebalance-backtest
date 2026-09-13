from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .metrics import performance_metrics


class Strategy(Protocol):
    def initial_weights(self, columns: list[str] | pd.Index) -> pd.Series: ...

    def target_weights(
        self,
        price_history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series: ...


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def evaluation_dates(index: pd.DatetimeIndex, schedule: str | None) -> set[pd.Timestamp]:
    if schedule is None:
        return set()
    if len(index) == 0:
        return set()

    marker = pd.Series(index=index, data=np.arange(len(index)))
    try:
        periods = index.to_period(schedule)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported schedule {schedule!r}. Try 'W-FRI' or 'M'."
        ) from exc
    last_rows = marker.groupby(periods).tail(1)
    return set(pd.DatetimeIndex(last_rows.index))


def _turnover(current: pd.Series, target: pd.Series) -> float:
    current_cash = 1.0 - float(current.sum())
    target_cash = 1.0 - float(target.sum())
    traded = float((target - current).abs().sum()) + abs(target_cash - current_cash)
    return 0.5 * traded


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    *,
    schedule: str | None = "W-FRI",
    initial_capital: float = 10_000_000.0,
    transaction_cost_bps: float = 5.0,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run a close-to-close long-only backtest with one-session signal lag.

    A signal calculated with prices through date t becomes eligible for trading
    on the next available session. This avoids using the same close both to
    generate a signal and to claim an already-earned return.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")

    prices = prices.copy().sort_index().dropna(how="any")
    if len(prices) < 2:
        raise ValueError("Need at least two aligned price rows.")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    columns = prices.columns
    returns = prices.pct_change().fillna(0.0)
    eval_dates = evaluation_dates(prices.index, schedule)

    weights = strategy.initial_weights(columns).reindex(columns).fillna(0.0).astype(float)
    if (weights < 0).any() or weights.sum() > 1.0 + 1e-9:
        raise ValueError("Initial risky weights must be long-only and sum to <= 1.")

    equity = float(initial_capital)
    equity_records: list[tuple[pd.Timestamp, float]] = [(prices.index[0], equity)]
    net_return_records: list[tuple[pd.Timestamp, float]] = [(prices.index[0], 0.0)]
    weight_records: list[dict[str, float | pd.Timestamp]] = []
    trade_records: list[dict[str, float | pd.Timestamp]] = []

    first_record = {"date": prices.index[0], **weights.to_dict(), "CASH": 1.0 - weights.sum()}
    weight_records.append(first_record)

    pending_target: pd.Series | None = None

    if prices.index[0] in eval_dates:
        pending_target = strategy.target_weights(prices.iloc[:1], weights.copy())

    for i in range(1, len(prices)):
        date = prices.index[i]
        prev_equity = equity
        asset_ret = returns.iloc[i]

        gross_portfolio_ret = float((weights * asset_ret).sum())
        equity *= 1.0 + gross_portfolio_ret

        denom = 1.0 + gross_portfolio_ret
        if denom <= 0:
            raise RuntimeError("Portfolio value became non-positive.")
        weights = weights * (1.0 + asset_ret) / denom

        if pending_target is not None:
            target = pending_target.reindex(columns).fillna(0.0).astype(float)
            if (target < -1e-12).any() or target.sum() > 1.0 + 1e-9:
                raise ValueError("Target weights must be long-only and sum to <= 1.")
            target = target.clip(lower=0.0)

            turnover = _turnover(weights, target)
            cost = equity * turnover * transaction_cost_bps / 10_000.0
            equity -= cost

            if turnover > 1e-12:
                record: dict[str, float | pd.Timestamp] = {
                    "date": date,
                    "turnover": turnover,
                    "cost": cost,
                    "equity_after_cost": equity,
                }
                for ticker in columns:
                    record[f"target_{ticker}"] = float(target[ticker])
                record["target_CASH"] = 1.0 - float(target.sum())
                trade_records.append(record)
            weights = target
            pending_target = None

        if date in eval_dates:
            pending_target = strategy.target_weights(prices.iloc[: i + 1], weights.copy())

        net_return = equity / prev_equity - 1.0
        equity_records.append((date, equity))
        net_return_records.append((date, net_return))
        weight_records.append(
            {"date": date, **weights.to_dict(), "CASH": 1.0 - float(weights.sum())}
        )

    equity_curve = pd.Series(
        data=[v for _, v in equity_records],
        index=pd.DatetimeIndex([d for d, _ in equity_records]),
        name="equity",
        dtype=float,
    )
    daily_returns = pd.Series(
        data=[v for _, v in net_return_records],
        index=pd.DatetimeIndex([d for d, _ in net_return_records]),
        name="return",
        dtype=float,
    )
    weights_df = pd.DataFrame(weight_records).set_index("date")
    trades_df = pd.DataFrame(trade_records)
    if not trades_df.empty:
        trades_df = trades_df.set_index("date")

    metrics = performance_metrics(
        equity_curve,
        daily_returns,
        trades_df,
        risk_free_rate=risk_free_rate,
    )
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights=weights_df,
        trades=trades_df,
        metrics=metrics,
    )
