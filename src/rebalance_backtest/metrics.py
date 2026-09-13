from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    equity = equity_curve.dropna()
    rets = daily_returns.dropna()
    if len(equity) < 2:
        raise ValueError("Need at least two equity observations.")

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    calendar_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = calendar_days / 365.25
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)

    annual_vol = float(rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 else 0.0
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess = rets - daily_rf
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * math.sqrt(252))
        if len(excess) > 1 and excess.std(ddof=1) > 0
        else float("nan")
    )

    downside = excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1) * math.sqrt(252))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else float("nan")

    if trades.empty:
        trade_count = 0.0
        total_turnover = 0.0
        total_cost = 0.0
    else:
        trade_count = float(len(trades))
        total_turnover = float(trades["turnover"].sum())
        total_cost = float(trades["cost"].sum())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "trade_count": trade_count,
        "total_turnover": total_turnover,
        "total_transaction_cost": total_cost,
    }
