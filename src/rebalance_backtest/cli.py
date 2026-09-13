from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import run_backtest
from .data import fetch_prices
from .strategy import FixedWeightStrategy, MomentumTiltStrategy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare buy-and-hold, fixed rebalancing, and asymmetric momentum rebalancing."
    )
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--initial-capital", type=float, default=10_000_000.0)
    parser.add_argument("--output", default="results")
    return parser.parse_args()


def _base_weights(tickers: list[str], supplied: list[float] | None) -> dict[str, float]:
    if supplied is None:
        w = 1.0 / len(tickers)
        return dict(zip(tickers, [w] * len(tickers)))
    if len(supplied) != len(tickers):
        raise SystemExit("--weights must have the same count as --tickers")
    if any(w < 0 for w in supplied) or sum(supplied) > 1.0 + 1e-9:
        raise SystemExit("weights must be non-negative and sum to <= 1")
    return dict(zip(tickers, supplied))


def main() -> None:
    args = _parse_args()
    tickers = [t.upper() for t in args.tickers]
    base = _base_weights(tickers, args.weights)
    prices = fetch_prices(tickers, args.start, args.end)

    strategies = {
        "buy_hold": (FixedWeightStrategy(base), None),
        "fixed_monthly": (FixedWeightStrategy(base), "M"),
        "fixed_weekly": (FixedWeightStrategy(base), "W-FRI"),
        "momentum_weekly": (MomentumTiltStrategy(base), "W-FRI"),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, float]] = {}
    curves: dict[str, pd.Series] = {}

    for name, (strategy, schedule) in strategies.items():
        result = run_backtest(
            prices,
            strategy,
            schedule=schedule,
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        summaries[name] = result.metrics
        curves[name] = result.equity_curve
        result.weights.to_csv(out / f"weights_{name}.csv")
        result.trades.to_csv(out / f"trades_{name}.csv")

    summary = pd.DataFrame(summaries).T
    summary.to_csv(out / "summary.csv")
    pd.DataFrame(curves).to_csv(out / "equity_curves.csv")

    ax = pd.DataFrame(curves).plot(figsize=(11, 6), logy=True)
    ax.set_title("Portfolio equity curves")
    ax.set_ylabel("Portfolio value (log scale)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "equity_curves.png", dpi=160)
    plt.close()

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.round(4))
    print(f"\nSaved results to: {out.resolve()}")


if __name__ == "__main__":
    main()
