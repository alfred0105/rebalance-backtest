from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import run_backtest
from .data import fetch_prices
from .rotation import AdaptiveRotationStrategy
from .strategy import FixedWeightStrategy


KR_STOCKS = {
    "069500.KS": "KODEX 200",
    "091160.KS": "KODEX 반도체",
    "102780.KS": "KODEX 삼성그룹",
    "229200.KS": "KODEX 코스닥150",
}
KR_BONDS = {"114260.KS": "KODEX 국고채3년"}
SAFE_CHOICES = {
    "shortbond": ("153130.KS", "KODEX 단기채권"),
    "kofr": ("423160.KS", "KODEX KOFR금리액티브(합성)"),
    "mmf": ("488770.KS", "KODEX 머니마켓액티브"),
}
MARKET_TICKER = "069500.KS"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily-monitored Korean ETF stock -> bond -> safe adaptive rotation."
    )
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--safe", choices=SAFE_CHOICES, default="shortbond")
    p.add_argument("--initial-capital", type=float, default=10_000_000.0)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--rebalance-band", type=float, default=0.05)
    p.add_argument("--risk-on-step", type=float, default=0.05)
    p.add_argument("--risk-off-step", type=float, default=0.20)
    p.add_argument("--emergency-step", type=float, default=0.35)
    p.add_argument("--sector-step", type=float, default=0.05)
    p.add_argument("--brake-lookback", type=int, default=20)
    p.add_argument("--brake-threshold", type=float, default=-0.08)
    p.add_argument("--brake-stock-cap", type=float, default=0.35)
    p.add_argument("--output", default="results_rotation")
    return p.parse_args()


def _pretty_ticker(ticker: str, safe_ticker: str, safe_name: str) -> str:
    if ticker in KR_STOCKS:
        return KR_STOCKS[ticker]
    if ticker in KR_BONDS:
        return KR_BONDS[ticker]
    if ticker == safe_ticker:
        return safe_name
    return ticker


def main() -> None:
    args = _parse_args()
    safe_ticker, safe_name = SAFE_CHOICES[args.safe]
    tickers = [*KR_STOCKS, *KR_BONDS, safe_ticker]
    prices = fetch_prices(tickers, args.start, args.end)

    rotation = AdaptiveRotationStrategy(
        stock_tickers=list(KR_STOCKS),
        bond_tickers=list(KR_BONDS),
        safe_ticker=safe_ticker,
        market_ticker=MARKET_TICKER,
        no_trade_band=args.rebalance_band,
        risk_on_step=args.risk_on_step,
        risk_off_step=args.risk_off_step,
        emergency_step=args.emergency_step,
        sector_step=args.sector_step,
        brake_lookback=args.brake_lookback,
        brake_threshold=args.brake_threshold,
        brake_stock_cap=args.brake_stock_cap,
    )

    baselines = {
        "kodex200_buy_hold": (FixedWeightStrategy({MARKET_TICKER: 1.0}), None),
        "static_60_40": (
            FixedWeightStrategy({MARKET_TICKER: 0.60, "114260.KS": 0.40}),
            "M",
        ),
        "safe_only": (FixedWeightStrategy({safe_ticker: 1.0}), None),
        "adaptive_rotation_daily": (rotation, "D"),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, float]] = {}
    curves: dict[str, pd.Series] = {}
    results = {}
    for name, (strategy, schedule) in baselines.items():
        result = run_backtest(
            prices,
            strategy,
            schedule=schedule,
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        results[name] = result
        summaries[name] = result.metrics
        curves[name] = result.equity_curve
        result.weights.to_csv(out / f"weights_{name}.csv")
        result.trades.to_csv(out / f"trades_{name}.csv")

    summary = pd.DataFrame(summaries).T
    summary.to_csv(out / "summary.csv")
    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(out / "equity_curves.csv")

    ax = curve_df.plot(figsize=(11, 6), logy=True)
    ax.set_title("Korean ETF adaptive rotation v0.4 - daily monitoring")
    ax.set_ylabel("Portfolio value (log scale)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "equity_curves.png", dpi=160)
    plt.close()

    current = results["adaptive_rotation_daily"].weights.iloc[-1].drop(
        labels=["CASH"], errors="ignore"
    )
    recommendation = rotation.recommend(prices, current, apply_no_trade_band=True)

    score_rows = []
    for ticker, score in recommendation.scores.sort_values(ascending=False).items():
        score_rows.append(
            {
                "ticker": ticker,
                "name": _pretty_ticker(ticker, safe_ticker, safe_name),
                "score": score,
                "bucket": (
                    "STOCK" if ticker in KR_STOCKS else "BOND" if ticker in KR_BONDS else "SAFE"
                ),
            }
        )
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(out / "latest_scores.csv", index=False)

    target_rows = []
    for ticker, weight in recommendation.target_weights.items():
        if weight > 1e-6:
            target_rows.append(
                {
                    "ticker": ticker,
                    "name": _pretty_ticker(ticker, safe_ticker, safe_name),
                    "target_weight": weight,
                }
            )
    target_df = pd.DataFrame(target_rows).sort_values("target_weight", ascending=False)
    target_df.to_csv(out / "latest_recommendation.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\n=== Backtest summary ===")
        print(summary.round(4))
        print("\n=== Latest regime ===")
        print(
            f"{recommendation.regime} | market={recommendation.market_score:.3f} "
            f"bond={recommendation.bond_score:.3f} | 20d return={recommendation.short_return:.1%} "
            f"| brake={recommendation.emergency_brake}"
        )
        print(
            f"Ideal sleeves: stock={recommendation.stock_target:.1%} "
            f"bond={recommendation.bond_target:.1%} safe={recommendation.safe_target:.1%}"
        )
        print(
            f"Daily monitor | band={args.rebalance_band:.1%} | risk-on +{args.risk_on_step:.1%}p/day | "
            f"risk-off -{args.risk_off_step:.1%}p/day | emergency -{args.emergency_step:.1%}p/day"
        )
        print("Sector selection refresh: monthly")
        print("\n=== Latest scores ===")
        print(scores_df.to_string(index=False, formatters={"score": "{:.3f}".format}))
        print("\n=== Next-step suggested allocation ===")
        shown = target_df.copy()
        shown["target_weight"] = shown["target_weight"].map(lambda x: f"{x:.1%}")
        print(shown.to_string(index=False))

    print(f"\nData range actually used: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Saved results to: {out.resolve()}")


if __name__ == "__main__":
    main()
