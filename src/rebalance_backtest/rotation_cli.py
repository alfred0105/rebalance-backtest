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
KR_BONDS = {
    "114260.KS": "KODEX 국고채3년",
}
SAFE_CHOICES = {
    "shortbond": ("153130.KS", "KODEX 단기채권"),
    "kofr": ("423160.KS", "KODEX KOFR금리액티브(합성)"),
    "mmf": ("488770.KS", "KODEX 머니마켓액티브"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest Korean ETF stock -> bond -> safe-asset adaptive rotation."
    )
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--safe", choices=SAFE_CHOICES, default="shortbond")
    p.add_argument("--initial-capital", type=float, default=10_000_000.0)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument(
        "--rebalance-band",
        "--no-trade-band",
        dest="no_trade_band",
        type=float,
        default=0.05,
        help="Allowed percentage-point deviation before a trade is triggered (default: 0.05).",
    )
    p.add_argument(
        "--max-weekly-shift",
        type=float,
        default=0.10,
        help="Maximum percentage-point change per risky ETF at one weekly evaluation (default: 0.10).",
    )
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--switch-margin", type=float, default=0.10)
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
        no_trade_band=args.no_trade_band,
        max_weekly_shift=args.max_weekly_shift,
        absolute_threshold=args.threshold,
        switch_margin=args.switch_margin,
    )

    baselines = {
        "kodex200_buy_hold": (FixedWeightStrategy({"069500.KS": 1.0}), None),
        "static_60_40": (
            FixedWeightStrategy({"069500.KS": 0.60, "114260.KS": 0.40}),
            "M",
        ),
        "safe_only": (FixedWeightStrategy({safe_ticker: 1.0}), None),
        "adaptive_rotation": (rotation, "W-FRI"),
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
    ax.set_title("Korean ETF adaptive rotation")
    ax.set_ylabel("Portfolio value (log scale)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "equity_curves.png", dpi=160)
    plt.close()

    current = results["adaptive_rotation"].weights.iloc[-1].drop(labels=["CASH"], errors="ignore")
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
            f"{recommendation.regime} | stock score={recommendation.stock_score:.3f} "
            f"bond score={recommendation.bond_score:.3f} | safe={safe_name}"
        )
        print(
            f"Rebalance band={args.no_trade_band:.1%} | "
            f"max weekly shift={args.max_weekly_shift:.1%}"
        )
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
