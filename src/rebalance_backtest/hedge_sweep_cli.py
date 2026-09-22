from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .console import finish_status, live_status
from .data import fetch_prices
from .hedge_rotation import PeakHedgeRotationStrategy
from .rotation import AdaptiveRotationStrategy
from .rotation_cli import KR_BONDS, KR_STOCKS, MARKET_TICKER, SAFE_CHOICES
from .strategy import FixedWeightStrategy


HEDGES = {
    "132030.KS": "KODEX 골드선물(H)",
    "261240.KS": "KODEX 미국달러선물",
    "304660.KS": "KODEX 미국30년국채울트라선물(H)",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare peak protection and rotating hedge assets."
    )
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--safe", choices=SAFE_CHOICES, default="shortbond")
    p.add_argument("--initial-capital", type=float, default=10_000_000.0)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--output", default="results_hedge")
    return p.parse_args()


def _broad_signal(safe_ticker: str) -> AdaptiveRotationStrategy:
    return AdaptiveRotationStrategy(
        stock_tickers=list(KR_STOCKS),
        bond_tickers=list(KR_BONDS),
        safe_ticker=safe_ticker,
        market_ticker=MARKET_TICKER,
        market_weight=0.75,
        sector_breadth_weight=0.25,
    )


def _peak_hedge(
    safe_ticker: str,
    *,
    peak: bool,
    hedge: bool,
    low_turnover: bool = False,
) -> PeakHedgeRotationStrategy:
    return PeakHedgeRotationStrategy(
        stock_tickers=list(KR_STOCKS),
        bond_tickers=list(KR_BONDS),
        hedge_tickers=list(HEDGES),
        safe_ticker=safe_ticker,
        market_ticker=MARKET_TICKER,
        market_weight=0.75,
        sector_breadth_weight=0.25,
        enable_peak_protection=peak,
        enable_hedge_rotation=hedge,
        no_trade_band=0.075 if low_turnover else 0.05,
        min_trade_turnover=0.03 if low_turnover else 0.02,
        defensive_step=0.15 if low_turnover else 0.20,
    )


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cagr",
        "annual_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "trade_count",
        "total_turnover",
        "total_transaction_cost",
    ]
    return df[cols].copy()


def main() -> None:
    args = _parse_args()
    safe_ticker, safe_name = SAFE_CHOICES[args.safe]
    tickers = [
        *KR_STOCKS,
        *KR_BONDS,
        *HEDGES,
        safe_ticker,
    ]

    live_status("[hedge] downloading and aligning prices")
    prices = fetch_prices(tickers, args.start, args.end)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path("runs")
    runs.mkdir(parents=True, exist_ok=True)

    variants = {
        "kodex200_buy_hold": (
            FixedWeightStrategy({MARKET_TICKER: 1.0}),
            None,
        ),
        "static_60_40": (
            FixedWeightStrategy({MARKET_TICKER: 0.60, "114260.KS": 0.40}),
            "M",
        ),
        "broad_signal": (_broad_signal(safe_ticker), "D"),
        "peak_only": (
            _peak_hedge(safe_ticker, peak=True, hedge=False),
            "D",
        ),
        "hedge_only": (
            _peak_hedge(safe_ticker, peak=False, hedge=True),
            "D",
        ),
        "peak_hedge": (
            _peak_hedge(safe_ticker, peak=True, hedge=True),
            "D",
        ),
        "peak_hedge_low_turnover": (
            _peak_hedge(
                safe_ticker,
                peak=True,
                hedge=True,
                low_turnover=True,
            ),
            "D",
        ),
    }

    rows: dict[str, dict[str, float]] = {}
    results = {}
    for idx, (name, (strategy, schedule)) in enumerate(variants.items(), start=1):
        live_status(f"[hedge {idx}/{len(variants)}] {name}")
        result = run_backtest(
            prices,
            strategy,
            schedule=schedule,
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        rows[name] = result.metrics
        results[name] = result

    summary = pd.DataFrame(rows).T
    summary.index.name = "variant"
    summary.to_csv(out / "hedge_summary.csv")

    ranked = summary.loc[
        [
            "broad_signal",
            "peak_only",
            "hedge_only",
            "peak_hedge",
            "peak_hedge_low_turnover",
        ]
    ].sort_values(["sharpe", "calmar"], ascending=False)

    strategy = variants["peak_hedge"][0]
    current = results["peak_hedge"].weights.iloc[-1].drop(
        labels=["CASH"], errors="ignore"
    )
    latest = strategy.recommend(
        prices,
        current,
        apply_no_trade_band=True,
    )

    shown_scores = latest.scores.sort_values(ascending=False).to_frame("score")
    shown_scores["name"] = [
        KR_STOCKS.get(
            ticker,
            KR_BONDS.get(
                ticker,
                HEDGES.get(
                    ticker,
                    safe_name if ticker == safe_ticker else ticker,
                ),
            ),
        )
        for ticker in shown_scores.index
    ]

    shown_target = latest.target_weights[latest.target_weights > 1e-6].sort_values(
        ascending=False
    )

    report = [
        "=== Peak protection + hedge rotation experiment ===",
        (
            f"Data: {prices.index[0].date()} -> {prices.index[-1].date()} "
            f"| requested start={args.start} | safe={safe_name}"
        ),
        (
            "Hedges: "
            + ", ".join(f"{name}({ticker})" for ticker, name in HEDGES.items())
        ),
        (
            f"Initial capital: {args.initial_capital:,.0f} "
            f"| transaction cost: {args.transaction_cost_bps:.1f} bps"
        ),
        "",
        "=== Same-window baselines ===",
        _compact(summary.loc[["kodex200_buy_hold", "static_60_40", "broad_signal"]])
        .round(4)
        .to_string(),
        "",
        "=== Experiment variants (Sharpe order) ===",
        _compact(ranked).round(4).to_string(),
        "",
        "=== Latest peak+hedge state ===",
        (
            f"{latest.regime} | market={latest.market_score:.3f} "
            f"| defensive={latest.defensive_score:.3f} "
            f"| 20d={latest.short_return:.1%} "
            f"| peakDD={latest.peak_drawdown:.1%} "
            f"| peak_lock={latest.peak_protection} "
            f"| brake={latest.emergency_brake}"
        ),
        (
            f"Ideal sleeves: stock={latest.stock_target:.1%} "
            f"defensive={latest.defensive_target:.1%} "
            f"safe={latest.safe_target:.1%}"
        ),
        "",
        "=== Latest scores ===",
        shown_scores.to_string(
            formatters={"score": "{:.3f}".format},
        ),
        "",
        "=== Next-step suggested allocation ===",
    ]

    for ticker, weight in shown_target.items():
        name = KR_STOCKS.get(
            ticker,
            KR_BONDS.get(
                ticker,
                HEDGES.get(
                    ticker,
                    safe_name if ticker == safe_ticker else ticker,
                ),
            ),
        )
        report.append(f"{ticker} | {name} | {weight:.1%}")

    report.extend(
        [
            "",
            "Peak protection rule:",
            (
                "60d peak drawdown <= -4% AND "
                "(20d return <= 0 OR sector breadth <= 0)"
            ),
            "Caps: -4%=>80%, -6%=>65%, -8%=>50%, -10%=>35% stock max.",
            (
                "Defensive rotation: choose up to 2 positive safe-relative "
                "momentum leaders across KR 3Y govt bond + gold + USD + US 30Y."
            ),
        ]
    )

    report_text = "\n".join(report) + "\n"
    (out / "hedge_report.txt").write_text(report_text, encoding="utf-8")
    (runs / "latest_hedge_report.txt").write_text(report_text, encoding="utf-8")

    best_name = str(ranked.index[0])
    best = ranked.iloc[0]
    finish_status(
        f"[hedge] done | best={best_name}"
        f" | CAGR={best['cagr']:.2%}"
        f" | MDD={best['max_drawdown']:.2%}"
        f" | Sharpe={best['sharpe']:.3f}"
        f" | turnover={best['total_turnover']:.2f}"
        " | report=runs/latest_hedge_report.txt"
    )


if __name__ == "__main__":
    main()
