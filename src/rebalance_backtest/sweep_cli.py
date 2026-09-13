from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .data import fetch_prices
from .rotation import AdaptiveRotationStrategy
from .rotation_cli import KR_BONDS, KR_STOCKS, MARKET_TICKER, SAFE_CHOICES
from .strategy import FixedWeightStrategy


@dataclass(frozen=True)
class SweepPreset:
    name: str
    description: str
    params: dict[str, float] = field(default_factory=dict)


PRESETS = [
    SweepPreset("baseline_v05", "Current v0.5 reference settings."),
    SweepPreset(
        "core60",
        "Raise broad KODEX200 share inside the stock sleeve from 50% to 60%.",
        {"market_core_fraction": 0.60},
    ),
    SweepPreset(
        "core70",
        "Raise broad KODEX200 share inside the stock sleeve to 70%.",
        {"market_core_fraction": 0.70},
    ),
    SweepPreset(
        "stock85",
        "Cap total stock exposure at 85% in the strongest regime.",
        {"max_stock_exposure": 0.85},
    ),
    SweepPreset(
        "stock80",
        "Cap total stock exposure at 80% for a larger defensive reserve.",
        {"max_stock_exposure": 0.80},
    ),
    SweepPreset(
        "core60_stock85",
        "More broad-market core plus an 85% stock cap.",
        {"market_core_fraction": 0.60, "max_stock_exposure": 0.85},
    ),
    SweepPreset(
        "low_turnover",
        "Wider 7.5% band and 3% minimum portfolio turnover per ordinary trade.",
        {"no_trade_band": 0.075, "min_trade_turnover": 0.03},
    ),
    SweepPreset(
        "very_low_turnover",
        "10% band and 5% minimum turnover: deliberately sparse ordinary trading.",
        {"no_trade_band": 0.10, "min_trade_turnover": 0.05},
    ),
    SweepPreset(
        "fast_reentry",
        "Allow stock exposure to rebuild by 10 percentage points per day.",
        {"risk_on_step": 0.10},
    ),
    SweepPreset(
        "early_brake",
        "Trigger the crash brake earlier and cut risk more aggressively.",
        {
            "brake_threshold": -0.06,
            "brake_stock_cap": 0.30,
            "risk_off_step": 0.25,
            "emergency_step": 0.45,
        },
    ),
    SweepPreset(
        "broad_signal",
        "Let KODEX200 drive 75% of the market signal and sector breadth 25%.",
        {"market_weight": 0.75, "sector_breadth_weight": 0.25},
    ),
    SweepPreset(
        "balanced_candidate",
        "Candidate: broader core, slightly lower stock cap, lower churn, faster re-entry, earlier brake.",
        {
            "market_core_fraction": 0.60,
            "max_stock_exposure": 0.85,
            "market_weight": 0.70,
            "sector_breadth_weight": 0.30,
            "no_trade_band": 0.075,
            "min_trade_turnover": 0.03,
            "risk_on_step": 0.10,
            "risk_off_step": 0.25,
            "emergency_step": 0.45,
            "brake_threshold": -0.07,
            "brake_stock_cap": 0.30,
            "bond_max_fraction_of_residual": 0.75,
        },
    ),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a curated parameter sweep for adaptive rotation.")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--safe", choices=SAFE_CHOICES, default="shortbond")
    p.add_argument("--initial-capital", type=float, default=10_000_000.0)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--output", default="results_sweep")
    return p.parse_args()


def _strategy(safe_ticker: str, params: dict[str, float]) -> AdaptiveRotationStrategy:
    defaults: dict[str, float] = {
        "no_trade_band": 0.05,
        "min_trade_turnover": 0.02,
        "risk_on_step": 0.05,
        "risk_off_step": 0.20,
        "emergency_step": 0.35,
        "sector_step": 0.05,
        "max_stock_exposure": 0.90,
        "market_core_fraction": 0.50,
        "sector_max_weight": 0.25,
        "brake_threshold": -0.08,
        "brake_stock_cap": 0.35,
        "market_weight": 0.60,
        "sector_breadth_weight": 0.40,
        "bond_max_fraction_of_residual": 0.90,
    }
    defaults.update(params)
    return AdaptiveRotationStrategy(
        stock_tickers=list(KR_STOCKS),
        bond_tickers=list(KR_BONDS),
        safe_ticker=safe_ticker,
        market_ticker=MARKET_TICKER,
        **defaults,
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
    tickers = [*KR_STOCKS, *KR_BONDS, safe_ticker]
    prices = fetch_prices(tickers, args.start, args.end)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, float]] = {}

    baselines = {
        "kodex200_buy_hold": (FixedWeightStrategy({MARKET_TICKER: 1.0}), None),
        "static_60_40": (
            FixedWeightStrategy({MARKET_TICKER: 0.60, "114260.KS": 0.40}),
            "M",
        ),
    }
    for name, (strategy, schedule) in baselines.items():
        result = run_backtest(
            prices,
            strategy,
            schedule=schedule,
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        rows[name] = result.metrics

    for preset in PRESETS:
        strategy = _strategy(safe_ticker, preset.params)
        result = run_backtest(
            prices,
            strategy,
            schedule="D",
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        rows[preset.name] = result.metrics

    summary = pd.DataFrame(rows).T
    summary.index.name = "variant"
    summary.to_csv(out / "sweep_summary.csv")

    variants = summary.loc[[preset.name for preset in PRESETS]].copy()
    variants_by_sharpe = variants.sort_values(["sharpe", "calmar"], ascending=False)
    guardrails = variants[
        (variants["max_drawdown"] >= -0.30)
        & (variants["total_turnover"] <= 10.0)
    ].sort_values(["sharpe", "cagr"], ascending=False)

    report: list[str] = [
        "=== Rotation parameter sweep ===",
        f"Data: {prices.index[0].date()} -> {prices.index[-1].date()} | safe={safe_name}",
        f"Initial capital: {args.initial_capital:,.0f} | transaction cost: {args.transaction_cost_bps:.1f} bps",
        "",
        "=== Baselines ===",
        _compact(summary.loc[["kodex200_buy_hold", "static_60_40"]]).round(4).to_string(),
        "",
        "=== All variants (Sharpe order) ===",
        _compact(variants_by_sharpe).round(4).to_string(),
        "",
        "=== Guardrail candidates: MDD <= 30% and turnover <= 10 ===",
    ]
    if guardrails.empty:
        report.append("No variant met both guardrails.")
    else:
        report.append(_compact(guardrails).round(4).to_string())

    report.extend(["", "=== Preset definitions ==="])
    for preset in PRESETS:
        params = ", ".join(f"{k}={v:g}" for k, v in preset.params.items()) or "defaults"
        report.append(f"- {preset.name}: {preset.description} [{params}]")

    report_text = "\n".join(report) + "\n"
    (out / "sweep_report.txt").write_text(report_text, encoding="utf-8")
    Path("runs/latest_sweep_report.txt").write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Saved CSV: {(out / 'sweep_summary.csv').resolve()}")
    print(f"GitHub-readable report: {(Path('runs') / 'latest_sweep_report.txt').resolve()}")


if __name__ == "__main__":
    main()
