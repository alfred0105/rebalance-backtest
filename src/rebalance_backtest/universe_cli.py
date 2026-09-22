from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .console import finish_status, live_status
from .universe_discovery import (
    DiscoveryConfig,
    fetch_current_kr_etf_listing,
    save_monthly_snapshot,
    score_universe,
    shortlist,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discover and rank Korean-listed ETF candidates automatically."
    )
    p.add_argument("--prefilter", type=int, default=80)
    p.add_argument("--history-days", type=int, default=550)
    p.add_argument("--min-history", type=int, default=260)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--output", default="universes")
    return p.parse_args()


def _format_table(df: pd.DataFrame) -> str:
    columns = [
        "ticker",
        "name",
        "bucket",
        "screen_score",
        "momentum_score",
        "return_20d",
        "return_63d",
        "annual_vol_63d",
        "corr_market_126d",
        "max_drawdown_252d",
    ]
    shown = df.loc[:, columns].copy()
    return shown.round(4).to_string(index=False)


def main() -> None:
    args = _parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(parents=True, exist_ok=True)

    live_status("[discover] fetching current Korean ETF universe")
    listing = fetch_current_kr_etf_listing()
    snapshot = save_monthly_snapshot(listing, root)

    config = DiscoveryConfig(
        prefilter_count=args.prefilter,
        history_calendar_days=args.history_days,
        min_history_sessions=args.min_history,
        top_n=args.top,
    )

    live_status(
        f"[discover] scoring liquid/momentum prefilter from {len(listing)} ETFs"
    )
    scored = score_universe(listing, config=config)
    aggressive, defensive = shortlist(scored, top_n=args.top)

    scored.to_csv(root / "latest_candidates.csv", index=False)

    report = [
        "=== Automatic Korean ETF discovery ===",
        (
            f"Observed universe: {len(listing)} | scored after filters/history: "
            f"{len(scored)} | monthly snapshot={snapshot.as_posix()}"
        ),
        (
            "Filter: exclude leverage/inverse; prefilter by liquidity + current "
            "3M strength + defensive coverage; require sufficient daily history."
        ),
        "",
        "=== Aggressive/equity candidates ===",
        _format_table(aggressive),
        "",
        "=== Defensive/real-asset candidates ===",
        _format_table(defensive),
        "",
        "=== How to use this ===",
        (
            "This is a discovery shortlist, not an automatic trade list. "
            "Candidates must be validated in dated walk-forward tests before "
            "they are promoted into the live rotation universe."
        ),
        (
            "Screen score: equity = 65% momentum percentile + 20% liquidity "
            "+ 15% diversification; defensive = 50% momentum + 25% "
            "diversification + 25% drawdown resilience."
        ),
    ]
    report_text = "\n".join(report) + "\n"
    (runs := Path("runs")) .mkdir(parents=True, exist_ok=True)
    (runs / "latest_universe_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    top_eq = aggressive.iloc[0] if not aggressive.empty else None
    top_def = defensive.iloc[0] if not defensive.empty else None
    eq_text = (
        f"{top_eq['name']}({top_eq['ticker']})"
        if top_eq is not None
        else "none"
    )
    def_text = (
        f"{top_def['name']}({top_def['ticker']})"
        if top_def is not None
        else "none"
    )
    finish_status(
        f"[discover] done | equity={eq_text} | defensive={def_text} "
        "| report=runs/latest_universe_report.txt"
    )


if __name__ == "__main__":
    main()
