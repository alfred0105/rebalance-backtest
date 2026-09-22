from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .console import finish_status, live_status
from .dynamic_selector import (
    DynamicSelectionConfig,
    load_state,
    save_state,
    select_dynamic_universe,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Maintain a stateful dynamic ETF universe with minimum holding "
            "periods, monthly replacement, emergency exits, and cooldowns."
        )
    )
    p.add_argument(
        "--candidates",
        default="universes/latest_candidates.csv",
    )
    p.add_argument(
        "--state",
        default="universes/dynamic_selector_state.json",
    )
    p.add_argument(
        "--active-output",
        default="universes/active_universe.json",
    )
    p.add_argument("--aggressive-slots", type=int, default=3)
    p.add_argument("--defensive-slots", type=int, default=2)
    p.add_argument("--min-hold", type=int, default=20)
    p.add_argument("--cooldown", type=int, default=5)
    p.add_argument("--replace-margin", type=float, default=0.15)
    p.add_argument("--reset-state", action="store_true")
    return p.parse_args()


def _holding_table(
    holdings: list[dict],
    scored: pd.DataFrame,
    as_of: pd.Timestamp,
) -> str:
    if not holdings:
        return "(none)"

    rows = scored.set_index("ticker", drop=False)
    output: list[dict[str, object]] = []
    for holding in holdings:
        ticker = str(holding["ticker"])
        row = rows.loc[ticker] if ticker in rows.index else None
        entered = pd.Timestamp(holding["entered"])
        held_days = int(
            max(
                0,
                __import__("numpy").busday_count(
                    entered.date(),
                    as_of.date(),
                ),
            )
        )
        output.append(
            {
                "ticker": ticker,
                "name": holding.get("name", ticker),
                "held_bd": held_days,
                "screen": (
                    float(row["screen_score"])
                    if row is not None
                    else float(holding.get("last_score", float("nan")))
                ),
                "momentum": (
                    float(row["momentum_score"])
                    if row is not None
                    else float("nan")
                ),
                "r1": float(row["return_1d"]) if row is not None else float("nan"),
                "r5": float(row["return_5d"]) if row is not None else float("nan"),
                "peak60": (
                    float(row["peak_drawdown_60d"])
                    if row is not None
                    else float("nan")
                ),
            }
        )
    frame = pd.DataFrame(output)
    return frame.round(4).to_string(index=False)


def main() -> None:
    args = _parse_args()
    candidates_path = Path(args.candidates)
    state_path = Path(args.state)
    active_path = Path(args.active_output)

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing candidates file: {candidates_path}. Run rebalance-discover first."
        )

    live_status("[select] loading ranked ETF candidates and prior selector state")
    scored = pd.read_csv(candidates_path)
    required = {
        "ticker",
        "name",
        "bucket",
        "screen_score",
        "momentum_score",
        "return_1d",
        "return_5d",
        "return_20d",
        "peak_drawdown_60d",
        "corr_market_126d",
        "annual_vol_63d",
        "cluster_id",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(
            "Candidate file is from an older discovery version; rerun "
            "rebalance-discover. Missing: " + ", ".join(missing)
        )

    previous = {} if args.reset_state else load_state(state_path)
    config = DynamicSelectionConfig(
        aggressive_slots=args.aggressive_slots,
        defensive_slots=args.defensive_slots,
        min_hold_business_days=args.min_hold,
        cooldown_business_days=args.cooldown,
        replacement_score_margin=args.replace_margin,
    )

    as_of = pd.Timestamp.today().normalize()
    state, active = select_dynamic_universe(
        scored,
        previous_state=previous,
        as_of=as_of,
        config=config,
    )

    save_state(state_path, state)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        json.dumps(active, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runs = Path("runs")
    runs.mkdir(parents=True, exist_ok=True)

    events = active.get("events") or []
    if events:
        event_lines = [
            json.dumps(event, ensure_ascii=False, sort_keys=True)
            for event in events
        ]
    else:
        event_lines = ["(no changes)"]

    cooldowns = state.get("cooldowns") or {}
    cooldown_lines = (
        [f"{ticker} -> {until}" for ticker, until in sorted(cooldowns.items())]
        if cooldowns
        else ["(none)"]
    )

    report = [
        "=== Dynamic ETF selector ===",
        f"As of: {active['as_of']}",
        (
            f"Market emergency: {active['market_emergency']} "
            f"| reason={active.get('market_emergency_reason')}"
        ),
        (
            f"Rules: min_hold={config.min_hold_business_days} business days "
            f"| cooldown={config.cooldown_business_days} business days "
            f"| replace_margin={config.replacement_score_margin:.2f}"
        ),
        (
            "Emergency: 1d <= -5% OR 5d <= -8% OR "
            "(60d peak drawdown <= -10% AND momentum < 0)."
        ),
        (
            "Market emergency: KODEX200 5d <= -6% OR 20d <= -8%; "
            "aggressive slots exit immediately and do not refill until shock clears."
        ),
        "",
        "=== Active aggressive satellites ===",
        _holding_table(state["aggressive"], scored, as_of),
        "",
        "=== Active defensive assets ===",
        _holding_table(state["defensive"], scored, as_of),
        "",
        "=== Selector actions this run ===",
        *event_lines,
        "",
        "=== Cooldowns ===",
        *cooldown_lines,
        "",
        "Notes:",
        "- KODEX200 is reserved as the market core and does not consume a satellite slot.",
        "- Normal replacements occur on monthly review only after minimum hold.",
        "- Empty slots created by an individual emergency can refill immediately with a different eligible ETF.",
        "- Correlation clusters prevent near-duplicate ETFs from occupying multiple slots.",
        "- This state is a research/paper selector state, not a brokerage position record.",
    ]
    report_text = "\n".join(report) + "\n"
    (runs / "latest_dynamic_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    aggr = ",".join(active["aggressive"]) or "none"
    defensive = ",".join(active["defensive"]) or "none"
    finish_status(
        f"[select] done | aggressive={aggr} | defensive={defensive} "
        "| report=runs/latest_dynamic_report.txt"
    )


if __name__ == "__main__":
    main()
