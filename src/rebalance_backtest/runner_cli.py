from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .console import finish_status, live_status


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run rotation + parameter sweep, then publish the two latest reports to GitHub."
    )
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--safe", default="shortbond")
    p.add_argument("--initial-capital", type=float, default=10_000_000.0)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Run both backtests but do not commit/push reports.",
    )
    return p.parse_args()


def _common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--start",
        args.start,
        "--safe",
        args.safe,
        "--initial-capital",
        str(args.initial_capital),
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
    ]
    if args.end:
        values.extend(["--end", args.end])
    return values


def _run_module(module: str, args: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *args],
        check=True,
    )


def _publish() -> None:
    reports = [
        Path("runs/latest_report.txt"),
        Path("runs/latest_sweep_report.txt"),
    ]
    missing = [str(path) for path in reports if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report(s): " + ", ".join(missing))

    live_status("[publish] staging latest reports")
    subprocess.run(
        ["git", "add", "--", *(str(path) for path in reports)],
        check=True,
    )

    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        check=False,
    ).returncode != 0

    if not changed:
        finish_status("[publish] no report changes; GitHub already current")
        return

    live_status("[publish] committing latest reports")
    subprocess.run(
        ["git", "commit", "-m", "Update latest backtest reports"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    live_status("[publish] pushing to GitHub")
    subprocess.run(
        ["git", "push"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    finish_status("[publish] GitHub reports updated")


def main() -> None:
    args = _parse_args()
    common = _common_args(args)

    live_status("[1/2] starting rotation backtest")
    _run_module("rebalance_backtest.rotation_cli", common)

    live_status("[2/2] starting parameter sweep")
    _run_module("rebalance_backtest.sweep_cli", common)

    if args.no_push:
        finish_status("[done] rotation + sweep complete; push skipped")
        return

    _publish()
    finish_status("[done] rotation + sweep + GitHub publish complete")


if __name__ == "__main__":
    main()
