from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .console import finish_status, live_status
from .paper_account import empty_paper_state, update_paper_account


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advance the persistent dynamic-strategy paper account."
    )
    p.add_argument("--initial-capital", type=float, default=300_000.0)
    p.add_argument("--active", default="universes/active_universe.json")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--state", default="runs/paper_account_state.json")
    return p.parse_args()


def _upsert_csv(path: Path, frame: pd.DataFrame, date_value: str) -> None:
    if path.exists():
        old = pd.read_csv(path)
        if "date" in old.columns:
            old = old.loc[old["date"].astype(str) != str(date_value)]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    active_path = Path(args.active)
    if not active_path.exists():
        raise FileNotFoundError(active_path)
    active = json.loads(active_path.read_text(encoding="utf-8"))
    as_of = str(active["as_of"])

    if args.snapshot:
        snapshot_path = Path(args.snapshot)
    else:
        snapshots = sorted(Path("universes/snapshots").glob("*.csv"))
        if not snapshots:
            raise FileNotFoundError("No universe snapshot found")
        snapshot_path = snapshots[-1]

    snapshot = pd.read_csv(snapshot_path)
    names = dict(
        zip(snapshot["yahoo_ticker"].astype(str), snapshot["name"].astype(str))
    )
    price_values = pd.to_numeric(snapshot["price_listing"], errors="coerce")
    prices = {
        str(ticker): float(price)
        for ticker, price in zip(snapshot["yahoo_ticker"], price_values)
        if pd.notna(price) and float(price) > 0
    }
    target_weights = {
        str(k): float(v)
        for k, v in active.get("allocation", {}).get("ideal_target_weights", {}).items()
    }

    state_path = Path(args.state)
    previous = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else empty_paper_state(args.initial_capital)
    )

    live_status("[paper] 동적 목표를 실제 정수주 paper 계좌에 반영")
    update = update_paper_account(
        target_weights=target_weights,
        prices=prices,
        names=names,
        as_of=as_of,
        previous_state=previous,
        initial_capital=args.initial_capital,
    )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(update.state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    live_plan = update.holdings_table.copy()
    cash_row = pd.DataFrame([{
        "ticker": "CASH",
        "name": "현금",
        "shares": 0,
        "price": 1,
        "value": int(round(float(update.state["cash"]))),
        "weight": (
            float(update.state["cash"]) / float(update.state["equity"])
            if float(update.state["equity"]) > 0 else 0.0
        ),
    }])
    live_plan = pd.concat([live_plan, cash_row], ignore_index=True)
    live_plan.to_csv("runs/latest_live_portfolio.csv", index=False)
    portfolio_dir = Path("runs/paper_portfolios")
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    live_plan.to_csv(portfolio_dir / f"{as_of}.csv", index=False)

    history_row = pd.DataFrame([{
        "date": as_of,
        "equity": float(update.state["equity"]),
        "cash": float(update.state["cash"]),
        "total_return": float(update.state["total_return"]),
        "total_commission": float(update.state["total_commission"]),
        "risk_state": active.get("allocation", {}).get("risk_state", "NORMAL"),
        "stock_target": active.get("allocation", {}).get("stock_target", 0.0),
        "defensive_target": active.get("allocation", {}).get("defensive_target", 0.0),
        "safe_target": active.get("allocation", {}).get("safe_target", 0.0),
    }])
    _upsert_csv(Path("runs/paper_account_history.csv"), history_row, as_of)

    if not update.trades.empty:
        _upsert_csv(Path("runs/paper_account_trades.csv"), update.trades, as_of)
    elif not Path("runs/paper_account_trades.csv").exists():
        pd.DataFrame(
            columns=["date","ticker","name","action","shares","price","notional","commission"]
        ).to_csv("runs/paper_account_trades.csv", index=False)

    finish_status(
        f"[paper] 완료 | equity={float(update.state['equity']):,.0f}원 "
        f"| return={float(update.state['total_return']):.2%} "
        f"| cash={float(update.state['cash']):,.0f}원 "
        f"| trades={len(update.trades)}"
    )


if __name__ == "__main__":
    main()
