from pathlib import Path

from rebalance_backtest.runner_cli import _reset_forward_state


def test_clean_start_removes_only_forward_state(tmp_path: Path):
    removable_files = [
        "universes/dynamic_selector_state.json",
        "universes/active_universe.json",
        "runs/paper_account_state.json",
        "runs/paper_account_history.csv",
        "runs/paper_account_trades.csv",
        "runs/latest_live_portfolio.csv",
        "runs/latest_execution_plan.csv",
        "runs/latest_trade_log.csv",
        "runs/latest_dashboard.html",
    ]
    removable_dirs = [
        "universes/daily_candidates",
        "universes/daily_active",
        "runs/paper_portfolios",
        "runs/daily_execution",
    ]
    preserved = [
        "runs/latest_report.txt",
        "runs/latest_hedge_report.txt",
        "src/rebalance_backtest/dummy.py",
    ]

    for relative in removable_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old", encoding="utf-8")

    for relative in removable_dirs:
        path = tmp_path / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "old.txt").write_text("old", encoding="utf-8")

    for relative in preserved:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("keep", encoding="utf-8")

    removed = _reset_forward_state(tmp_path)

    assert len(removed) == len(removable_files) + len(removable_dirs)
    for relative in removable_files + removable_dirs:
        assert not (tmp_path / relative).exists()
    for relative in preserved:
        assert (tmp_path / relative).exists()


def test_clean_start_is_idempotent(tmp_path: Path):
    assert _reset_forward_state(tmp_path) == []
    assert _reset_forward_state(tmp_path) == []
