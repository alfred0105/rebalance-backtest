from rebalance_backtest.paper_account import update_paper_account


def test_paper_account_carries_positions_forward_without_rebuying_same_target():
    first = update_paper_account(
        target_weights={"A.KS": 1.0},
        prices={"A.KS": 100_000.0},
        names={"A.KS": "A"},
        as_of="2026-09-28",
        previous_state=None,
        initial_capital=300_000.0,
    )

    assert first.state["holdings"]["A.KS"] == 2
    assert first.state["cash"] >= 0
    assert len(first.trades) == 1
    assert first.trades.iloc[0]["action"] == "BUY"

    second = update_paper_account(
        target_weights={"A.KS": 1.0},
        prices={"A.KS": 100_000.0},
        names={"A.KS": "A"},
        as_of="2026-09-29",
        previous_state=first.state,
        initial_capital=300_000.0,
    )

    assert second.state["holdings"]["A.KS"] == 2
    assert second.trades.empty
    assert second.state["cash"] == first.state["cash"]
    assert second.state["total_commission"] == first.state["total_commission"]


def test_paper_account_sells_then_buys_and_charges_commission():
    first = update_paper_account(
        target_weights={"A.KS": 1.0},
        prices={"A.KS": 50_000.0, "B.KS": 50_000.0},
        names={"A.KS": "A", "B.KS": "B"},
        as_of="2026-09-28",
        previous_state=None,
        initial_capital=300_000.0,
    )
    second = update_paper_account(
        target_weights={"B.KS": 1.0},
        prices={"A.KS": 50_000.0, "B.KS": 50_000.0},
        names={"A.KS": "A", "B.KS": "B"},
        as_of="2026-09-29",
        previous_state=first.state,
        initial_capital=300_000.0,
    )

    actions = set(second.trades["action"])
    assert "SELL" in actions
    assert "BUY" in actions
    assert second.state["holdings"].get("A.KS", 0) == 0
    assert second.state["holdings"].get("B.KS", 0) > 0
    assert second.state["total_commission"] > first.state["total_commission"]
    assert second.state["cash"] >= 0
