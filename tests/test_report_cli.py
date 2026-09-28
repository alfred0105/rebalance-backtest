import pandas as pd

from rebalance_backtest.report_cli import _integer_plan


def test_integer_plan_respects_capital_and_whole_shares():
    weights = {
        "A.KS": 0.45,
        "B.KS": 0.30,
        "C.KS": 0.25,
    }
    prices = {
        "A.KS": 113_000.0,
        "B.KS": 61_000.0,
        "C.KS": 14_500.0,
    }
    names = {ticker: ticker for ticker in weights}

    plan, cash = _integer_plan(weights, prices, names, 300_000.0)

    assert not plan.empty
    assert (plan["shares"] >= 0).all()
    assert all(float(x).is_integer() for x in plan["shares"].astype(float))
    assert plan["actual_value"].sum() + cash <= 300_000.0 + 1e-9
    assert cash >= -1e-9


def test_integer_plan_keeps_unaffordable_target_as_zero_shares():
    weights = {"EXPENSIVE.KS": 1.0}
    prices = {"EXPENSIVE.KS": 500_000.0}
    names = {"EXPENSIVE.KS": "Expensive"}

    plan, cash = _integer_plan(weights, prices, names, 300_000.0)

    assert int(plan.loc[0, "shares"]) == 0
    assert cash == 300_000.0


def test_weights_chart_handles_tiny_negative_cash():
    from rebalance_backtest.report_cli import _weights_chart

    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    weights = pd.DataFrame(
        {
            "A.KS": [0.6, 0.7, 0.8],
            "B.KS": [0.4, 0.3, 0.2],
            "CASH": [1e-16, -2e-16, 0.0],
        },
        index=dates,
    )

    uri = _weights_chart(weights)

    assert isinstance(uri, str)
    assert uri.startswith("data:image/png;base64,")


def test_paper_portfolio_table_handles_cash_without_int_dtype_error():
    from rebalance_backtest.report_cli import _paper_portfolio_table

    frame = pd.DataFrame(
        [
            {
                "ticker": "069500.KS",
                "name": "KODEX 200",
                "shares": 1,
                "price": 111_400,
                "value": 111_400,
                "weight": 0.3713,
            },
            {
                "ticker": "CASH",
                "name": "현금",
                "shares": 0,
                "price": 1,
                "value": 188_600,
                "weight": 0.6287,
            },
        ]
    )

    shown = _paper_portfolio_table(frame)

    assert shown.loc[shown["종목명"] == "현금", "수량"].iloc[0] == ""
    assert shown.loc[shown["종목명"] == "KODEX 200", "수량"].iloc[0] == 1
