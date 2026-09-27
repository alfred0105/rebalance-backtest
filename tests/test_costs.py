import pandas as pd

from rebalance_backtest.costs import (
    ETF_SECURITIES_TRANSACTION_TAX_RATE,
    TOSS_KRX_COMMISSION_BPS,
    toss_krx_commission,
    toss_krx_commissions_by_ticker,
)


def test_toss_krx_commission_matches_official_example_rounding():
    # 81,000 KRW * 12 shares * 0.015% = 145.8 KRW -> sub-won truncated.
    assert TOSS_KRX_COMMISSION_BPS == 1.5
    assert toss_krx_commission(81_000 * 12) == 145


def test_toss_commission_is_charged_per_ticker_notional():
    notionals = pd.Series({"A": 109_980.0, "B": 52_920.0})
    fees = toss_krx_commissions_by_ticker(notionals)

    assert fees["A"] == 16
    assert fees["B"] == 7
    assert int(fees.sum()) == 23


def test_domestic_listed_etf_transaction_tax_is_zero():
    assert ETF_SECURITIES_TRANSACTION_TAX_RATE == 0.0
