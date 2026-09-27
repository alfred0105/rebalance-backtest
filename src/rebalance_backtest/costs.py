from __future__ import annotations

import math

import pandas as pd


TOSS_KRX_COMMISSION_BPS = 1.5
TOSS_KRX_COMMISSION_RATE = TOSS_KRX_COMMISSION_BPS / 10_000.0
ETF_SECURITIES_TRANSACTION_TAX_RATE = 0.0


def truncate_won(value: float) -> int:
    """Toss domestic stock commission examples truncate sub-won amounts."""
    if value <= 0:
        return 0
    return int(math.floor(float(value) + 1e-12))


def toss_krx_commission(
    notional: float,
    *,
    commission_bps: float = TOSS_KRX_COMMISSION_BPS,
) -> int:
    if commission_bps < 0:
        raise ValueError("commission_bps cannot be negative")
    return truncate_won(float(notional) * float(commission_bps) / 10_000.0)


def toss_krx_commissions_by_ticker(
    notionals: pd.Series,
    *,
    commission_bps: float = TOSS_KRX_COMMISSION_BPS,
) -> pd.Series:
    values = notionals.fillna(0.0).clip(lower=0.0).astype(float)
    return values.map(
        lambda notional: toss_krx_commission(
            float(notional),
            commission_bps=commission_bps,
        )
    ).astype(int)
