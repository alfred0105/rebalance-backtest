from __future__ import annotations

from typing import Sequence

import pandas as pd


def fetch_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance.

    The yfinance import is intentionally local so the core engine and tests can
    run without network access.
    """
    if not tickers:
        raise ValueError("At least one ticker is required.")

    import yfinance as yf

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.reindex(columns=list(tickers))
    prices = prices.dropna(how="any")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    if prices.empty:
        raise ValueError("No common trading dates remain after aligning tickers.")

    return prices.astype(float)
