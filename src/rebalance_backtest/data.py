from __future__ import annotations

import time
from typing import Sequence

import pandas as pd


def _close_prices(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = raw["Close"].copy()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    return prices.reindex(columns=list(tickers))


def fetch_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance.

    yfinance keeps a small local SQLite cache. Parallel downloads can
    occasionally contend for that cache on Windows, so downloads are forced
    to a single thread and retried before failing.
    """
    if not tickers:
        raise ValueError("At least one ticker is required.")

    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    prices = pd.DataFrame()
    missing = tickers

    for attempt in range(3):
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        prices = _close_prices(raw, tickers)
        missing = [
            ticker
            for ticker in tickers
            if ticker not in prices.columns or prices[ticker].dropna().empty
        ]
        if not missing:
            break
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))

    if prices.empty:
        raise ValueError("No price data returned. Check tickers and date range.")
    if missing:
        raise ValueError(
            "Price download failed for: "
            + ", ".join(missing)
            + ". Close other Python/yfinance processes and retry."
        )

    prices = prices.dropna(how="any")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    if prices.empty:
        raise ValueError("No common trading dates remain after aligning tickers.")

    return prices.astype(float)
