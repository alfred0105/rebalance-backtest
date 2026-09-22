from __future__ import annotations

import hashlib
import time
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd


CACHE_DIR = Path(".cache") / "rebalance_backtest"


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


def _cache_path(tickers: Sequence[str], start: str, end: str | None) -> Path:
    freshness = end or date.today().isoformat()
    key = "|".join([",".join(tickers), start, freshness, "auto_adjust=true"])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"prices_{digest}.csv"


def _load_cache(path: Path, tickers: Sequence[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        cached = cached.reindex(columns=list(tickers))
        if cached.empty or cached.isna().all().any():
            return None
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        return cached.astype(float)
    except Exception:
        return None


def _save_cache(path: Path, prices: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)


def fetch_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance.

    Downloads use a same-day local CSV cache so repeated rotation/sweep runs do
    not hit Yahoo multiple times. yfinance itself uses SQLite caches; parallel
    downloads are disabled to avoid Windows locking contention.
    """
    if not tickers:
        raise ValueError("At least one ticker is required.")

    tickers = list(dict.fromkeys(tickers))
    cache_path = _cache_path(tickers, start, end)
    cached = _load_cache(cache_path, tickers)
    if cached is not None:
        return cached

    import yfinance as yf

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

    prices = prices.astype(float)
    _save_cache(cache_path, prices)
    return prices
