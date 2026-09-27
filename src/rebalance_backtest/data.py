from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


CACHE_DIR = Path(".cache") / "rebalance_backtest"


@dataclass(frozen=True)
class ExecutionData:
    close: pd.DataFrame
    dividends: pd.DataFrame
    splits: pd.DataFrame


def _field_frame(
    raw: pd.DataFrame,
    field: str,
    tickers: Sequence[str],
    *,
    default: float | None = None,
) -> pd.DataFrame:
    if raw.empty:
        if default is None:
            return pd.DataFrame()
        return pd.DataFrame(default, index=raw.index, columns=list(tickers))

    frame: pd.DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            if default is None:
                return pd.DataFrame()
            frame = pd.DataFrame(default, index=raw.index, columns=list(tickers))
        else:
            values = raw[field].copy()
            if isinstance(values, pd.Series):
                frame = values.to_frame(name=tickers[0])
            else:
                frame = values
    else:
        if field not in raw.columns:
            if default is None:
                return pd.DataFrame()
            frame = pd.DataFrame(default, index=raw.index, columns=list(tickers))
        else:
            frame = raw[[field]].copy()
            frame.columns = [tickers[0]]

    return frame.reindex(columns=list(tickers))


def _close_prices(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    prices = _field_frame(raw, "Close", tickers)
    if prices.empty:
        raise ValueError("Downloaded data does not contain Close prices.")
    return prices


def _cache_key(
    tickers: Sequence[str],
    start: str,
    end: str | None,
    mode: str,
) -> str:
    freshness = end or date.today().isoformat()
    key = "|".join([",".join(tickers), start, freshness, mode])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _cache_path(tickers: Sequence[str], start: str, end: str | None) -> Path:
    digest = _cache_key(tickers, start, end, "auto_adjust=true")
    return CACHE_DIR / f"prices_{digest}.csv"


def _execution_cache_paths(
    tickers: Sequence[str],
    start: str,
    end: str | None,
) -> tuple[Path, Path, Path]:
    digest = _cache_key(tickers, start, end, "auto_adjust=false_actions=true")
    return (
        CACHE_DIR / f"execution_close_{digest}.csv",
        CACHE_DIR / f"execution_dividends_{digest}.csv",
        CACHE_DIR / f"execution_splits_{digest}.csv",
    )


def _load_cache(path: Path, tickers: Sequence[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        cached = cached.reindex(columns=list(tickers))
        if cached.empty:
            return None
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        return cached.astype(float)
    except Exception:
        return None


def _save_cache(path: Path, prices: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)


def _clean_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.to_datetime(output.index).tz_localize(None)
    return output.sort_index()


def fetch_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close prices used for signals and total returns."""
    if not tickers:
        raise ValueError("At least one ticker is required.")

    tickers = list(dict.fromkeys(tickers))
    cache_path = _cache_path(tickers, start, end)
    cached = _load_cache(cache_path, tickers)
    if cached is not None and not cached.isna().all().any():
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

    prices = _clean_index(prices).dropna(how="any")
    if prices.empty:
        raise ValueError("No common trading dates remain after aligning tickers.")

    prices = prices.astype(float)
    _save_cache(cache_path, prices)
    return prices


def fetch_execution_data(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
) -> ExecutionData:
    """Download raw closes plus cash distributions for executable replay.

    Strategy signals use adjusted prices via :func:`fetch_prices`. This
    function intentionally keeps raw KRX close prices for order sizing and
    separately carries distributions so the execution layer does not display
    adjusted-price decimals as if they were tradable quotes.
    """
    if not tickers:
        raise ValueError("At least one ticker is required.")

    tickers = list(dict.fromkeys(tickers))
    close_path, dividend_path, split_path = _execution_cache_paths(
        tickers,
        start,
        end,
    )
    cached_close = _load_cache(close_path, tickers)
    cached_dividends = _load_cache(dividend_path, tickers)
    cached_splits = _load_cache(split_path, tickers)
    if (
        cached_close is not None
        and cached_dividends is not None
        and cached_splits is not None
        and not cached_close.isna().all().any()
    ):
        return ExecutionData(
            close=cached_close,
            dividends=cached_dividends.fillna(0.0),
            splits=cached_splits.fillna(0.0),
        )

    import yfinance as yf

    raw = pd.DataFrame()
    missing = tickers
    for attempt in range(3):
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        close = _field_frame(raw, "Close", tickers)
        missing = [
            ticker
            for ticker in tickers
            if ticker not in close.columns or close[ticker].dropna().empty
        ]
        if not missing:
            break
        if attempt < 2:
            time.sleep(1.0 * (attempt + 1))

    if raw.empty or missing:
        raise ValueError(
            "Raw execution price download failed for: " + ", ".join(missing)
        )

    close = _clean_index(_field_frame(raw, "Close", tickers)).dropna(how="any")
    dividends = _clean_index(
        _field_frame(raw, "Dividends", tickers, default=0.0)
    ).reindex(close.index).fillna(0.0)
    splits = _clean_index(
        _field_frame(raw, "Stock Splits", tickers, default=0.0)
    ).reindex(close.index).fillna(0.0)

    # KRX cash ETF quotes are won-denominated tick prices. Keeping them as
    # integer won values prevents adjusted-price artifacts such as 112965.351562.
    for ticker in close.columns:
        if str(ticker).endswith(".KS") or str(ticker).endswith(".KQ"):
            close[ticker] = np.rint(close[ticker].astype(float))

    close = close.astype(float)
    dividends = dividends.astype(float)
    splits = splits.astype(float)

    _save_cache(close_path, close)
    _save_cache(dividend_path, dividends)
    _save_cache(split_path, splits)
    return ExecutionData(close=close, dividends=dividends, splits=splits)
