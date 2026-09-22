from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NAVER_ETF_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"

EXCLUDE_KEYWORDS = (
    "레버리지",
    "인버스",
    "2X",
    "2배",
)

DEFENSIVE_KEYWORDS = (
    "국채",
    "채권",
    "미국채",
    "달러",
    "KOFR",
    "CD금리",
    "머니마켓",
    "단기채",
    "초단기",
    "금리액티브",
)

REAL_ASSET_KEYWORDS = (
    "골드",
    "금선물",
    "금현물",
    "원유",
    "구리",
    "은선물",
    "원자재",
    "농산물",
)


@dataclass(frozen=True)
class DiscoveryConfig:
    safe_ticker: str = "153130.KS"
    market_ticker: str = "069500.KS"
    prefilter_count: int = 80
    history_calendar_days: int = 550
    min_history_sessions: int = 260
    top_n: int = 12
    cluster_corr_threshold: float = 0.90


def _classify_name(name: str) -> str:
    upper = str(name).upper()
    if any(keyword.upper() in upper for keyword in EXCLUDE_KEYWORDS):
        return "EXCLUDED"
    if any(keyword.upper() in upper for keyword in DEFENSIVE_KEYWORDS):
        return "DEFENSIVE"
    if any(keyword.upper() in upper for keyword in REAL_ASSET_KEYWORDS):
        return "REAL_ASSET"
    return "EQUITY"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def fetch_current_kr_etf_listing() -> pd.DataFrame:
    """Fetch the current Korean-listed ETF universe from Naver Finance.

    This endpoint is used for live discovery only. Historical walk-forward work
    must use dated snapshots rather than pretending today's universe existed in
    the past.
    """
    import requests

    response = requests.get(
        NAVER_ETF_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.naver.com/",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("result", {}).get("etfItemList", [])
    if not items:
        raise ValueError("ETF listing endpoint returned no products.")

    raw = pd.DataFrame(items)
    rename = {
        "itemcode": "symbol",
        "itemname": "name",
        "marketSum": "market_cap_raw",
        "quant": "volume",
        "threeMonthEarnRate": "return_3m_listing",
        "nowVal": "price_listing",
        "nav": "nav",
        "etfTabCode": "category_raw",
    }
    listing = raw.rename(columns=rename)
    required = ["symbol", "name"]
    missing = [column for column in required if column not in listing.columns]
    if missing:
        raise ValueError(f"ETF listing is missing columns: {missing}")

    for column in [
        "market_cap_raw",
        "volume",
        "return_3m_listing",
        "price_listing",
        "nav",
    ]:
        if column not in listing:
            listing[column] = 0.0
        listing[column] = _numeric(listing[column])

    listing["symbol"] = listing["symbol"].astype(str).str.zfill(6)
    listing["yahoo_ticker"] = listing["symbol"] + ".KS"
    listing["bucket"] = listing["name"].map(_classify_name)
    listing["listing_date_observed"] = date.today().isoformat()

    listing = listing.drop_duplicates("symbol").reset_index(drop=True)
    return listing


def _liquidity_score(listing: pd.DataFrame) -> pd.Series:
    cap_rank = listing["market_cap_raw"].rank(pct=True)
    volume_rank = listing["volume"].rank(pct=True)
    return 0.65 * cap_rank + 0.35 * volume_rank


def _prefilter(listing: pd.DataFrame, count: int) -> pd.DataFrame:
    eligible = listing.loc[listing["bucket"] != "EXCLUDED"].copy()
    eligible = eligible.loc[
        (eligible["market_cap_raw"] > 0) & (eligible["volume"] > 0)
    ].copy()
    if eligible.empty:
        raise ValueError("No liquid ETF candidates remain after filtering.")

    eligible["liquidity_score"] = _liquidity_score(eligible)
    half = max(10, count // 2)

    liquid = eligible.nlargest(half, "liquidity_score")
    momentum = eligible.nlargest(half, "return_3m_listing")
    defensive = (
        eligible.loc[eligible["bucket"].isin(["DEFENSIVE", "REAL_ASSET"])]
        .nlargest(max(12, count // 4), "liquidity_score")
    )

    combined = pd.concat([liquid, momentum, defensive], ignore_index=True)
    combined = combined.drop_duplicates("symbol")

    if len(combined) > count:
        combined = combined.sort_values(
            ["liquidity_score", "return_3m_listing"],
            ascending=False,
        ).head(count)

    return combined.reset_index(drop=True)


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.Series(dtype=float)
        close = raw["Close"]
        if isinstance(close, pd.Series):
            return close.astype(float)
        if ticker not in close.columns:
            return pd.Series(dtype=float)
        return close[ticker].astype(float)

    if "Close" not in raw.columns:
        return pd.Series(dtype=float)
    return raw["Close"].astype(float)


def _download_history(
    tickers: Iterable[str],
    *,
    calendar_days: int,
) -> dict[str, pd.Series]:
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    start = (date.today() - timedelta(days=calendar_days)).isoformat()
    output: dict[str, pd.Series] = {}

    # Chunks keep requests manageable while avoiding yfinance's Windows SQLite
    # contention from highly parallel downloads.
    chunk_size = 40
    for offset in range(0, len(tickers), chunk_size):
        chunk = tickers[offset : offset + chunk_size]
        raw = yf.download(
            chunk,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
        for ticker in chunk:
            series = _extract_close(raw, ticker).dropna()
            if not series.empty:
                series.index = pd.to_datetime(series.index).tz_localize(None)
                output[ticker] = series

    return output


def _relative_momentum(
    asset: pd.Series,
    safe: pd.Series,
    *,
    lookbacks: tuple[int, ...] = (63, 126, 252),
    weights: tuple[float, ...] = (0.20, 0.30, 0.50),
    vol_window: int = 63,
) -> float:
    aligned = pd.concat([asset.rename("asset"), safe.rename("safe")], axis=1).dropna()
    required = max(max(lookbacks) + 1, vol_window + 1)
    if len(aligned) < required:
        return float("nan")

    log_values = np.log(aligned[["asset", "safe"]].to_numpy(dtype=float))
    daily = np.diff(log_values[:, 0])
    vol = max(float(np.std(daily[-vol_window:], ddof=1)), 0.0025)

    score = 0.0
    for horizon, weight in zip(lookbacks, weights):
        asset_return = log_values[-1, 0] - log_values[-(horizon + 1), 0]
        safe_return = log_values[-1, 1] - log_values[-(horizon + 1), 1]
        score += weight * ((asset_return - safe_return) / (vol * np.sqrt(horizon)))
    return float(np.clip(score, -3.0, 3.0))


def _market_correlation(asset: pd.Series, market: pd.Series, window: int = 126) -> float:
    aligned = pd.concat(
        [asset.rename("asset"), market.rename("market")],
        axis=1,
    ).dropna()
    if len(aligned) < window + 1:
        return float("nan")
    returns = np.log(aligned / aligned.shift(1)).dropna().tail(window)
    return float(returns["asset"].corr(returns["market"]))


def _max_drawdown(series: pd.Series, window: int = 252) -> float:
    sample = series.dropna().tail(window)
    if len(sample) < 2:
        return float("nan")
    drawdown = sample / sample.cummax() - 1.0
    return float(drawdown.min())


def _peak_drawdown(series: pd.Series, window: int = 60) -> float:
    sample = series.dropna().tail(window)
    if len(sample) < 2:
        return float("nan")
    peak = float(sample.max())
    if peak <= 0:
        return float("nan")
    return float(sample.iloc[-1] / peak - 1.0)


def _simple_return(series: pd.Series, sessions: int) -> float:
    series = series.dropna()
    if len(series) <= sessions:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[-(sessions + 1)] - 1.0)


def score_universe(
    listing: pd.DataFrame,
    *,
    config: DiscoveryConfig,
) -> pd.DataFrame:
    candidates = _prefilter(listing, config.prefilter_count)
    requested = [
        *candidates["yahoo_ticker"].tolist(),
        config.safe_ticker,
        config.market_ticker,
    ]
    history = _download_history(
        requested,
        calendar_days=config.history_calendar_days,
    )

    safe = history.get(config.safe_ticker)
    market = history.get(config.market_ticker)
    if safe is None or market is None:
        raise ValueError("Safe or market benchmark price history could not be downloaded.")

    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        ticker = str(row.yahoo_ticker)
        series = history.get(ticker)
        if series is None or len(series) < config.min_history_sessions:
            continue

        log_returns = np.log(series / series.shift(1)).dropna()
        annual_vol = (
            float(log_returns.tail(63).std(ddof=1) * np.sqrt(252))
            if len(log_returns) >= 63
            else float("nan")
        )
        rows.append(
            {
                "symbol": row.symbol,
                "ticker": ticker,
                "name": row.name,
                "bucket": row.bucket,
                "history_sessions": int(len(series)),
                "momentum_score": _relative_momentum(series, safe),
                "return_1d": _simple_return(series, 1),
                "return_5d": _simple_return(series, 5),
                "return_20d": _simple_return(series, 20),
                "return_63d": _simple_return(series, 63),
                "peak_drawdown_60d": _peak_drawdown(series, 60),
                "annual_vol_63d": annual_vol,
                "corr_market_126d": _market_correlation(series, market),
                "max_drawdown_252d": _max_drawdown(series),
                "market_cap_raw": float(row.market_cap_raw),
                "volume": float(row.volume),
                "liquidity_score": float(row.liquidity_score),
                "return_3m_listing": float(row.return_3m_listing),
            }
        )

    scored = pd.DataFrame(rows)
    if scored.empty:
        raise ValueError("No ETF candidate had sufficient price history.")

    scored = scored.replace([np.inf, -np.inf], np.nan)
    scored = scored.dropna(subset=["momentum_score"])

    scored["momentum_pct"] = scored["momentum_score"].rank(pct=True)
    scored["liquidity_pct"] = scored["liquidity_score"].rank(pct=True)
    corr = scored["corr_market_126d"].fillna(1.0).clip(-1.0, 1.0)
    scored["diversification_pct"] = (1.0 - corr.clip(lower=0.0)).rank(pct=True)
    scored["drawdown_resilience_pct"] = scored["max_drawdown_252d"].rank(pct=True)

    equity = scored["bucket"] == "EQUITY"
    defensive = scored["bucket"].isin(["DEFENSIVE", "REAL_ASSET"])

    scored["screen_score"] = np.nan
    scored.loc[equity, "screen_score"] = (
        0.65 * scored.loc[equity, "momentum_pct"]
        + 0.20 * scored.loc[equity, "liquidity_pct"]
        + 0.15 * scored.loc[equity, "diversification_pct"]
    )
    scored.loc[defensive, "screen_score"] = (
        0.50 * scored.loc[defensive, "momentum_pct"]
        + 0.25 * scored.loc[defensive, "diversification_pct"]
        + 0.25 * scored.loc[defensive, "drawdown_resilience_pct"]
    )

    scored = scored.sort_values("screen_score", ascending=False).reset_index(drop=True)

    # Build one return matrix/correlation matrix for all scored ETFs, then
    # perform cheap lookups during greedy clustering. This avoids repeatedly
    # concatenating pairwise histories as the universe grows.
    return_columns: dict[str, pd.Series] = {}
    for ticker in scored["ticker"].astype(str):
        series = history.get(ticker)
        if series is not None and len(series) >= 64:
            return_columns[ticker] = np.log(series / series.shift(1)).dropna()

    if return_columns:
        return_frame = pd.concat(return_columns, axis=1).sort_index().tail(126)
        corr_matrix = return_frame.corr(min_periods=63)
    else:
        corr_matrix = pd.DataFrame()

    representatives: list[tuple[int, str]] = []
    cluster_ids: list[int] = []
    next_cluster = 0
    for ticker in scored["ticker"].astype(str):
        assigned: int | None = None
        for cluster_id, representative_ticker in representatives:
            corr_value = float("nan")
            if (
                ticker in corr_matrix.index
                and representative_ticker in corr_matrix.columns
            ):
                corr_value = float(corr_matrix.at[ticker, representative_ticker])
            if (
                np.isfinite(corr_value)
                and corr_value >= config.cluster_corr_threshold
            ):
                assigned = cluster_id
                break
        if assigned is None:
            assigned = next_cluster
            representatives.append((assigned, ticker))
            next_cluster += 1
        cluster_ids.append(assigned)

    scored["cluster_id"] = cluster_ids
    return scored


def shortlist(scored: pd.DataFrame, *, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggressive = (
        scored.loc[scored["bucket"] == "EQUITY"]
        .sort_values(["screen_score", "momentum_score"], ascending=False)
        .head(top_n)
        .copy()
    )
    defensive = (
        scored.loc[scored["bucket"].isin(["DEFENSIVE", "REAL_ASSET"])]
        .sort_values(["screen_score", "momentum_score"], ascending=False)
        .head(top_n)
        .copy()
    )
    return aggressive, defensive


def save_monthly_snapshot(listing: pd.DataFrame, root: Path) -> Path:
    snapshot_dir = root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{date.today():%Y-%m}.csv"
    listing.to_csv(path, index=False)
    return path
