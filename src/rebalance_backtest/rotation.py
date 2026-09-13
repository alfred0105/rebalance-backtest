from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RotationRecommendation:
    regime: str
    target_weights: pd.Series
    scores: pd.Series
    stock_score: float
    bond_score: float
    safe_ticker: str


@dataclass
class AdaptiveRotationStrategy:
    """Hierarchical stock -> bond -> safe-asset rotation.

    Each risky asset is scored against a designated safe ETF. The default score
    blends 3/6/12-month volatility-normalized excess log returns, putting the
    largest weight on the 12-month horizon. If stocks no longer beat the safe
    asset, capital rotates to bonds; if bonds also fail the hurdle, the strategy
    moves to the safe ETF.
    """

    stock_tickers: Sequence[str]
    bond_tickers: Sequence[str]
    safe_ticker: str
    lookbacks: tuple[int, ...] = (63, 126, 252)
    lookback_weights: tuple[float, ...] = (0.20, 0.30, 0.50)
    vol_window: int = 63
    absolute_threshold: float = 0.05
    switch_margin: float = 0.10
    no_trade_band: float = 0.05
    stock_exposure: float = 0.95
    bond_exposure: float = 0.95
    stock_top_n: int = 2
    bond_top_n: int = 1
    score_clip: float = 3.0

    def __post_init__(self) -> None:
        if not self.stock_tickers:
            raise ValueError("stock_tickers cannot be empty")
        if not self.bond_tickers:
            raise ValueError("bond_tickers cannot be empty")
        if self.safe_ticker in set(self.stock_tickers) | set(self.bond_tickers):
            raise ValueError("safe_ticker must be separate from stock/bond buckets")
        if len(self.lookbacks) != len(self.lookback_weights):
            raise ValueError("lookbacks and lookback_weights must have same length")
        if not np.isclose(sum(self.lookback_weights), 1.0):
            raise ValueError("lookback_weights must sum to 1")
        if min(self.lookbacks) <= 0 or self.vol_window <= 1:
            raise ValueError("lookbacks and vol_window must be positive")
        if not (0 <= self.no_trade_band <= 1):
            raise ValueError("no_trade_band must be between 0 and 1")
        if not (0 <= self.stock_exposure <= 1 and 0 <= self.bond_exposure <= 1):
            raise ValueError("bucket exposures must be between 0 and 1")

    @property
    def universe(self) -> list[str]:
        return list(dict.fromkeys([*self.stock_tickers, *self.bond_tickers, self.safe_ticker]))

    def initial_weights(self, columns: Sequence[str]) -> pd.Series:
        weights = pd.Series(0.0, index=columns, dtype=float)
        if self.safe_ticker not in weights.index:
            raise ValueError(f"safe ticker {self.safe_ticker!r} is missing from price columns")
        weights[self.safe_ticker] = 1.0
        return weights

    def _scores(self, price_history: pd.DataFrame) -> pd.Series | None:
        required = max(max(self.lookbacks) + 1, self.vol_window + 1)
        if len(price_history) < required:
            return None
        missing = [ticker for ticker in self.universe if ticker not in price_history.columns]
        if missing:
            raise ValueError(f"Missing rotation tickers: {missing}")

        prices = price_history[self.universe].astype(float)
        log_daily = np.log(prices / prices.shift(1))
        daily_vol = log_daily.tail(self.vol_window).std(ddof=1)
        vol_floor = 0.0025
        daily_vol = daily_vol.clip(lower=vol_floor)

        safe = prices[self.safe_ticker]
        scores = pd.Series(0.0, index=self.universe, dtype=float)
        for h, coeff in zip(self.lookbacks, self.lookback_weights):
            asset_log_return = np.log(prices.iloc[-1] / prices.iloc[-(h + 1)])
            safe_log_return = float(np.log(safe.iloc[-1] / safe.iloc[-(h + 1)]))
            excess = asset_log_return - safe_log_return
            horizon_vol = daily_vol * np.sqrt(h)
            scores += coeff * (excess / horizon_vol)

        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scores[self.safe_ticker] = 0.0
        return scores.clip(lower=-self.score_clip, upper=self.score_clip)

    @staticmethod
    def _current_bucket(
        current: pd.Series,
        stock_tickers: Sequence[str],
        bond_tickers: Sequence[str],
        safe_ticker: str,
    ) -> str:
        stock = float(current.reindex(stock_tickers).fillna(0.0).sum())
        bond = float(current.reindex(bond_tickers).fillna(0.0).sum())
        safe = float(current.get(safe_ticker, 0.0))
        if stock >= max(bond, safe):
            return "STOCK"
        if bond >= max(stock, safe):
            return "BOND"
        return "SAFE"

    def _choose_regime(self, scores: pd.Series, current_weights: pd.Series) -> tuple[str, float, float]:
        stock_score = float(scores.reindex(self.stock_tickers).max())
        bond_score = float(scores.reindex(self.bond_tickers).max())
        current_bucket = self._current_bucket(
            current_weights, self.stock_tickers, self.bond_tickers, self.safe_ticker
        )

        stock_ok = stock_score > self.absolute_threshold
        bond_ok = bond_score > self.absolute_threshold

        if not stock_ok and not bond_ok:
            return "SAFE", stock_score, bond_score
        if stock_ok and not bond_ok:
            return "STOCK", stock_score, bond_score
        if bond_ok and not stock_ok:
            return "BOND", stock_score, bond_score

        if abs(stock_score - bond_score) < self.switch_margin:
            if current_bucket in {"STOCK", "BOND"}:
                return current_bucket, stock_score, bond_score

        if stock_score >= bond_score + self.switch_margin:
            return "STOCK", stock_score, bond_score
        if bond_score >= stock_score + self.switch_margin:
            return "BOND", stock_score, bond_score
        return ("STOCK" if stock_score >= bond_score else "BOND"), stock_score, bond_score

    @staticmethod
    def _equal_weight_top(scores: pd.Series, tickers: Sequence[str], n: int) -> pd.Series:
        subset = scores.reindex(tickers).sort_values(ascending=False)
        selected = subset.head(max(1, min(n, len(subset)))).index
        weights = pd.Series(0.0, index=tickers, dtype=float)
        weights.loc[selected] = 1.0 / len(selected)
        return weights

    def recommend(
        self,
        price_history: pd.DataFrame,
        current_weights: pd.Series | None = None,
        *,
        apply_no_trade_band: bool = False,
    ) -> RotationRecommendation:
        columns = price_history.columns
        if current_weights is None:
            current_weights = self.initial_weights(columns)
        else:
            current_weights = current_weights.reindex(columns).fillna(0.0).astype(float)

        scores = self._scores(price_history)
        if scores is None:
            return RotationRecommendation(
                regime="WARMUP",
                target_weights=current_weights.copy(),
                scores=pd.Series(np.nan, index=self.universe, dtype=float),
                stock_score=float("nan"),
                bond_score=float("nan"),
                safe_ticker=self.safe_ticker,
            )

        regime, stock_score, bond_score = self._choose_regime(scores, current_weights)
        target = pd.Series(0.0, index=columns, dtype=float)

        if regime == "STOCK":
            within = self._equal_weight_top(scores, self.stock_tickers, self.stock_top_n)
            target.loc[within.index] = within * self.stock_exposure
            target[self.safe_ticker] = 1.0 - self.stock_exposure
        elif regime == "BOND":
            within = self._equal_weight_top(scores, self.bond_tickers, self.bond_top_n)
            target.loc[within.index] = within * self.bond_exposure
            target[self.safe_ticker] = 1.0 - self.bond_exposure
        else:
            target[self.safe_ticker] = 1.0

        if apply_no_trade_band:
            max_delta = float((target - current_weights).abs().max())
            if max_delta < self.no_trade_band:
                target = current_weights.copy()

        return RotationRecommendation(
            regime=regime,
            target_weights=target,
            scores=scores,
            stock_score=stock_score,
            bond_score=bond_score,
            safe_ticker=self.safe_ticker,
        )

    def target_weights(
        self,
        price_history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        return self.recommend(
            price_history,
            current_weights,
            apply_no_trade_band=True,
        ).target_weights
