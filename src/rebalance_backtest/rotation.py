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
    market_score: float
    best_stock_score: float
    bond_score: float
    safe_ticker: str


@dataclass
class AdaptiveRotationStrategy:
    """Hierarchical stock -> bond -> safe ETF rotation with staged execution.

    v0.3 deliberately separates two jobs:
    1) market regime: broad market + median sector breadth decides whether stock
       risk is attractive versus the safe asset;
    2) security selection: only after a STOCK regime is confirmed do sector
       scores decide which stock ETFs receive the satellite allocation.

    Regime changes use hysteresis. Risk is added slowly, cut faster, and sector
    switches have their own transfer cap. A tolerance band suppresses small
    rebalances. The safe ETF absorbs all residual weight during transitions.
    """

    stock_tickers: Sequence[str]
    bond_tickers: Sequence[str]
    safe_ticker: str
    market_ticker: str | None = None
    lookbacks: tuple[int, ...] = (63, 126, 252)
    lookback_weights: tuple[float, ...] = (0.20, 0.30, 0.50)
    vol_window: int = 63
    market_weight: float = 0.60
    sector_breadth_weight: float = 0.40
    stock_enter_threshold: float = 0.15
    stock_exit_threshold: float = 0.00
    bond_enter_threshold: float = 0.10
    bond_exit_threshold: float = 0.00
    no_trade_band: float = 0.05
    risk_on_step: float = 0.05
    risk_off_step: float = 0.15
    sector_step: float = 0.05
    stock_exposure: float = 0.85
    bond_exposure: float = 0.85
    market_core_weight: float = 0.30
    sector_max_weight: float = 0.30
    stock_top_n: int = 2
    bond_top_n: int = 1
    score_clip: float = 3.0

    def __post_init__(self) -> None:
        self.stock_tickers = list(dict.fromkeys(self.stock_tickers))
        self.bond_tickers = list(dict.fromkeys(self.bond_tickers))
        if not self.stock_tickers:
            raise ValueError("stock_tickers cannot be empty")
        if not self.bond_tickers:
            raise ValueError("bond_tickers cannot be empty")
        if self.market_ticker is None:
            self.market_ticker = self.stock_tickers[0]
        if self.market_ticker not in self.stock_tickers:
            raise ValueError("market_ticker must be included in stock_tickers")
        if self.safe_ticker in set(self.stock_tickers) | set(self.bond_tickers):
            raise ValueError("safe_ticker must be separate from stock/bond buckets")
        if len(self.lookbacks) != len(self.lookback_weights):
            raise ValueError("lookbacks and lookback_weights must have same length")
        if not np.isclose(sum(self.lookback_weights), 1.0):
            raise ValueError("lookback_weights must sum to 1")
        if not np.isclose(self.market_weight + self.sector_breadth_weight, 1.0):
            raise ValueError("market/breadth weights must sum to 1")
        if min(self.lookbacks) <= 0 or self.vol_window <= 1:
            raise ValueError("lookbacks and vol_window must be positive")
        for name, value in {
            "no_trade_band": self.no_trade_band,
            "risk_on_step": self.risk_on_step,
            "risk_off_step": self.risk_off_step,
            "sector_step": self.sector_step,
            "stock_exposure": self.stock_exposure,
            "bond_exposure": self.bond_exposure,
            "market_core_weight": self.market_core_weight,
            "sector_max_weight": self.sector_max_weight,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.risk_on_step <= 0 or self.risk_off_step <= 0 or self.sector_step <= 0:
            raise ValueError("step sizes must be positive")
        if self.market_core_weight > self.stock_exposure:
            raise ValueError("market_core_weight cannot exceed stock_exposure")

    @property
    def universe(self) -> list[str]:
        return list(dict.fromkeys([*self.stock_tickers, *self.bond_tickers, self.safe_ticker]))

    @property
    def sector_tickers(self) -> list[str]:
        return [ticker for ticker in self.stock_tickers if ticker != self.market_ticker]

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
        daily_vol = log_daily.tail(self.vol_window).std(ddof=1).clip(lower=0.0025)
        safe = prices[self.safe_ticker]

        scores = pd.Series(0.0, index=self.universe, dtype=float)
        for h, coeff in zip(self.lookbacks, self.lookback_weights):
            asset_log_return = np.log(prices.iloc[-1] / prices.iloc[-(h + 1)])
            safe_log_return = float(np.log(safe.iloc[-1] / safe.iloc[-(h + 1)]))
            excess = asset_log_return - safe_log_return
            scores += coeff * (excess / (daily_vol * np.sqrt(h)))

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

    def _market_score(self, scores: pd.Series) -> float:
        broad = float(scores[self.market_ticker])
        sectors = scores.reindex(self.sector_tickers).dropna()
        breadth = float(sectors.median()) if len(sectors) else broad
        return self.market_weight * broad + self.sector_breadth_weight * breadth

    def _choose_regime(
        self, scores: pd.Series, current_weights: pd.Series
    ) -> tuple[str, float, float, float]:
        market_score = self._market_score(scores)
        best_stock_score = float(scores.reindex(self.stock_tickers).max())
        bond_score = float(scores.reindex(self.bond_tickers).max())
        current_bucket = self._current_bucket(
            current_weights, self.stock_tickers, self.bond_tickers, self.safe_ticker
        )

        if current_bucket == "STOCK":
            if market_score > self.stock_exit_threshold:
                return "STOCK", market_score, best_stock_score, bond_score
            if bond_score > self.bond_enter_threshold:
                return "BOND", market_score, best_stock_score, bond_score
            return "SAFE", market_score, best_stock_score, bond_score

        if current_bucket == "BOND":
            if market_score > self.stock_enter_threshold:
                return "STOCK", market_score, best_stock_score, bond_score
            if bond_score > self.bond_exit_threshold:
                return "BOND", market_score, best_stock_score, bond_score
            return "SAFE", market_score, best_stock_score, bond_score

        if market_score > self.stock_enter_threshold:
            return "STOCK", market_score, best_stock_score, bond_score
        if bond_score > self.bond_enter_threshold:
            return "BOND", market_score, best_stock_score, bond_score
        return "SAFE", market_score, best_stock_score, bond_score

    @staticmethod
    def _equal_weight_top(scores: pd.Series, tickers: Sequence[str], n: int) -> pd.Series:
        subset = scores.reindex(tickers).sort_values(ascending=False)
        selected = subset.head(max(1, min(n, len(subset)))).index
        weights = pd.Series(0.0, index=tickers, dtype=float)
        weights.loc[selected] = 1.0 / len(selected)
        return weights

    def _ideal_target(self, regime: str, scores: pd.Series, columns: Sequence[str]) -> pd.Series:
        target = pd.Series(0.0, index=columns, dtype=float)
        if regime == "SAFE":
            target[self.safe_ticker] = 1.0
            return target

        if regime == "BOND":
            within = self._equal_weight_top(scores, self.bond_tickers, self.bond_top_n)
            target.loc[within.index] = within * self.bond_exposure
            target[self.safe_ticker] = 1.0 - self.bond_exposure
            return target

        core = min(self.market_core_weight, self.stock_exposure)
        target[self.market_ticker] = core
        remaining = self.stock_exposure - core
        if self.sector_tickers and remaining > 0:
            ranked = scores.reindex(self.sector_tickers).sort_values(ascending=False)
            selected = list(ranked.head(max(1, min(self.stock_top_n, len(ranked)))).index)
            per_sector = min(self.sector_max_weight, remaining / len(selected))
            for ticker in selected:
                target[ticker] = per_sector
            target[self.safe_ticker] = 1.0 - float(target.drop(labels=[self.safe_ticker]).sum())
        else:
            target[self.safe_ticker] = 1.0 - core
        return target

    def _adjust_bucket(
        self,
        current: pd.Series,
        tickers: Sequence[str],
        desired: pd.Series,
        *,
        allow_rotation: bool,
    ) -> pd.Series:
        tickers = list(tickers)
        cur = current.reindex(tickers).fillna(0.0).astype(float)
        des = desired.reindex(tickers).fillna(0.0).astype(float)
        cur_total = float(cur.sum())
        des_total = float(des.sum())
        out = cur.copy()

        gap = des_total - cur_total
        if gap > self.no_trade_band:
            add = min(self.risk_on_step, gap)
            if des_total > 0:
                out += add * (des / des_total)
        elif gap < -self.no_trade_band:
            cut = min(self.risk_off_step, -gap)
            if cur_total > 0:
                out *= max(0.0, (cur_total - cut) / cur_total)

        if allow_rotation:
            total = float(out.sum())
            if total > 0 and des_total > 0:
                desired_comp = des / des_total
                desired_abs = desired_comp * total
                comp_gap = desired_abs - out
                under = comp_gap.clip(lower=0.0)
                over = -comp_gap.clip(upper=0.0)
                transferable = min(self.sector_step, float(under.sum()), float(over.sum()))
                if (
                    transferable > 1e-12
                    and float(comp_gap.abs().max()) > self.no_trade_band
                    and under.sum() > 0
                    and over.sum() > 0
                ):
                    out -= transferable * (over / over.sum())
                    out += transferable * (under / under.sum())

        return out.clip(lower=0.0)

    def _apply_staged_transition(
        self,
        ideal: pd.Series,
        current: pd.Series,
        regime: str,
    ) -> pd.Series:
        current = current.reindex(ideal.index).fillna(0.0).astype(float)
        current_bucket = self._current_bucket(
            current, self.stock_tickers, self.bond_tickers, self.safe_ticker
        )

        stock_next = self._adjust_bucket(
            current,
            self.stock_tickers,
            ideal,
            allow_rotation=(regime == "STOCK" and current_bucket == "STOCK"),
        )
        bond_next = self._adjust_bucket(
            current,
            self.bond_tickers,
            ideal,
            allow_rotation=(regime == "BOND" and current_bucket == "BOND"),
        )

        proposed = pd.Series(0.0, index=ideal.index, dtype=float)
        proposed.loc[self.stock_tickers] = stock_next
        proposed.loc[self.bond_tickers] = bond_next
        risky_total = float(proposed.drop(labels=[self.safe_ticker], errors="ignore").sum())
        if risky_total > 1.0:
            risky = [t for t in proposed.index if t != self.safe_ticker]
            proposed.loc[risky] *= 1.0 / risky_total
            risky_total = 1.0
        proposed[self.safe_ticker] = 1.0 - risky_total
        return proposed.clip(lower=0.0)

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
                market_score=float("nan"),
                best_stock_score=float("nan"),
                bond_score=float("nan"),
                safe_ticker=self.safe_ticker,
            )

        regime, market_score, best_stock_score, bond_score = self._choose_regime(
            scores, current_weights
        )
        ideal = self._ideal_target(regime, scores, columns)
        target = (
            self._apply_staged_transition(ideal, current_weights, regime)
            if apply_no_trade_band
            else ideal
        )

        return RotationRecommendation(
            regime=regime,
            target_weights=target,
            scores=scores,
            market_score=market_score,
            best_stock_score=best_stock_score,
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
