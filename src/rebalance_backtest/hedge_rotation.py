from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .rotation import AdaptiveRotationStrategy


@dataclass(frozen=True)
class HedgeRotationRecommendation:
    regime: str
    target_weights: pd.Series
    scores: pd.Series
    market_score: float
    defensive_score: float
    short_return: float
    peak_drawdown: float
    peak_protection: bool
    emergency_brake: bool
    stock_target: float
    defensive_target: float
    safe_target: float


@dataclass
class PeakHedgeRotationStrategy(AdaptiveRotationStrategy):
    """Adaptive stock allocation with peak protection and rotating hedge assets."""

    hedge_tickers: Sequence[str] = ()
    peak_window: int = 60
    peak_trigger: float = -0.04
    peak_caps: tuple[tuple[float, float], ...] = (
        (-0.10, 0.35),
        (-0.08, 0.50),
        (-0.06, 0.65),
        (-0.04, 0.80),
    )
    peak_short_confirm: float = 0.0
    peak_breadth_confirm: float = 0.0
    enable_peak_protection: bool = True
    enable_hedge_rotation: bool = True

    defensive_top_n: int = 2
    defensive_max_fraction_of_residual: float = 0.90
    defensive_max_weight: float = 0.35
    defensive_step: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hedge_tickers = list(dict.fromkeys(self.hedge_tickers))
        overlap = set(self.hedge_tickers) & (
            set(self.stock_tickers) | set(self.bond_tickers) | {self.safe_ticker}
        )
        if overlap:
            raise ValueError(f"hedge_tickers overlap another bucket: {sorted(overlap)}")
        if self.peak_window <= 1:
            raise ValueError("peak_window must be > 1")
        if not -1.0 < self.peak_trigger < 0.0:
            raise ValueError("peak_trigger must be between -1 and 0")
        if self.defensive_top_n <= 0:
            raise ValueError("defensive_top_n must be positive")
        for name, value in {
            "defensive_max_fraction_of_residual": self.defensive_max_fraction_of_residual,
            "defensive_max_weight": self.defensive_max_weight,
            "defensive_step": self.defensive_step,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def universe(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *self.stock_tickers,
                    *self.bond_tickers,
                    *self.hedge_tickers,
                    self.safe_ticker,
                ]
            )
        )

    @property
    def defensive_tickers(self) -> list[str]:
        if self.enable_hedge_rotation:
            return list(dict.fromkeys([*self.bond_tickers, *self.hedge_tickers]))
        return list(self.bond_tickers)

    def _peak_drawdown(self, price_history: pd.DataFrame) -> float:
        market = price_history[self.market_ticker]
        window = market.iloc[-min(len(market), self.peak_window) :]
        peak = float(window.max())
        if peak <= 0:
            return 0.0
        return float(window.iloc[-1] / peak - 1.0)

    def _breadth_score(self, scores: pd.Series) -> float:
        sectors = scores.reindex(self.sector_tickers).dropna()
        if len(sectors) == 0:
            return float(scores[self.market_ticker])
        return float(sectors.median())

    def _peak_cap(self, drawdown: float) -> float:
        cap = 1.0
        # Rules are ordered from deepest drawdown to shallowest.
        for threshold, candidate_cap in self.peak_caps:
            if drawdown <= threshold:
                cap = candidate_cap
                break
        return float(cap)

    def _stock_and_defensive_targets(
        self,
        *,
        market_score: float,
        defensive_score: float,
        short_return: float,
        peak_drawdown: float,
        breadth_score: float,
    ) -> tuple[float, float, float, bool, bool]:
        stock_target = self._stock_target_from_score(market_score)

        emergency = short_return <= self.brake_threshold
        if emergency:
            stock_target = min(stock_target, self.brake_stock_cap)

        peak_active = False
        if self.enable_peak_protection and peak_drawdown <= self.peak_trigger:
            confirmation = (
                short_return <= self.peak_short_confirm
                or breadth_score <= self.peak_breadth_confirm
            )
            if confirmation:
                peak_active = True
                stock_target = min(stock_target, self._peak_cap(peak_drawdown))

        residual = max(0.0, 1.0 - stock_target)
        defensive_strength = float(
            np.clip(defensive_score / self.bond_score_full, 0.0, 1.0)
        )
        defensive_target = (
            residual
            * self.defensive_max_fraction_of_residual
            * defensive_strength
        )
        safe_target = 1.0 - stock_target - defensive_target
        return stock_target, defensive_target, safe_target, emergency, peak_active

    def _ideal_peak_hedge_target(
        self,
        scores: pd.Series,
        columns: Sequence[str],
        date: pd.Timestamp,
        stock_target: float,
        defensive_target: float,
    ) -> pd.Series:
        target = pd.Series(0.0, index=columns, dtype=float)

        if stock_target > 0:
            selected = self._refresh_monthly_sectors(scores, date)
            core = stock_target * self.market_core_fraction
            remaining = stock_target - core
            if selected and remaining > 0:
                per_sector = min(self.sector_max_weight, remaining / len(selected))
                assigned = per_sector * len(selected)
                target[self.market_ticker] = core + (remaining - assigned)
                for ticker in selected:
                    target[ticker] = per_sector
            else:
                target[self.market_ticker] = stock_target

        if defensive_target > 0:
            ranked = (
                scores.reindex(self.defensive_tickers)
                .dropna()
                .loc[lambda s: s > 0]
                .sort_values(ascending=False)
            )
            selected = ranked.head(min(self.defensive_top_n, len(ranked)))
            if len(selected):
                weights = selected.clip(lower=0.0)
                weights = weights / float(weights.sum())
                desired = weights * defensive_target
                desired = desired.clip(upper=self.defensive_max_weight)
                assigned = float(desired.sum())
                # Re-distribute leftover once among names that still have headroom.
                leftover = max(0.0, defensive_target - assigned)
                if leftover > 1e-12:
                    room = (self.defensive_max_weight - desired).clip(lower=0.0)
                    if float(room.sum()) > 0:
                        extra = leftover * room / float(room.sum())
                        desired += np.minimum(extra, room)
                target.loc[desired.index] = desired

        target[self.safe_ticker] = 1.0 - float(
            target.drop(labels=[self.safe_ticker]).sum()
        )
        return target.clip(lower=0.0)

    def _apply_peak_hedge_transition(
        self,
        ideal: pd.Series,
        current: pd.Series,
        *,
        emergency: bool,
    ) -> pd.Series:
        current = current.reindex(ideal.index).fillna(0.0).astype(float)
        stock_cut = self.emergency_step if emergency else self.risk_off_step
        stock_band = 0.0 if emergency else self.no_trade_band

        stock_next = self._adjust_bucket(
            current,
            self.stock_tickers,
            ideal,
            add_step=self.risk_on_step,
            cut_step=stock_cut,
            allow_rotation=True,
            band=stock_band,
        )

        defensive = self.defensive_tickers
        defensive_next = self._adjust_bucket(
            current,
            defensive,
            ideal,
            add_step=self.emergency_step if emergency else self.defensive_step,
            cut_step=self.defensive_step,
            allow_rotation=True,
            band=0.0 if emergency else self.no_trade_band,
        )

        proposed = pd.Series(0.0, index=ideal.index, dtype=float)
        proposed.loc[self.stock_tickers] = stock_next
        proposed.loc[defensive] = defensive_next

        risky = [ticker for ticker in proposed.index if ticker != self.safe_ticker]
        risky_total = float(proposed.loc[risky].sum())
        if risky_total > 1.0:
            proposed.loc[risky] *= 1.0 / risky_total
            risky_total = 1.0
        proposed[self.safe_ticker] = 1.0 - risky_total
        proposed = proposed.clip(lower=0.0)

        turnover = 0.5 * float((proposed - current).abs().sum())
        if not emergency and turnover < self.min_trade_turnover:
            return current.copy()
        return proposed

    def recommend(
        self,
        price_history: pd.DataFrame,
        current_weights: pd.Series | None = None,
        *,
        apply_no_trade_band: bool = False,
    ) -> HedgeRotationRecommendation:
        columns = price_history.columns
        if current_weights is None:
            current_weights = self.initial_weights(columns)
        else:
            current_weights = current_weights.reindex(columns).fillna(0.0).astype(float)

        scores = self._scores(price_history)
        if scores is None:
            return HedgeRotationRecommendation(
                regime="WARMUP",
                target_weights=current_weights.copy(),
                scores=pd.Series(np.nan, index=self.universe, dtype=float),
                market_score=float("nan"),
                defensive_score=float("nan"),
                short_return=float("nan"),
                peak_drawdown=float("nan"),
                peak_protection=False,
                emergency_brake=False,
                stock_target=float("nan"),
                defensive_target=float("nan"),
                safe_target=float("nan"),
            )

        market_score = self._market_score(scores)
        short_return = self._short_return(price_history)
        peak_drawdown = self._peak_drawdown(price_history)
        breadth_score = self._breadth_score(scores)

        defensive_scores = scores.reindex(self.defensive_tickers).dropna()
        defensive_score = (
            max(0.0, float(defensive_scores.max()))
            if len(defensive_scores)
            else 0.0
        )

        (
            stock_target,
            defensive_target,
            safe_target,
            emergency,
            peak_active,
        ) = self._stock_and_defensive_targets(
            market_score=market_score,
            defensive_score=defensive_score,
            short_return=short_return,
            peak_drawdown=peak_drawdown,
            breadth_score=breadth_score,
        )

        date = pd.Timestamp(price_history.index[-1])
        ideal = self._ideal_peak_hedge_target(
            scores,
            columns,
            date,
            stock_target,
            defensive_target,
        )
        target = (
            self._apply_peak_hedge_transition(
                ideal,
                current_weights,
                emergency=emergency,
            )
            if apply_no_trade_band
            else ideal
        )

        regime = self._regime_label(stock_target, emergency)
        if peak_active and not emergency:
            regime = "PEAK_LOCK"

        return HedgeRotationRecommendation(
            regime=regime,
            target_weights=target,
            scores=scores,
            market_score=market_score,
            defensive_score=defensive_score,
            short_return=short_return,
            peak_drawdown=peak_drawdown,
            peak_protection=peak_active,
            emergency_brake=emergency,
            stock_target=stock_target,
            defensive_target=defensive_target,
            safe_target=safe_target,
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
