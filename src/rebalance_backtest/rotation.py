from __future__ import annotations

from dataclasses import dataclass, field
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
    short_return: float
    emergency_brake: bool
    stock_target: float
    bond_target: float
    safe_target: float


@dataclass
class AdaptiveRotationStrategy:
    """Daily monitored stock -> bond -> safe ETF rotation.

    Long-horizon momentum determines desired stock exposure continuously. Bond
    momentum decides how much residual capital should rotate into bonds; the
    rest stays in the safe ETF. A short-horizon crash brake can cap stock risk
    quickly.

    Signals are evaluated daily, but ordinary trades use tolerance-band edge
    rebalancing rather than chasing the exact target. Small whole-portfolio
    trades are suppressed. Emergency de-risking ignores those normal frictions.
    Sector selection is refreshed only once per calendar month.
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

    no_trade_band: float = 0.05
    min_trade_turnover: float = 0.02
    risk_on_step: float = 0.05
    risk_off_step: float = 0.20
    emergency_step: float = 0.35
    sector_step: float = 0.05

    max_stock_exposure: float = 0.90
    market_core_fraction: float = 0.50
    sector_max_weight: float = 0.25
    stock_top_n: int = 2
    bond_top_n: int = 1

    brake_lookback: int = 20
    brake_threshold: float = -0.08
    brake_stock_cap: float = 0.35
    bond_score_full: float = 0.30
    bond_max_fraction_of_residual: float = 0.90
    score_clip: float = 3.0

    stock_score_points: tuple[float, ...] = (-0.30, -0.15, 0.00, 0.15, 0.30, 0.60)
    stock_target_points: tuple[float, ...] = (0.00, 0.15, 0.35, 0.60, 0.75, 0.90)

    _selected_sectors: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _sector_month: tuple[int, int] | None = field(default=None, init=False, repr=False)

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
        if len(self.stock_score_points) != len(self.stock_target_points):
            raise ValueError("stock score/target point counts must match")
        if any(b <= a for a, b in zip(self.stock_score_points, self.stock_score_points[1:])):
            raise ValueError("stock_score_points must be strictly increasing")
        if min(self.lookbacks) <= 0 or self.vol_window <= 1 or self.brake_lookback <= 0:
            raise ValueError("lookbacks and windows must be positive")
        for name, value in {
            "no_trade_band": self.no_trade_band,
            "min_trade_turnover": self.min_trade_turnover,
            "risk_on_step": self.risk_on_step,
            "risk_off_step": self.risk_off_step,
            "emergency_step": self.emergency_step,
            "sector_step": self.sector_step,
            "max_stock_exposure": self.max_stock_exposure,
            "market_core_fraction": self.market_core_fraction,
            "sector_max_weight": self.sector_max_weight,
            "brake_stock_cap": self.brake_stock_cap,
            "bond_max_fraction_of_residual": self.bond_max_fraction_of_residual,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if min(self.risk_on_step, self.risk_off_step, self.emergency_step, self.sector_step) <= 0:
            raise ValueError("step sizes must be positive")
        if self.bond_score_full <= 0:
            raise ValueError("bond_score_full must be positive")

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
        required = max(max(self.lookbacks) + 1, self.vol_window + 1, self.brake_lookback + 1)
        if len(price_history) < required:
            return None

        universe = self.universe
        missing = [ticker for ticker in universe if ticker not in price_history.columns]
        if missing:
            raise ValueError(f"Missing rotation tickers: {missing}")

        # Only the latest signal window is needed. The old implementation
        # repeatedly recalculated returns over the entire expanding history on
        # every trading day, turning a daily backtest into unnecessary O(N^2)
        # work. Keep the math identical but operate on a fixed NumPy window.
        values = (
            price_history.loc[:, universe]
            .iloc[-required:]
            .to_numpy(dtype=float, copy=False)
        )
        log_values = np.log(values)
        log_daily = np.diff(log_values, axis=0)
        daily_vol = np.std(log_daily[-self.vol_window :], axis=0, ddof=1)
        daily_vol = np.maximum(daily_vol, 0.0025)

        safe_idx = universe.index(self.safe_ticker)
        score_values = np.zeros(len(universe), dtype=float)

        for h, coeff in zip(self.lookbacks, self.lookback_weights):
            asset_log_return = log_values[-1] - log_values[-(h + 1)]
            safe_log_return = float(asset_log_return[safe_idx])
            excess = asset_log_return - safe_log_return
            score_values += coeff * (excess / (daily_vol * np.sqrt(h)))

        score_values = np.nan_to_num(
            score_values,
            nan=0.0,
            posinf=self.score_clip,
            neginf=-self.score_clip,
        )
        score_values[safe_idx] = 0.0
        score_values = np.clip(score_values, -self.score_clip, self.score_clip)
        return pd.Series(score_values, index=universe, dtype=float)

    def _market_score(self, scores: pd.Series) -> float:
        broad = float(scores[self.market_ticker])
        sectors = scores.reindex(self.sector_tickers).dropna()
        breadth = float(sectors.median()) if len(sectors) else broad
        return self.market_weight * broad + self.sector_breadth_weight * breadth

    def _short_return(self, price_history: pd.DataFrame) -> float:
        if len(price_history) <= self.brake_lookback:
            return 0.0
        market = price_history[self.market_ticker]
        return float(market.iloc[-1] / market.iloc[-(self.brake_lookback + 1)] - 1.0)

    def _stock_target_from_score(self, market_score: float) -> float:
        raw = float(
            np.interp(
                market_score,
                np.asarray(self.stock_score_points, dtype=float),
                np.asarray(self.stock_target_points, dtype=float),
            )
        )
        return float(np.clip(raw, 0.0, self.max_stock_exposure))

    def _allocation_targets(
        self,
        market_score: float,
        bond_score: float,
        short_return: float,
    ) -> tuple[float, float, float, bool]:
        stock_target = self._stock_target_from_score(market_score)
        emergency = short_return <= self.brake_threshold
        if emergency:
            stock_target = min(stock_target, self.brake_stock_cap)

        residual = max(0.0, 1.0 - stock_target)
        bond_strength = float(np.clip(bond_score / self.bond_score_full, 0.0, 1.0))
        bond_target = residual * self.bond_max_fraction_of_residual * bond_strength
        safe_target = 1.0 - stock_target - bond_target
        return stock_target, bond_target, safe_target, emergency

    def _refresh_monthly_sectors(self, scores: pd.Series, date: pd.Timestamp) -> tuple[str, ...]:
        month_key = (date.year, date.month)
        if self._sector_month != month_key or not self._selected_sectors:
            ranked = scores.reindex(self.sector_tickers).sort_values(ascending=False)
            n = max(1, min(self.stock_top_n, len(ranked))) if len(ranked) else 0
            self._selected_sectors = tuple(ranked.head(n).index)
            self._sector_month = month_key
        return self._selected_sectors

    def _ideal_target(
        self,
        scores: pd.Series,
        columns: Sequence[str],
        date: pd.Timestamp,
        stock_target: float,
        bond_target: float,
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

        if bond_target > 0:
            ranked_bonds = scores.reindex(self.bond_tickers).sort_values(ascending=False)
            selected_bonds = ranked_bonds.head(max(1, min(self.bond_top_n, len(ranked_bonds)))).index
            if len(selected_bonds):
                target.loc[selected_bonds] = bond_target / len(selected_bonds)

        target[self.safe_ticker] = 1.0 - float(target.drop(labels=[self.safe_ticker]).sum())
        return target.clip(lower=0.0)

    def _adjust_bucket(
        self,
        current: pd.Series,
        tickers: Sequence[str],
        desired: pd.Series,
        *,
        add_step: float,
        cut_step: float,
        allow_rotation: bool,
        band: float | None = None,
    ) -> pd.Series:
        tickers = list(tickers)
        cur = current.reindex(tickers).fillna(0.0).astype(float)
        des = desired.reindex(tickers).fillna(0.0).astype(float)
        cur_total = float(cur.sum())
        des_total = float(des.sum())
        out = cur.copy()
        tolerance = self.no_trade_band if band is None else float(band)

        gap = des_total - cur_total
        if gap > tolerance:
            add = min(add_step, gap - tolerance)
            if add > 0 and des_total > 0:
                out += add * (des / des_total)
        elif gap < -tolerance:
            cut = min(cut_step, -gap - tolerance)
            if cut > 0 and cur_total > 0:
                out *= max(0.0, (cur_total - cut) / cur_total)

        if allow_rotation:
            total = float(out.sum())
            if total > 0 and des_total > 0:
                desired_abs = (des / des_total) * total
                comp_gap = desired_abs - out
                under = comp_gap.clip(lower=0.0)
                over = -comp_gap.clip(upper=0.0)
                excess = max(0.0, float(comp_gap.abs().max()) - tolerance)
                transferable = min(
                    self.sector_step,
                    excess,
                    float(under.sum()),
                    float(over.sum()),
                )
                if transferable > 1e-12 and under.sum() > 0 and over.sum() > 0:
                    out -= transferable * (over / over.sum())
                    out += transferable * (under / under.sum())

        return out.clip(lower=0.0)

    def _apply_staged_transition(
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
        bond_next = self._adjust_bucket(
            current,
            self.bond_tickers,
            ideal,
            add_step=self.risk_on_step,
            cut_step=self.risk_off_step,
            allow_rotation=False,
            band=self.no_trade_band,
        )

        proposed = pd.Series(0.0, index=ideal.index, dtype=float)
        proposed.loc[self.stock_tickers] = stock_next
        proposed.loc[self.bond_tickers] = bond_next
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

    @staticmethod
    def _regime_label(stock_target: float, emergency: bool) -> str:
        if emergency:
            return "BRAKE"
        if stock_target >= 0.70:
            return "RISK_ON"
        if stock_target >= 0.35:
            return "BALANCED"
        if stock_target > 0:
            return "DEFENSIVE"
        return "SAFE"

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
                short_return=float("nan"),
                emergency_brake=False,
                stock_target=float("nan"),
                bond_target=float("nan"),
                safe_target=float("nan"),
            )

        market_score = self._market_score(scores)
        best_stock_score = float(scores.reindex(self.stock_tickers).max())
        bond_score = float(scores.reindex(self.bond_tickers).max())
        short_return = self._short_return(price_history)
        stock_target, bond_target, safe_target, emergency = self._allocation_targets(
            market_score, bond_score, short_return
        )

        date = pd.Timestamp(price_history.index[-1])
        ideal = self._ideal_target(scores, columns, date, stock_target, bond_target)
        target = (
            self._apply_staged_transition(ideal, current_weights, emergency=emergency)
            if apply_no_trade_band
            else ideal
        )

        return RotationRecommendation(
            regime=self._regime_label(stock_target, emergency),
            target_weights=target,
            scores=scores,
            market_score=market_score,
            best_stock_score=best_stock_score,
            bond_score=bond_score,
            safe_ticker=self.safe_ticker,
            short_return=short_return,
            emergency_brake=emergency,
            stock_target=stock_target,
            bond_target=bond_target,
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
