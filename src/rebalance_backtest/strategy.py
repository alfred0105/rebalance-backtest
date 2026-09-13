from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _aligned_weights(
    weights: Mapping[str, float], columns: Sequence[str]
) -> pd.Series:
    series = pd.Series(weights, dtype=float).reindex(columns).fillna(0.0)
    if (series < 0).any():
        raise ValueError("This v1 engine supports long-only weights.")
    if series.sum() <= 0:
        raise ValueError("Base weights must have positive total exposure.")
    if series.sum() > 1.0 + 1e-9:
        raise ValueError("Base risky weights must sum to <= 1.0.")
    return series


@dataclass
class FixedWeightStrategy:
    """Rebalance back to fixed risky-asset weights on each evaluation date."""

    base_weights: Mapping[str, float]

    def initial_weights(self, columns: Sequence[str]) -> pd.Series:
        return _aligned_weights(self.base_weights, columns)

    def target_weights(
        self,
        price_history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        return self.initial_weights(price_history.columns)


@dataclass
class MomentumTiltStrategy:
    """Asymmetric, volatility-normalized, nonlinear momentum overlay.

    Signal for asset i and lookback h:
        z(i,h) = log(P_t / P_{t-h}) / (daily_vol_i * sqrt(h))

    The lookback signals are blended and squashed with tanh. Positive signals
    can be amplified more than negative signals. The signal tilts the base
    composition, while the portfolio-level signal may reduce overall risky
    exposure during weak regimes. Entry and exit speeds are intentionally
    asymmetric so de-risking can be faster than adding exposure.
    """

    base_weights: Mapping[str, float]
    lookbacks: tuple[int, ...] = (20, 60, 120)
    lookback_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    vol_window: int = 60
    tanh_k: float = 1.25
    positive_multiplier: float = 1.25
    negative_multiplier: float = 0.85
    tilt_strength: float = 0.50
    min_relative_weight: float = 0.50
    max_relative_weight: float = 1.50
    min_risk_exposure: float = 0.55
    max_risk_exposure: float = 1.00
    regime_strength: float = 0.35
    entry_speed: float = 0.70
    exit_speed: float = 1.00
    no_trade_band: float = 0.015

    def __post_init__(self) -> None:
        if len(self.lookbacks) != len(self.lookback_weights):
            raise ValueError("lookbacks and lookback_weights must have same length")
        if any(h <= 0 for h in self.lookbacks):
            raise ValueError("lookbacks must be positive")
        if not np.isclose(sum(self.lookback_weights), 1.0):
            raise ValueError("lookback_weights must sum to 1.0")
        if not (0 <= self.entry_speed <= 1 and 0 <= self.exit_speed <= 1):
            raise ValueError("entry_speed and exit_speed must be within [0, 1]")
        if not (0 <= self.min_risk_exposure <= self.max_risk_exposure <= 1.0):
            raise ValueError("risk exposure bounds must satisfy 0 <= min <= max <= 1")

    def initial_weights(self, columns: Sequence[str]) -> pd.Series:
        return _aligned_weights(self.base_weights, columns)

    def _signal(self, price_history: pd.DataFrame) -> pd.Series | None:
        required = max(max(self.lookbacks) + 1, self.vol_window + 1)
        if len(price_history) < required:
            return None

        prices = price_history.astype(float)
        log_daily = np.log(prices / prices.shift(1))
        daily_vol = log_daily.tail(self.vol_window).std(ddof=1)
        daily_vol = daily_vol.replace(0.0, np.nan)

        z = pd.Series(0.0, index=prices.columns)
        for h, coeff in zip(self.lookbacks, self.lookback_weights):
            log_momentum = np.log(prices.iloc[-1] / prices.iloc[-(h + 1)])
            horizon_vol = daily_vol * np.sqrt(h)
            z_h = log_momentum / horizon_vol
            z = z.add(coeff * z_h, fill_value=0.0)

        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bounded = np.tanh(self.tanh_k * z)
        asymmetric = bounded.where(
            bounded < 0,
            bounded * self.positive_multiplier,
        )
        asymmetric = asymmetric.where(
            bounded >= 0,
            bounded * self.negative_multiplier,
        )
        return asymmetric

    def target_weights(
        self,
        price_history: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        columns = price_history.columns
        base = self.initial_weights(columns)
        current = current_weights.reindex(columns).fillna(0.0).astype(float)
        signal = self._signal(price_history)
        if signal is None:
            return current

        base_exposure = float(base.sum())
        base_composition = base / base_exposure

        factor = 1.0 + self.tilt_strength * signal
        factor = factor.clip(
            lower=self.min_relative_weight,
            upper=self.max_relative_weight,
        )

        composition = base_composition * factor
        composition = composition / composition.sum()

        regime_signal = float((base_composition * signal).sum())
        desired_exposure = base_exposure + self.regime_strength * regime_signal
        desired_exposure = float(
            np.clip(
                desired_exposure,
                self.min_risk_exposure,
                self.max_risk_exposure,
            )
        )

        desired = composition * desired_exposure
        delta = desired - current
        speed = pd.Series(
            np.where(delta >= 0, self.entry_speed, self.exit_speed),
            index=columns,
            dtype=float,
        )
        adjusted = current + speed * delta

        gross = float(adjusted.sum())
        if gross > self.max_risk_exposure and gross > 0:
            adjusted *= self.max_risk_exposure / gross

        if float((adjusted - current).abs().max()) < self.no_trade_band:
            return current

        return adjusted.clip(lower=0.0)
