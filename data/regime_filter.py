"""
Market regime classification based on volatility indicators.

For crypto markets, uses realized volatility since VIX is not available.
"""

from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

# Crypto trades 24/7, use 365 days for annualization
CRYPTO_TRADING_DAYS_PER_YEAR = 365


class MarketRegime(Enum):
    """Market volatility regime classification."""

    RISK_ON = "risk_on"  # Low volatility, trending markets
    NEUTRAL = "neutral"  # Normal volatility
    RISK_OFF = "risk_off"  # High volatility, choppy markets


class RegimeClassifier:
    """
    Classify market regime based on volatility metrics.

    Uses realized volatility percentiles since crypto markets don't have VIX.
    """

    def __init__(
        self,
        risk_off_threshold: float = 75.0,  # Percentile
        risk_on_threshold: float = 25.0,  # Percentile
        lookback_period: int = 30,  # Days for volatility calculation
    ):
        """
        Initialize regime classifier.

        Args:
            risk_off_threshold: Volatility percentile above which = RISK_OFF
            risk_on_threshold: Volatility percentile below which = RISK_ON
            lookback_period: Lookback period for volatility calculation
        """
        self.risk_off_threshold = risk_off_threshold
        self.risk_on_threshold = risk_on_threshold
        self.lookback_period = lookback_period

    def compute_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 30,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Calculate realized volatility from returns.

        Args:
            returns: Return series (fractional, not percentage)
            window: Rolling window size
            annualize: If True, annualize volatility (assumes daily returns)

        Returns:
            Realized volatility series
        """
        vol = returns.rolling(window=window).std()

        if annualize:
            vol = vol * np.sqrt(CRYPTO_TRADING_DAYS_PER_YEAR)

        return vol

    def classify_regime(
        self,
        close: pd.Series,
        window: int | None = None,
    ) -> pd.Series:
        """
        Classify market regime based on volatility percentiles.

        Args:
            close: Close price series
            window: Volatility calculation window (defaults to lookback_period)

        Returns:
            Series of MarketRegime enum values
        """
        if window is None:
            window = self.lookback_period

        # Calculate returns (fill_method=None to avoid FutureWarning)
        returns = close.pct_change(fill_method=None)

        # Calculate realized volatility
        vol = self.compute_realized_volatility(returns, window=window)

        # Calculate percentile ranks (0-100)
        vol_percentile = vol.rank(pct=True) * 100

        # Classify regime
        regime = pd.Series(index=close.index, dtype=object)
        regime[vol_percentile >= self.risk_off_threshold] = MarketRegime.RISK_OFF
        regime[vol_percentile <= self.risk_on_threshold] = MarketRegime.RISK_ON
        regime[
            (vol_percentile > self.risk_on_threshold) & (vol_percentile < self.risk_off_threshold)
        ] = MarketRegime.NEUTRAL

        # Fill NaN at start with NEUTRAL
        regime = regime.fillna(MarketRegime.NEUTRAL)

        return regime

    def get_current_regime(self, close: pd.Series) -> tuple[MarketRegime, float]:
        """
        Get current market regime and volatility.

        Args:
            close: Close price series

        Returns:
            Tuple of (regime, current_volatility)
        """
        if len(close) < self.lookback_period:
            logger.warning(
                "Insufficient data for regime classification",
                available=len(close),
                required=self.lookback_period,
            )
            return MarketRegime.NEUTRAL, 0.0

        regime_series = self.classify_regime(close)
        current_regime = regime_series.iloc[-1]

        # Calculate current volatility (fill_method=None to avoid FutureWarning)
        returns = close.pct_change(fill_method=None)
        vol_series = self.compute_realized_volatility(returns, window=self.lookback_period)
        current_vol = vol_series.iloc[-1] if not vol_series.isna().all() else 0.0

        logger.info(
            "Regime classified",
            regime=current_regime.value,
            volatility=f"{current_vol:.2%}",
        )

        return current_regime, current_vol
