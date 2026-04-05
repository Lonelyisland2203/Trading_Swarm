"""
Fee model configuration for Binance Futures USDT-M.

Models trading costs including:
- Maker/taker fees with optional BNB discount
- Funding costs for holding periods
- Slippage estimates
"""

from pydantic import BaseModel, Field


class FeeModelSettings(BaseModel):
    """
    Fee structure for Binance Futures USDT-M perpetual contracts.

    Includes entry/exit fees, funding costs, and slippage estimates.
    All fees expressed as percentages of position size.

    Attributes:
        maker_fee_pct: Maker fee as % of notional (default: 0.02% for limit orders)
        taker_fee_pct: Taker fee as % of notional (default: 0.05% for market orders)
        bnb_discount_enabled: Apply 10% BNB discount to trading fees (Binance Futures discount)
        bnb_discount_pct: BNB discount as % (default: 10% = 0.1, NOT 25% which is Spot-only)
        funding_rate_pct: Hourly funding rate as % (default: 0.01% per hour)
        funding_interval_hours: Funding payment interval (default: 8 hours per period)
        slippage_pct: Market impact/slippage estimate as % (default: 0.02%)
    """

    maker_fee_pct: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Maker fee as % of notional (limit orders, typically 0.02%)"
    )
    taker_fee_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Taker fee as % of notional (market orders, typically 0.05%)"
    )
    bnb_discount_enabled: bool = Field(
        default=True,
        description="Apply 10% BNB discount to maker and taker fees (Binance Futures only)"
    )
    bnb_discount_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="BNB discount percentage (10% = 10.0, not 25% which is Spot-only)"
    )
    funding_rate_pct: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Hourly funding rate as % of notional (default: 0.01% per hour)"
    )
    funding_interval_hours: int = Field(
        default=8,
        ge=1,
        le=24,
        description="Funding payment interval in hours (default: 8)"
    )
    slippage_pct: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Market impact/slippage estimate as % (default: 0.02%)"
    )

    def round_trip_cost_pct(self, holding_periods_8h: int = 1) -> float:
        """
        Calculate total round-trip cost for a position trade.

        Includes:
        1. Entry fee (maker rate, limit order)
        2. Exit fee (taker rate, market order)
        3. Funding costs (proportional to holding periods)
        4. Slippage estimate

        Args:
            holding_periods_8h: Number of 8-hour funding periods held (default: 1)

        Returns:
            Total cost as percentage of notional (0.093 = 0.093%)

        Example:
            >>> fee_model = FeeModelSettings()
            >>> cost = fee_model.round_trip_cost_pct(holding_periods_8h=1)
            >>> # With defaults: (0.02 + 0.05) * 0.9 + 0.01 + 0.02 = 0.093%
        """
        # Apply BNB discount if enabled
        discount_multiplier = (100.0 - self.bnb_discount_pct) / 100.0 if self.bnb_discount_enabled else 1.0

        # Entry fee (maker, limit order)
        entry_fee = self.maker_fee_pct * discount_multiplier

        # Exit fee (taker, market order)
        exit_fee = self.taker_fee_pct * discount_multiplier

        # Funding cost (proportional to holding periods)
        funding_cost = self.funding_rate_pct * holding_periods_8h

        # Total cost
        total_cost = entry_fee + exit_fee + funding_cost + self.slippage_pct

        return total_cost
