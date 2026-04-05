"""
Fee model configuration for Binance Futures USDT-M.

Models trading costs including:
- Maker/taker fees with optional BNB discount
- Funding costs for holding periods
- Slippage estimates
"""

from pydantic import BaseModel, Field, model_validator


class FeeModelSettings(BaseModel):
    """
    Fee structure for Binance Futures USDT-M perpetual contracts.

    Includes entry/exit fees, funding costs, and slippage estimates.
    All fees expressed as percentages of position size.

    Attributes:
        maker_fee_pct: Maker fee as % of notional (default: 0.02% for limit orders)
        taker_fee_pct: Taker fee as % of notional (default: 0.05% for market orders)
        entry_order_type: Entry order type ("maker" for limit, "taker" for market, default: "maker")
        exit_order_type: Exit order type ("maker" for limit, "taker" for market, default: "taker")
        bnb_discount_enabled: Apply 10% BNB discount to trading fees (Binance Futures discount)
        bnb_discount_pct: BNB discount as % (default: 10% = 0.1, NOT 25% which is Spot-only)
        funding_rate_pct: Hourly funding rate as % (default: 0.01% per hour)
        funding_interval_hours: Funding payment interval (default: 8 hours per period)
        slippage_pct: Market impact/slippage estimate as % (default: 0.02%)
        include_funding: Include funding costs in round_trip calculation (default: True)
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
    entry_order_type: str = Field(
        default="maker",
        pattern="^(maker|taker)$",
        description="Entry order type: 'maker' (limit) or 'taker' (market)"
    )
    exit_order_type: str = Field(
        default="taker",
        pattern="^(maker|taker)$",
        description="Exit order type: 'maker' (limit) or 'taker' (market)"
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
    include_funding: bool = Field(
        default=True,
        description="Include funding costs in round_trip calculation"
    )

    @model_validator(mode="before")
    @classmethod
    def handle_bnb_discount_alias(cls, data):
        """Handle bnb_discount boolean parameter as alias for bnb_discount_enabled."""
        if isinstance(data, dict) and "bnb_discount" in data:
            # Convert bnb_discount boolean to bnb_discount_enabled
            data["bnb_discount_enabled"] = data.pop("bnb_discount")
        return data

    def round_trip_cost_pct(self, holding_periods_8h: float = 1.0) -> float:
        """
        Calculate total round-trip cost for a position trade.

        Includes:
        1. Entry fee (based on entry_order_type)
        2. Exit fee (based on exit_order_type)
        3. Funding costs (proportional to holding periods, if include_funding=True)
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
        if holding_periods_8h < 0:
            raise ValueError("holding_periods_8h must be non-negative")

        # Apply BNB discount if enabled
        discount_multiplier = (100.0 - self.bnb_discount_pct) / 100.0 if self.bnb_discount_enabled else 1.0

        # Entry fee based on order type
        entry_fee_rate = self.maker_fee_pct if self.entry_order_type == "maker" else self.taker_fee_pct
        entry_fee = entry_fee_rate * discount_multiplier

        # Exit fee based on order type
        exit_fee_rate = self.maker_fee_pct if self.exit_order_type == "maker" else self.taker_fee_pct
        exit_fee = exit_fee_rate * discount_multiplier

        # Funding cost (proportional to holding periods)
        funding_cost = (self.funding_rate_pct * holding_periods_8h) if self.include_funding else 0.0

        # Total cost
        total_cost = entry_fee + exit_fee + funding_cost + self.slippage_pct

        return total_cost

    def net_return(
        self,
        gross_return_pct: float,
        holding_periods_8h: float,
    ) -> float:
        """
        Compute net return after subtracting all fees.

        Args:
            gross_return_pct: Gross return as percentage (e.g., 0.15 for +0.15%)
            holding_periods_8h: Holding period in 8-hour funding periods

        Returns:
            Net return as percentage after all fees

        Examples:
            >>> fee_model = FeeModelSettings()
            >>> fee_model.net_return(0.15, 0)  # +0.15% gross, no funding
            0.067  # 0.15 - 0.083 = 0.067%
            >>> fee_model.net_return(0.08, 0)  # +0.08% gross
            -0.003  # Below fee hurdle - net loss
        """
        total_cost = self.round_trip_cost_pct(holding_periods_8h)
        return gross_return_pct - total_cost

    def minimum_profitable_return_pct(self, holding_periods_8h: float) -> float:
        """
        Compute minimum gross return needed to break even after fees.

        This is the break-even threshold - any signal with gross return below
        this value will result in a net loss after fees.

        Args:
            holding_periods_8h: Holding period in 8-hour funding periods

        Returns:
            Minimum profitable return as percentage

        Examples:
            >>> fee_model = FeeModelSettings()
            >>> fee_model.minimum_profitable_return_pct(0)  # No funding
            0.083  # Must exceed 0.083% to profit
            >>> fee_model.minimum_profitable_return_pct(3)  # 3 periods
            0.113  # 0.083% + 0.03% funding = 0.113%
        """
        return self.round_trip_cost_pct(holding_periods_8h)
