"""
Position sizing calculation with fee awareness and risk management.

Calculates trade size based on:
- Risk tolerance (percentage of balance to risk per trade)
- Stop loss distance (tighter stops allow larger positions)
- Trading fees (reduce position size to maintain risk target)
- Position cap (hard limit as percentage of balance)

The position sizing ensures that the actual risk (including fees) matches the
desired risk percentage, while respecting maximum position size constraints.
"""

from dataclasses import dataclass

from config.fee_model import FeeModelSettings


@dataclass
class PositionSizeResult:
    """
    Result of position size calculation.

    Attributes:
        amount: Position size in base currency (e.g., BTC amount)
        notional: Position value in quote currency (e.g., USDT value)
        risk_amount: Actual risk in quote currency after fees
        stop_distance_pct: Distance to stop loss as percentage
        fees_included: Total fees deducted from risk budget
        capped_by_max: Whether position was capped by max_position_pct limit
        reason: Human-readable explanation of calculation
    """

    amount: float
    notional: float
    risk_amount: float
    stop_distance_pct: float
    fees_included: float
    capped_by_max: bool
    reason: str


def calculate_position_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    fee_model: FeeModelSettings,
    max_position_pct: float = 0.02,
) -> PositionSizeResult:
    """
    Calculate position size based on risk tolerance and stop distance.

    This function determines how much to trade based on the desired risk per trade,
    accounting for both stop loss distance and trading fees. The calculation ensures
    that the actual risk (including fees) matches the risk target, while respecting
    maximum position size constraints.

    Args:
        balance: Account balance in quote currency
        risk_pct: Risk per trade as decimal (0.01 = 1%)
        entry_price: Entry price for the position
        stop_price: Stop loss price
        fee_model: FeeModelSettings for cost calculation
        max_position_pct: Maximum position as fraction of balance (default: 0.02 = 2%)

    Returns:
        PositionSizeResult with calculated position size and metadata

    Raises:
        ValueError: If risk_pct is negative

    Examples:
        >>> fee_model = FeeModelSettings()
        >>> result = calculate_position_size(
        ...     balance=10000.0,
        ...     risk_pct=0.01,  # 1% risk
        ...     entry_price=50000.0,
        ...     stop_price=49000.0,  # 2% stop
        ...     fee_model=fee_model,
        ... )
        >>> result.amount > 0
        True
        >>> result.risk_amount <= 100.0  # ~1% of 10k
        True
    """
    # Validate risk_pct
    if risk_pct < 0:
        raise ValueError(f"risk_pct must be non-negative, got {risk_pct}")

    # Handle edge cases: zero or negative balance
    if balance <= 0:
        return PositionSizeResult(
            amount=0.0,
            notional=0.0,
            risk_amount=0.0,
            stop_distance_pct=0.0,
            fees_included=0.0,
            capped_by_max=False,
            reason="Zero or negative balance",
        )

    # Handle edge case: zero or negative entry price
    if entry_price <= 0:
        return PositionSizeResult(
            amount=0.0,
            notional=0.0,
            risk_amount=0.0,
            stop_distance_pct=0.0,
            fees_included=0.0,
            capped_by_max=False,
            reason="Invalid entry price (must be positive)",
        )

    # Calculate stop distance as percentage
    stop_distance_pct = abs(entry_price - stop_price) / entry_price

    # Handle edge case: entry equals stop (zero stop distance)
    if stop_distance_pct == 0:
        return PositionSizeResult(
            amount=0.0,
            notional=0.0,
            risk_amount=0.0,
            stop_distance_pct=0.0,
            fees_included=0.0,
            capped_by_max=False,
            reason="Invalid stop distance (entry price equals stop price)",
        )

    # Calculate fees as percentage (using 1 period as default for position sizing)
    fees_pct = fee_model.round_trip_cost_pct(holding_periods_8h=1.0) / 100.0

    # Total loss percentage includes both stop distance and fees
    total_loss_pct = stop_distance_pct + fees_pct

    # Calculate risk amount in quote currency
    risk_amount = balance * risk_pct

    # Calculate maximum notional from risk budget
    # risk_amount = notional * total_loss_pct
    # Therefore: notional = risk_amount / total_loss_pct
    max_notional_from_risk = risk_amount / total_loss_pct

    # Calculate maximum notional from position cap
    max_notional_from_cap = balance * max_position_pct

    # Use the smaller of the two limits
    notional = min(max_notional_from_risk, max_notional_from_cap)

    # Calculate amount in base currency
    amount = notional / entry_price

    # Calculate actual risk taken (after potential capping)
    actual_risk = notional * total_loss_pct

    # Check if position was capped
    capped_by_max = max_notional_from_risk > max_notional_from_cap

    # Calculate fees in absolute terms
    fees_included = notional * fees_pct

    # Build reason string
    if capped_by_max:
        reason = (
            f"Position capped at {max_position_pct*100:.1f}% of balance "
            f"(would have been {max_notional_from_risk:.2f} based on risk)"
        )
    else:
        reason = (
            f"Position sized for {risk_pct*100:.1f}% risk with "
            f"{stop_distance_pct*100:.2f}% stop distance and "
            f"{fees_pct*100:.3f}% fees"
        )

    return PositionSizeResult(
        amount=amount,
        notional=notional,
        risk_amount=actual_risk,
        stop_distance_pct=stop_distance_pct,
        fees_included=fees_included,
        capped_by_max=capped_by_max,
        reason=reason,
    )
