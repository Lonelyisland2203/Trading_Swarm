"""
Pydantic models for the execution layer.

Defines data structures for:
- Order results and status tracking
- Position management
- Trade decision outputs from accept_signal()
- Daily statistics for circuit breakers
- Signal inputs from the generation pipeline
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, computed_field


class OrderResult(BaseModel):
    """Result of placing an order on the exchange."""

    order_id: str = Field(
        description="Unique order identifier from exchange"
    )
    symbol: str = Field(
        description="Trading pair symbol (e.g., 'BTCUSDT')"
    )
    side: Literal["buy", "sell"] = Field(
        description="Order side: 'buy' or 'sell'"
    )
    order_type: Literal["limit", "market"] = Field(
        description="Order type: 'limit' or 'market'"
    )
    amount: float = Field(
        ge=0,
        description="Order amount/quantity"
    )
    price: Optional[float] = Field(
        default=None,
        description="Order price (None for market orders)"
    )
    status: str = Field(
        description="Order status (e.g., 'open', 'closed', 'canceled')"
    )
    filled: float = Field(
        ge=0,
        description="Amount filled"
    )
    remaining: float = Field(
        ge=0,
        description="Amount remaining to fill"
    )
    cost: float = Field(
        ge=0,
        description="Total cost of filled amount"
    )
    fee: float = Field(
        ge=0,
        description="Trading fee paid"
    )
    timestamp: datetime = Field(
        description="Order timestamp"
    )


class OrderStatus(BaseModel):
    """Current status of an order."""

    order_id: str = Field(
        description="Unique order identifier"
    )
    symbol: str = Field(
        description="Trading pair symbol"
    )
    status: str = Field(
        description="Current order status"
    )
    filled: float = Field(
        ge=0,
        description="Amount filled"
    )
    remaining: float = Field(
        ge=0,
        description="Amount remaining to fill"
    )
    average_price: Optional[float] = Field(
        default=None,
        description="Average price of filled amount"
    )

    @computed_field
    @property
    def is_complete(self) -> bool:
        """
        Check if order is complete.

        Returns:
            True if order status is in terminal state (closed, canceled, expired, filled)
        """
        return self.status in ["closed", "canceled", "expired", "filled"]


class Position(BaseModel):
    """Open position information."""

    symbol: str = Field(
        description="Trading pair symbol"
    )
    side: Literal["long", "short"] = Field(
        description="Position side: 'long' or 'short'"
    )
    amount: float = Field(
        ge=0,
        description="Position size"
    )
    entry_price: float = Field(
        gt=0,
        description="Entry price (must be positive)"
    )
    mark_price: float = Field(
        gt=0,
        description="Current mark price (must be positive)"
    )
    unrealized_pnl: float = Field(
        description="Unrealized profit/loss"
    )
    leverage: int = Field(
        ge=1,
        le=125,
        description="Position leverage (1-125 for Binance)"
    )
    liquidation_price: Optional[float] = Field(
        default=None,
        description="Liquidation price if applicable"
    )


class TradeDecision(BaseModel):
    """Decision returned by accept_signal() to execute or reject a signal."""

    execute: bool = Field(
        description="Whether to execute the trade"
    )
    reason: str = Field(
        description="Explanation for the decision (accept or reject reason)"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Trading pair symbol"
    )
    side: Optional[Literal["buy", "sell"]] = Field(
        default=None,
        description="Trade side: 'buy' or 'sell' (None if rejected)"
    )
    amount: Optional[float] = Field(
        default=None,
        ge=0,
        description="Amount to trade"
    )
    price: Optional[float] = Field(
        default=None,
        description="Entry price (None for market orders)"
    )
    order_type: Optional[Literal["limit", "market"]] = Field(
        default=None,
        description="Order type: 'limit' or 'market'"
    )
    stop_loss_price: Optional[float] = Field(
        default=None,
        description="Stop loss price"
    )
    take_profit_price: Optional[float] = Field(
        default=None,
        description="Take profit price"
    )


class DailyStats(BaseModel):
    """Daily trading statistics for circuit breaker tracking."""

    date: str = Field(
        description="Trading date in YYYY-MM-DD format"
    )
    trade_count: int = Field(
        default=0,
        ge=0,
        description="Number of trades executed today"
    )
    realized_pnl: float = Field(
        default=0.0,
        description="Realized profit/loss for the day"
    )
    starting_balance: float = Field(
        ge=0,
        description="Starting account balance at market open"
    )
    last_order_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last order executed"
    )

    @computed_field
    @property
    def daily_loss_pct(self) -> float:
        """
        Calculate daily loss as percentage of starting balance.

        Returns:
            Daily loss percentage (0.0 if no loss, e.g., 2.5 for 2.5% loss)
            Returns 0.0 if starting_balance <= 0
            Returns 0.0 if realized_pnl >= 0 (only count losses)
        """
        if self.starting_balance <= 0:
            return 0.0
        if self.realized_pnl >= 0:
            return 0.0
        return abs(self.realized_pnl) / self.starting_balance * 100


class SignalInput(BaseModel):
    """Input signal from the signal generation pipeline."""

    symbol: str = Field(
        description="Trading pair symbol (e.g., 'BTCUSDT')"
    )
    direction: Literal["long", "short"] = Field(
        description="Trade direction: 'long' or 'short'"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Signal confidence as decimal [0.0, 1.0]"
    )
    expected_return_pct: float = Field(
        description="Expected return as percentage (can be negative)"
    )
    stop_loss_pct: float = Field(
        gt=0,
        description="Stop loss distance as percentage (must be positive)"
    )
    take_profit_pct: Optional[float] = Field(
        default=None,
        description="Take profit target as percentage"
    )
    timeframe: str = Field(
        default="1h",
        description="Signal timeframe (default: '1h')"
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Suggested entry price"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional signal metadata"
    )

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate direction is 'long' or 'short'."""
        if v not in ["long", "short"]:
            raise ValueError("direction must be 'long' or 'short'")
        return v
