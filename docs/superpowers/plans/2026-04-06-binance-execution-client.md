# Binance Execution Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-safe execution layer for deploying trading signals to Binance, with comprehensive safety controls including kill switches, daily loss circuit breakers, position limits, and fee-aware sizing.

**Architecture:** The execution client wraps CCXT's binance async support, defaulting to testnet mode for safety. All order decisions flow through a centralized `accept_signal()` method that enforces multiple safety checks before returning a decision. State tracking (daily stats, order logs) persists to JSON files, enabling inspection and debugging. A physical STOP file acts as an instant kill switch.

**Tech Stack:** Python 3.13, CCXT (async), Pydantic settings, pytest with mocked CCXT

---

## File Structure

```
execution/
  __init__.py          - Package exports
  exceptions.py        - ExecutionError hierarchy
  models.py            - Pydantic models (OrderResult, Position, TradeDecision, etc.)
  binance_client.py    - BinanceExecutionClient class
  state/               - Runtime state directory (gitignored)
    daily_stats.json   - Daily trade counts, P&L tracking
    order_log.jsonl    - Append-only order history
    STOP               - Kill switch file (presence stops all trading)

config/
  settings.py          - Add ExecutionSettings nested model

tests/
  test_execution/
    __init__.py
    test_exceptions.py
    test_models.py
    test_binance_client.py
    test_safety_controls.py
    test_position_sizing.py
```

---

## Task 1: Create Execution Exception Hierarchy

**Files:**
- Create: `/Users/javierlee/Trading Swarm/execution/__init__.py`
- Create: `/Users/javierlee/Trading Swarm/execution/exceptions.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/__init__.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_exceptions.py`

- [ ] **Step 1: Create execution package init**

```python
# execution/__init__.py
"""
Execution layer for deploying trading signals to Binance.

This module provides a production-safe execution client with:
- Testnet-first design (live trading requires explicit opt-in)
- Kill switch support (STOP file halts all trading)
- Daily loss circuit breaker
- Position size limits
- Order cooldowns
- Comprehensive logging

Usage:
    from execution import BinanceExecutionClient, ExecutionSettings

Warning:
    Live trading requires ALLOW_LIVE_TRADING=true environment variable.
    Without this, testnet=False will raise an error.
"""

from execution.exceptions import (
    ExecutionError,
    KillSwitchActiveError,
    DailyLossLimitError,
    PositionLimitError,
    CooldownActiveError,
    InsufficientBalanceError,
    LiveTradingNotAllowedError,
)
from execution.models import (
    OrderResult,
    OrderStatus,
    Position,
    TradeDecision,
    DailyStats,
)
from execution.binance_client import BinanceExecutionClient

__all__ = [
    # Exceptions
    "ExecutionError",
    "KillSwitchActiveError",
    "DailyLossLimitError",
    "PositionLimitError",
    "CooldownActiveError",
    "InsufficientBalanceError",
    "LiveTradingNotAllowedError",
    # Models
    "OrderResult",
    "OrderStatus",
    "Position",
    "TradeDecision",
    "DailyStats",
    # Client
    "BinanceExecutionClient",
]
```

- [ ] **Step 2: Create exception hierarchy**

```python
# execution/exceptions.py
"""
Custom exception hierarchy for execution layer.

Distinguishes between:
- Safety-related blocks (kill switch, limits)
- Transient errors (network, rate limits)
- Configuration errors (live trading not allowed)
"""


class ExecutionError(Exception):
    """Base class for execution-related errors."""
    pass


class KillSwitchActiveError(ExecutionError):
    """
    STOP file detected - all trading halted.

    This is a non-retryable, non-negotiable block.
    Remove the STOP file manually to resume trading.
    """
    pass


class DailyLossLimitError(ExecutionError):
    """
    Daily loss limit exceeded - trading halted until next day.

    This is a circuit breaker to prevent catastrophic losses.
    """
    def __init__(self, current_loss_pct: float, limit_pct: float):
        self.current_loss_pct = current_loss_pct
        self.limit_pct = limit_pct
        super().__init__(
            f"Daily loss limit breached: {current_loss_pct:.2f}% exceeds {limit_pct:.2f}% limit"
        )


class PositionLimitError(ExecutionError):
    """
    Maximum open positions reached - cannot open new positions.

    Close existing positions before opening new ones.
    """
    def __init__(self, current_positions: int, max_positions: int):
        self.current_positions = current_positions
        self.max_positions = max_positions
        super().__init__(
            f"Position limit reached: {current_positions}/{max_positions} positions open"
        )


class CooldownActiveError(ExecutionError):
    """
    Order cooldown period active - must wait before placing another order.

    This prevents rapid-fire trading that could exceed rate limits
    or indicate a malfunctioning strategy.
    """
    def __init__(self, seconds_remaining: float):
        self.seconds_remaining = seconds_remaining
        super().__init__(
            f"Order cooldown active: {seconds_remaining:.1f}s remaining"
        )


class DailyTradeCountError(ExecutionError):
    """
    Maximum daily trade count reached.

    This prevents overtrading and protects against runaway loops.
    """
    def __init__(self, current_count: int, max_count: int):
        self.current_count = current_count
        self.max_count = max_count
        super().__init__(
            f"Daily trade limit reached: {current_count}/{max_count} trades today"
        )


class InsufficientBalanceError(ExecutionError):
    """
    Insufficient balance to execute the requested order.
    """
    def __init__(self, required: float, available: float, asset: str):
        self.required = required
        self.available = available
        self.asset = asset
        super().__init__(
            f"Insufficient {asset}: required {required:.4f}, available {available:.4f}"
        )


class LiveTradingNotAllowedError(ExecutionError):
    """
    Attempted to initialize live trading without explicit permission.

    Live trading requires ALLOW_LIVE_TRADING=true environment variable.
    This is a safety measure to prevent accidental live trading.
    """
    pass


class OrderRejectedError(ExecutionError):
    """
    Order was rejected by the exchange.
    """
    def __init__(self, reason: str, exchange_error: str | None = None):
        self.reason = reason
        self.exchange_error = exchange_error
        super().__init__(f"Order rejected: {reason}")


class SignalRejectedError(ExecutionError):
    """
    Signal did not pass acceptance criteria.
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Signal rejected: {reason}")
```

- [ ] **Step 3: Create test directory and init**

```python
# tests/test_execution/__init__.py
"""Tests for execution layer."""
```

- [ ] **Step 4: Write exception tests**

```python
# tests/test_execution/test_exceptions.py
"""Tests for execution exception hierarchy."""

import pytest
from execution.exceptions import (
    ExecutionError,
    KillSwitchActiveError,
    DailyLossLimitError,
    PositionLimitError,
    CooldownActiveError,
    DailyTradeCountError,
    InsufficientBalanceError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
    SignalRejectedError,
)


class TestExceptionHierarchy:
    """Test exception inheritance."""

    def test_all_exceptions_inherit_from_execution_error(self):
        """All custom exceptions should inherit from ExecutionError."""
        exceptions = [
            KillSwitchActiveError(),
            DailyLossLimitError(2.5, 2.0),
            PositionLimitError(3, 3),
            CooldownActiveError(30.0),
            DailyTradeCountError(10, 10),
            InsufficientBalanceError(100.0, 50.0, "USDT"),
            LiveTradingNotAllowedError(),
            OrderRejectedError("Invalid quantity"),
            SignalRejectedError("Below confidence threshold"),
        ]

        for exc in exceptions:
            assert isinstance(exc, ExecutionError)
            assert isinstance(exc, Exception)


class TestDailyLossLimitError:
    """Test DailyLossLimitError formatting."""

    def test_stores_values(self):
        """Error should store current and limit values."""
        exc = DailyLossLimitError(2.5, 2.0)
        assert exc.current_loss_pct == 2.5
        assert exc.limit_pct == 2.0

    def test_message_format(self):
        """Error message should include both values."""
        exc = DailyLossLimitError(2.5, 2.0)
        assert "2.50%" in str(exc)
        assert "2.00%" in str(exc)


class TestPositionLimitError:
    """Test PositionLimitError formatting."""

    def test_stores_values(self):
        """Error should store position counts."""
        exc = PositionLimitError(3, 3)
        assert exc.current_positions == 3
        assert exc.max_positions == 3

    def test_message_format(self):
        """Error message should include position counts."""
        exc = PositionLimitError(3, 5)
        assert "3/5" in str(exc)


class TestCooldownActiveError:
    """Test CooldownActiveError formatting."""

    def test_stores_remaining_time(self):
        """Error should store remaining seconds."""
        exc = CooldownActiveError(45.5)
        assert exc.seconds_remaining == 45.5

    def test_message_format(self):
        """Error message should include remaining time."""
        exc = CooldownActiveError(45.5)
        assert "45.5s" in str(exc)


class TestInsufficientBalanceError:
    """Test InsufficientBalanceError formatting."""

    def test_stores_values(self):
        """Error should store balance details."""
        exc = InsufficientBalanceError(100.0, 50.0, "USDT")
        assert exc.required == 100.0
        assert exc.available == 50.0
        assert exc.asset == "USDT"

    def test_message_format(self):
        """Error message should include all values."""
        exc = InsufficientBalanceError(100.0, 50.0, "USDT")
        assert "USDT" in str(exc)
        assert "100.0" in str(exc)
        assert "50.0" in str(exc)


class TestOrderRejectedError:
    """Test OrderRejectedError formatting."""

    def test_stores_reason(self):
        """Error should store rejection reason."""
        exc = OrderRejectedError("Invalid quantity", "MIN_NOTIONAL")
        assert exc.reason == "Invalid quantity"
        assert exc.exchange_error == "MIN_NOTIONAL"

    def test_message_without_exchange_error(self):
        """Error should work without exchange error."""
        exc = OrderRejectedError("Invalid quantity")
        assert "Invalid quantity" in str(exc)
        assert exc.exchange_error is None
```

- [ ] **Step 5: Run tests to verify exceptions**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_exceptions.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add execution/__init__.py execution/exceptions.py tests/test_execution/__init__.py tests/test_execution/test_exceptions.py
git commit -m "feat(execution): add exception hierarchy for execution layer"
```

---

## Task 2: Create Pydantic Models

**Files:**
- Create: `/Users/javierlee/Trading Swarm/execution/models.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_models.py`

- [ ] **Step 1: Write model tests**

```python
# tests/test_execution/test_models.py
"""Tests for execution data models."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from execution.models import (
    OrderResult,
    OrderStatus,
    Position,
    TradeDecision,
    DailyStats,
    SignalInput,
)


class TestOrderResult:
    """Test OrderResult model."""

    def test_create_order_result(self):
        """Test creating a valid order result."""
        result = OrderResult(
            order_id="12345",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            amount=0.001,
            price=50000.0,
            status="open",
            filled=0.0,
            remaining=0.001,
            cost=0.0,
            fee=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.order_id == "12345"
        assert result.symbol == "BTC/USDT"
        assert result.side == "buy"

    def test_order_result_requires_valid_side(self):
        """Side must be 'buy' or 'sell'."""
        with pytest.raises(ValueError):
            OrderResult(
                order_id="12345",
                symbol="BTC/USDT",
                side="invalid",
                order_type="limit",
                amount=0.001,
                price=50000.0,
                status="open",
                filled=0.0,
                remaining=0.001,
                cost=0.0,
                fee=0.0,
                timestamp=datetime.now(timezone.utc),
            )


class TestOrderStatus:
    """Test OrderStatus model."""

    def test_create_order_status(self):
        """Test creating a valid order status."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTC/USDT",
            status="filled",
            filled=0.001,
            remaining=0.0,
            average_price=50000.0,
        )
        assert status.order_id == "12345"
        assert status.status == "filled"
        assert status.is_complete is True

    def test_is_complete_for_open_order(self):
        """Open orders should not be complete."""
        status = OrderStatus(
            order_id="12345",
            symbol="BTC/USDT",
            status="open",
            filled=0.0005,
            remaining=0.0005,
            average_price=50000.0,
        )
        assert status.is_complete is False


class TestPosition:
    """Test Position model."""

    def test_create_long_position(self):
        """Test creating a long position."""
        pos = Position(
            symbol="BTC/USDT",
            side="long",
            amount=0.001,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=1.0,
            leverage=1,
        )
        assert pos.symbol == "BTC/USDT"
        assert pos.side == "long"
        assert pos.unrealized_pnl == 1.0

    def test_create_short_position(self):
        """Test creating a short position."""
        pos = Position(
            symbol="BTC/USDT",
            side="short",
            amount=0.001,
            entry_price=50000.0,
            mark_price=49000.0,
            unrealized_pnl=1.0,
            leverage=1,
        )
        assert pos.side == "short"


class TestTradeDecision:
    """Test TradeDecision model."""

    def test_create_execute_decision(self):
        """Test creating an execute decision."""
        decision = TradeDecision(
            execute=True,
            reason="All safety checks passed",
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            price=50000.0,
            order_type="limit",
        )
        assert decision.execute is True
        assert decision.symbol == "BTC/USDT"

    def test_create_reject_decision(self):
        """Test creating a rejection decision."""
        decision = TradeDecision(
            execute=False,
            reason="Daily loss limit exceeded",
            symbol=None,
            side=None,
            amount=None,
            price=None,
            order_type=None,
        )
        assert decision.execute is False
        assert "loss limit" in decision.reason


class TestDailyStats:
    """Test DailyStats model."""

    def test_create_daily_stats(self):
        """Test creating daily stats."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=5,
            realized_pnl=-1.5,
            starting_balance=1000.0,
        )
        assert stats.trade_count == 5
        assert stats.realized_pnl == -1.5

    def test_daily_loss_pct_calculation(self):
        """Test daily loss percentage calculation."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=5,
            realized_pnl=-20.0,
            starting_balance=1000.0,
        )
        assert stats.daily_loss_pct == 2.0  # 20/1000 = 2%

    def test_daily_loss_pct_with_zero_balance(self):
        """Test handling of zero starting balance."""
        stats = DailyStats(
            date="2026-04-06",
            trade_count=0,
            realized_pnl=0.0,
            starting_balance=0.0,
        )
        assert stats.daily_loss_pct == 0.0


class TestSignalInput:
    """Test SignalInput model."""

    def test_create_signal_input(self):
        """Test creating a signal input."""
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.75,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            take_profit_pct=1.5,
            timeframe="1h",
        )
        assert signal.symbol == "BTC/USDT"
        assert signal.direction == "long"
        assert signal.confidence == 0.75

    def test_confidence_must_be_between_0_and_1(self):
        """Confidence must be in [0, 1] range."""
        with pytest.raises(ValueError):
            SignalInput(
                symbol="BTC/USDT",
                direction="long",
                confidence=1.5,
                expected_return_pct=0.5,
                stop_loss_pct=1.0,
            )

    def test_direction_must_be_valid(self):
        """Direction must be 'long' or 'short'."""
        with pytest.raises(ValueError):
            SignalInput(
                symbol="BTC/USDT",
                direction="invalid",
                confidence=0.75,
                expected_return_pct=0.5,
                stop_loss_pct=1.0,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_models.py -v`
Expected: FAIL with "No module named 'execution.models'"

- [ ] **Step 3: Implement models**

```python
# execution/models.py
"""
Pydantic models for execution layer.

Defines data structures for:
- Order results and status
- Position tracking
- Trade decisions
- Daily statistics
- Signal inputs
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class OrderResult(BaseModel):
    """
    Result of placing an order.

    Contains all information returned by the exchange after order submission.
    """
    order_id: str = Field(description="Exchange-assigned order ID")
    symbol: str = Field(description="Trading pair (e.g., 'BTC/USDT')")
    side: Literal["buy", "sell"] = Field(description="Order side")
    order_type: Literal["limit", "market"] = Field(description="Order type")
    amount: float = Field(ge=0, description="Order quantity in base currency")
    price: float | None = Field(description="Limit price (None for market orders)")
    status: str = Field(description="Order status from exchange")
    filled: float = Field(ge=0, description="Amount filled")
    remaining: float = Field(ge=0, description="Amount remaining")
    cost: float = Field(ge=0, description="Total cost in quote currency")
    fee: float = Field(ge=0, description="Fee paid")
    timestamp: datetime = Field(description="Order creation timestamp")


class OrderStatus(BaseModel):
    """
    Current status of an order.

    Used for polling order state after submission.
    """
    order_id: str = Field(description="Exchange-assigned order ID")
    symbol: str = Field(description="Trading pair")
    status: str = Field(description="Order status (open, closed, canceled, expired)")
    filled: float = Field(ge=0, description="Amount filled")
    remaining: float = Field(ge=0, description="Amount remaining")
    average_price: float | None = Field(description="Average fill price")

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or canceled)."""
        return self.status in ("closed", "canceled", "expired", "filled")


class Position(BaseModel):
    """
    Open position information.

    Represents a futures or margin position with P&L tracking.
    """
    symbol: str = Field(description="Trading pair")
    side: Literal["long", "short"] = Field(description="Position direction")
    amount: float = Field(ge=0, description="Position size in base currency")
    entry_price: float = Field(gt=0, description="Average entry price")
    mark_price: float = Field(gt=0, description="Current mark price")
    unrealized_pnl: float = Field(description="Unrealized profit/loss in quote currency")
    leverage: int = Field(ge=1, le=125, description="Leverage multiplier")
    liquidation_price: float | None = Field(default=None, description="Liquidation price")


class TradeDecision(BaseModel):
    """
    Decision returned by accept_signal().

    Contains whether to execute and all parameters needed for order placement.
    If execute=False, only reason is populated.
    """
    execute: bool = Field(description="Whether to execute the trade")
    reason: str = Field(description="Reason for decision (pass or rejection reason)")
    symbol: str | None = Field(default=None, description="Trading pair")
    side: Literal["buy", "sell"] | None = Field(default=None, description="Order side")
    amount: float | None = Field(default=None, ge=0, description="Order quantity")
    price: float | None = Field(default=None, description="Limit price (None for market)")
    order_type: Literal["limit", "market"] | None = Field(default=None, description="Order type")
    stop_loss_price: float | None = Field(default=None, description="Stop loss price")
    take_profit_price: float | None = Field(default=None, description="Take profit price")


class DailyStats(BaseModel):
    """
    Daily trading statistics.

    Persisted to JSON for tracking limits and circuit breakers.
    """
    date: str = Field(description="Date in YYYY-MM-DD format")
    trade_count: int = Field(ge=0, default=0, description="Number of trades today")
    realized_pnl: float = Field(default=0.0, description="Realized P&L in quote currency")
    starting_balance: float = Field(ge=0, description="Balance at start of day")
    last_order_timestamp: datetime | None = Field(default=None, description="Last order time for cooldown")

    @computed_field
    @property
    def daily_loss_pct(self) -> float:
        """Calculate daily loss as percentage of starting balance."""
        if self.starting_balance <= 0:
            return 0.0
        # Only count losses (negative P&L)
        if self.realized_pnl >= 0:
            return 0.0
        return abs(self.realized_pnl) / self.starting_balance * 100


class SignalInput(BaseModel):
    """
    Input signal from the signal generation pipeline.

    Contains all information needed to evaluate and potentially execute a trade.
    """
    symbol: str = Field(description="Trading pair (e.g., 'BTC/USDT')")
    direction: Literal["long", "short"] = Field(description="Signal direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence [0, 1]")
    expected_return_pct: float = Field(description="Expected return as percentage")
    stop_loss_pct: float = Field(gt=0, description="Stop loss distance as percentage")
    take_profit_pct: float | None = Field(default=None, description="Take profit distance as percentage")
    timeframe: str = Field(default="1h", description="Signal timeframe")
    entry_price: float | None = Field(default=None, description="Suggested entry price")
    metadata: dict | None = Field(default=None, description="Additional signal metadata")

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Ensure direction is valid."""
        if v not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")
        return v
```

- [ ] **Step 4: Run tests to verify models**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_models.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add execution/models.py tests/test_execution/test_models.py
git commit -m "feat(execution): add Pydantic models for orders, positions, and decisions"
```

---

## Task 3: Add ExecutionSettings to Config

**Files:**
- Modify: `/Users/javierlee/Trading Swarm/config/settings.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_settings.py`

- [ ] **Step 1: Write settings tests**

```python
# tests/test_execution/test_settings.py
"""Tests for ExecutionSettings configuration."""

import os
import pytest
from pathlib import Path


class TestExecutionSettings:
    """Test ExecutionSettings model."""

    def test_default_values(self):
        """Test default safety limits."""
        from config.settings import ExecutionSettings

        settings = ExecutionSettings()

        assert settings.testnet is True
        assert settings.max_daily_trades == 10
        assert settings.max_daily_loss_pct == 2.0
        assert settings.max_open_positions == 3
        assert settings.max_position_pct == 0.02
        assert settings.order_cooldown_seconds == 60
        assert settings.min_confidence == 0.6

    def test_testnet_default_is_true(self):
        """Testnet must default to True for safety."""
        from config.settings import ExecutionSettings

        settings = ExecutionSettings()
        assert settings.testnet is True

    def test_max_position_pct_capped(self):
        """Max position percentage cannot exceed 10%."""
        from config.settings import ExecutionSettings

        with pytest.raises(ValueError):
            ExecutionSettings(max_position_pct=0.15)

    def test_max_daily_loss_pct_minimum(self):
        """Max daily loss must be at least 0.1%."""
        from config.settings import ExecutionSettings

        with pytest.raises(ValueError):
            ExecutionSettings(max_daily_loss_pct=0.05)

    def test_state_dir_default(self):
        """State directory should default to execution/state."""
        from config.settings import ExecutionSettings

        settings = ExecutionSettings()
        assert settings.state_dir == Path("execution/state")


class TestExecutionSettingsInAppSettings:
    """Test ExecutionSettings integration with AppSettings."""

    def test_execution_in_app_settings(self):
        """ExecutionSettings should be accessible from AppSettings."""
        from config.settings import AppSettings

        app_settings = AppSettings()
        assert hasattr(app_settings, 'execution')
        assert app_settings.execution.testnet is True

    def test_env_var_override(self):
        """Environment variables should override defaults."""
        os.environ["EXECUTION_MAX_DAILY_TRADES"] = "20"
        os.environ["EXECUTION_MAX_POSITION_PCT"] = "0.05"

        from config.settings import AppSettings

        # Force reload
        app_settings = AppSettings()

        assert app_settings.execution.max_daily_trades == 20
        assert app_settings.execution.max_position_pct == 0.05

        # Cleanup
        del os.environ["EXECUTION_MAX_DAILY_TRADES"]
        del os.environ["EXECUTION_MAX_POSITION_PCT"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_settings.py -v`
Expected: FAIL with AttributeError or ImportError

- [ ] **Step 3: Add ExecutionSettings to config/settings.py**

Add after FeeModelSettings import (around line 15):

```python
# In config/settings.py, add this class after DatasetGenerationSettings (around line 370):

class ExecutionSettings(BaseModel):
    """
    Execution layer configuration with safety limits.

    All limits are designed to prevent catastrophic losses.
    Defaults are conservative - adjust based on risk tolerance.
    """

    # Connection settings
    testnet: bool = Field(
        default=True,
        description="Use testnet (REQUIRED: set to False only with ALLOW_LIVE_TRADING=true)"
    )
    api_key: str = Field(
        default="",
        description="Binance API key (loaded from environment)"
    )
    api_secret: str = Field(
        default="",
        description="Binance API secret (loaded from environment)"
    )

    # Trading mode
    mode: str = Field(
        default="futures",
        pattern="^(spot|futures)$",
        description="Trading mode: 'spot' or 'futures'"
    )

    # Safety limits
    max_daily_trades: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum trades per day"
    )
    max_daily_loss_pct: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Maximum daily loss as % of balance (circuit breaker)"
    )
    max_open_positions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum concurrent open positions"
    )
    max_position_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.10,
        description="Maximum position size as % of portfolio (2% = 0.02)"
    )
    order_cooldown_seconds: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Minimum seconds between orders"
    )

    # Signal acceptance thresholds
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum signal confidence to accept"
    )
    min_expected_return_pct: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum expected return % (before fees)"
    )

    # State persistence
    state_dir: Path = Field(
        default=Path("execution/state"),
        description="Directory for state files (daily_stats.json, order_log.jsonl, STOP)"
    )

    @field_validator("state_dir", mode="before")
    @classmethod
    def parse_state_dir(cls, v):
        """Convert string path to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v
```

- [ ] **Step 4: Add ExecutionSettings to AppSettings class**

In the AppSettings class (around line 390), add to nested settings:

```python
    # Add after fee_model in AppSettings:
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
```

- [ ] **Step 5: Add environment variable mappings**

In the `model_post_init` method, add these mappings to `env_mappings` dict (around line 490):

```python
            # Execution settings
            "EXECUTION_TESTNET": ("execution", "testnet"),
            "BINANCE_API_KEY": ("execution", "api_key"),
            "BINANCE_API_SECRET": ("execution", "api_secret"),
            "EXECUTION_MODE": ("execution", "mode"),
            "EXECUTION_MAX_DAILY_TRADES": ("execution", "max_daily_trades"),
            "EXECUTION_MAX_DAILY_LOSS_PCT": ("execution", "max_daily_loss_pct"),
            "EXECUTION_MAX_OPEN_POSITIONS": ("execution", "max_open_positions"),
            "EXECUTION_MAX_POSITION_PCT": ("execution", "max_position_pct"),
            "EXECUTION_ORDER_COOLDOWN": ("execution", "order_cooldown_seconds"),
            "EXECUTION_MIN_CONFIDENCE": ("execution", "min_confidence"),
            "EXECUTION_STATE_DIR": ("execution", "state_dir"),
```

- [ ] **Step 6: Add re-validation for execution settings**

In `model_post_init`, add after the other re-validations (around line 527):

```python
        self.execution = ExecutionSettings.model_validate(self.execution.model_dump())
```

- [ ] **Step 7: Run tests to verify settings**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_settings.py -v`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add config/settings.py tests/test_execution/test_settings.py
git commit -m "feat(config): add ExecutionSettings with safety limits"
```

---

## Task 4: Implement State Management

**Files:**
- Create: `/Users/javierlee/Trading Swarm/execution/state_manager.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_state_manager.py`

- [ ] **Step 1: Write state manager tests**

```python
# tests/test_execution/test_state_manager.py
"""Tests for execution state management."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from execution.state_manager import StateManager
from execution.models import DailyStats


class TestStateManager:
    """Test StateManager functionality."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def state_manager(self, state_dir):
        """Create StateManager with temp directory."""
        return StateManager(state_dir)

    def test_init_creates_directory(self, state_dir):
        """StateManager should create state directory if missing."""
        subdir = state_dir / "subdir"
        manager = StateManager(subdir)
        assert subdir.exists()

    def test_check_kill_switch_no_file(self, state_manager):
        """No STOP file means kill switch is inactive."""
        assert state_manager.is_kill_switch_active() is False

    def test_check_kill_switch_with_file(self, state_manager, state_dir):
        """STOP file presence activates kill switch."""
        (state_dir / "STOP").touch()
        assert state_manager.is_kill_switch_active() is True

    def test_get_daily_stats_creates_new(self, state_manager):
        """Should create new stats if none exist for today."""
        stats = state_manager.get_daily_stats(starting_balance=1000.0)

        assert stats.date == datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert stats.trade_count == 0
        assert stats.realized_pnl == 0.0
        assert stats.starting_balance == 1000.0

    def test_get_daily_stats_loads_existing(self, state_manager, state_dir):
        """Should load existing stats for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        existing = DailyStats(
            date=today,
            trade_count=5,
            realized_pnl=-10.0,
            starting_balance=1000.0,
        )

        with open(state_dir / "daily_stats.json", "w") as f:
            f.write(existing.model_dump_json())

        stats = state_manager.get_daily_stats(starting_balance=1000.0)
        assert stats.trade_count == 5
        assert stats.realized_pnl == -10.0

    def test_get_daily_stats_resets_for_new_day(self, state_manager, state_dir):
        """Should reset stats when date changes."""
        old_date = "2026-04-05"
        existing = DailyStats(
            date=old_date,
            trade_count=5,
            realized_pnl=-10.0,
            starting_balance=1000.0,
        )

        with open(state_dir / "daily_stats.json", "w") as f:
            f.write(existing.model_dump_json())

        stats = state_manager.get_daily_stats(starting_balance=1200.0)
        assert stats.trade_count == 0  # Reset
        assert stats.starting_balance == 1200.0  # New balance

    def test_update_daily_stats(self, state_manager, state_dir):
        """Should persist stats updates."""
        stats = state_manager.get_daily_stats(starting_balance=1000.0)
        stats.trade_count = 3
        stats.realized_pnl = -5.0

        state_manager.update_daily_stats(stats)

        # Reload and verify
        loaded = state_manager.get_daily_stats(starting_balance=1000.0)
        assert loaded.trade_count == 3
        assert loaded.realized_pnl == -5.0

    def test_log_order(self, state_manager, state_dir):
        """Should append order to log file."""
        order_data = {
            "order_id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.001,
            "price": 50000.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Signal accepted",
        }

        state_manager.log_order(order_data)

        log_file = state_dir / "order_log.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            line = f.readline()
            logged = json.loads(line)
            assert logged["order_id"] == "12345"

    def test_log_order_appends(self, state_manager, state_dir):
        """Multiple orders should append to log."""
        for i in range(3):
            state_manager.log_order({"order_id": str(i)})

        log_file = state_dir / "order_log.jsonl"
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 3

    def test_get_order_history(self, state_manager):
        """Should read order history from log."""
        for i in range(5):
            state_manager.log_order({"order_id": str(i), "symbol": "BTC/USDT"})

        history = state_manager.get_order_history(limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0]["order_id"] == "4"

    def test_activate_kill_switch(self, state_manager, state_dir):
        """Should create STOP file."""
        state_manager.activate_kill_switch("Emergency stop")

        stop_file = state_dir / "STOP"
        assert stop_file.exists()
        assert "Emergency stop" in stop_file.read_text()

    def test_deactivate_kill_switch(self, state_manager, state_dir):
        """Should remove STOP file."""
        (state_dir / "STOP").touch()
        assert state_manager.is_kill_switch_active() is True

        state_manager.deactivate_kill_switch()
        assert state_manager.is_kill_switch_active() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_state_manager.py -v`
Expected: FAIL with "No module named 'execution.state_manager'"

- [ ] **Step 3: Implement StateManager**

```python
# execution/state_manager.py
"""
State management for execution layer.

Handles:
- Daily statistics persistence
- Order logging
- Kill switch detection
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from execution.models import DailyStats


class StateManager:
    """
    Manages persistent state for execution layer.

    State files:
    - daily_stats.json: Daily trade counts, P&L, timestamps
    - order_log.jsonl: Append-only order history
    - STOP: Kill switch file (presence halts all trading)
    """

    def __init__(self, state_dir: Path):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.daily_stats_path = self.state_dir / "daily_stats.json"
        self.order_log_path = self.state_dir / "order_log.jsonl"
        self.kill_switch_path = self.state_dir / "STOP"

    def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is active.

        Returns:
            True if STOP file exists
        """
        return self.kill_switch_path.exists()

    def activate_kill_switch(self, reason: str) -> None:
        """
        Activate kill switch by creating STOP file.

        Args:
            reason: Reason for activation (written to file)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"Kill switch activated at {timestamp}\nReason: {reason}\n"
        self.kill_switch_path.write_text(content)
        logger.warning("Kill switch activated", reason=reason)

    def deactivate_kill_switch(self) -> None:
        """Remove kill switch file to resume trading."""
        if self.kill_switch_path.exists():
            self.kill_switch_path.unlink()
            logger.info("Kill switch deactivated")

    def get_daily_stats(self, starting_balance: float) -> DailyStats:
        """
        Get or create daily statistics.

        If stats exist for today, load them.
        If stats are from a previous day, create fresh stats.

        Args:
            starting_balance: Current balance (used for new day)

        Returns:
            DailyStats for today
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self.daily_stats_path.exists():
            try:
                with open(self.daily_stats_path) as f:
                    data = json.load(f)
                    stats = DailyStats.model_validate(data)

                    # Check if stats are from today
                    if stats.date == today:
                        return stats

                    # Stats from previous day - reset
                    logger.info(
                        "New trading day detected",
                        previous_date=stats.date,
                        new_date=today
                    )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to load daily stats, creating new", error=str(e))

        # Create fresh stats for today
        return DailyStats(
            date=today,
            trade_count=0,
            realized_pnl=0.0,
            starting_balance=starting_balance,
            last_order_timestamp=None,
        )

    def update_daily_stats(self, stats: DailyStats) -> None:
        """
        Persist daily statistics.

        Args:
            stats: Updated DailyStats to save
        """
        with open(self.daily_stats_path, "w") as f:
            f.write(stats.model_dump_json(indent=2))

        logger.debug(
            "Daily stats updated",
            trade_count=stats.trade_count,
            realized_pnl=stats.realized_pnl
        )

    def log_order(self, order_data: dict) -> None:
        """
        Append order to log file.

        Args:
            order_data: Order details to log
        """
        # Add timestamp if not present
        if "logged_at" not in order_data:
            order_data["logged_at"] = datetime.now(timezone.utc).isoformat()

        with open(self.order_log_path, "a") as f:
            f.write(json.dumps(order_data) + "\n")

        logger.info(
            "Order logged",
            order_id=order_data.get("order_id"),
            symbol=order_data.get("symbol"),
            side=order_data.get("side")
        )

    def get_order_history(self, limit: int = 100) -> list[dict]:
        """
        Read recent order history.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of order dicts, most recent first
        """
        if not self.order_log_path.exists():
            return []

        orders = []
        with open(self.order_log_path) as f:
            for line in f:
                try:
                    orders.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Return most recent first, limited
        return orders[-limit:][::-1]
```

- [ ] **Step 4: Run tests to verify state manager**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_state_manager.py -v`
Expected: All tests pass

- [ ] **Step 5: Update execution/__init__.py**

Add StateManager to exports:

```python
# Add to execution/__init__.py imports:
from execution.state_manager import StateManager

# Add to __all__:
    "StateManager",
```

- [ ] **Step 6: Commit**

```bash
git add execution/state_manager.py tests/test_execution/test_state_manager.py execution/__init__.py
git commit -m "feat(execution): add state management for daily stats, order logs, and kill switch"
```

---

## Task 5: Implement Position Sizing

**Files:**
- Create: `/Users/javierlee/Trading Swarm/execution/position_sizing.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_position_sizing.py`

- [ ] **Step 1: Write position sizing tests**

```python
# tests/test_execution/test_position_sizing.py
"""Tests for fee-aware position sizing."""

import pytest
from config.fee_model import FeeModelSettings
from execution.position_sizing import calculate_position_size, PositionSizeResult


class TestCalculatePositionSize:
    """Test position size calculation."""

    @pytest.fixture
    def fee_model(self):
        """Default fee model."""
        return FeeModelSettings()

    def test_basic_position_size(self, fee_model):
        """Test basic position sizing with risk."""
        result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,  # 1% risk
            entry_price=50000.0,
            stop_price=49000.0,  # 2% stop
            fee_model=fee_model,
            max_position_pct=0.10,  # 10% max
        )

        assert isinstance(result, PositionSizeResult)
        assert result.amount > 0
        assert result.notional > 0
        assert result.risk_amount <= 100.0  # 1% of 10000

    def test_fee_reduces_position_size(self, fee_model):
        """Fees should reduce position size compared to no fees."""
        no_fee_model = FeeModelSettings(
            maker_fee_pct=0.0,
            taker_fee_pct=0.0,
            slippage_pct=0.0,
            funding_rate_pct=0.0,
        )

        result_with_fees = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=49000.0,
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        result_no_fees = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=49000.0,
            fee_model=no_fee_model,
            max_position_pct=0.10,
        )

        # Position with fees should be smaller
        assert result_with_fees.amount < result_no_fees.amount

    def test_max_position_cap(self, fee_model):
        """Position should not exceed max_position_pct."""
        # Very large risk tolerance that would exceed max
        result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.50,  # 50% risk (extreme)
            entry_price=50000.0,
            stop_price=49000.0,
            fee_model=fee_model,
            max_position_pct=0.02,  # 2% max
        )

        # Position notional should not exceed 2% of balance
        max_notional = 10000.0 * 0.02
        assert result.notional <= max_notional
        assert result.capped_by_max is True

    def test_long_position_sizing(self, fee_model):
        """Test long position with stop below entry."""
        result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=49000.0,  # 2% below
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        # Risk amount should be approximately 1% of balance
        assert 90.0 <= result.risk_amount <= 100.0

    def test_short_position_sizing(self, fee_model):
        """Test short position with stop above entry."""
        result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=51000.0,  # 2% above (stop for short)
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        assert result.amount > 0
        # Risk calculation should work the same

    def test_zero_balance_returns_zero(self, fee_model):
        """Zero balance should return zero position."""
        result = calculate_position_size(
            balance=0.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=49000.0,
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        assert result.amount == 0.0
        assert result.notional == 0.0

    def test_entry_equals_stop_returns_zero(self, fee_model):
        """Entry equal to stop should return zero (infinite risk)."""
        result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=50000.0,  # Same as entry
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        assert result.amount == 0.0
        assert "invalid" in result.reason.lower() or "zero" in result.reason.lower()

    def test_negative_risk_raises_error(self, fee_model):
        """Negative risk percentage should raise error."""
        with pytest.raises(ValueError):
            calculate_position_size(
                balance=10000.0,
                risk_pct=-0.01,
                entry_price=50000.0,
                stop_price=49000.0,
                fee_model=fee_model,
                max_position_pct=0.10,
            )

    def test_position_size_result_fields(self, fee_model):
        """Test all fields in PositionSizeResult."""
        result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=49000.0,
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        assert hasattr(result, "amount")
        assert hasattr(result, "notional")
        assert hasattr(result, "risk_amount")
        assert hasattr(result, "stop_distance_pct")
        assert hasattr(result, "fees_included")
        assert hasattr(result, "capped_by_max")
        assert hasattr(result, "reason")

    def test_very_tight_stop(self, fee_model):
        """Very tight stop should result in smaller position."""
        wide_stop_result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=45000.0,  # 10% stop
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        tight_stop_result = calculate_position_size(
            balance=10000.0,
            risk_pct=0.01,
            entry_price=50000.0,
            stop_price=49500.0,  # 1% stop
            fee_model=fee_model,
            max_position_pct=0.10,
        )

        # Tighter stop allows larger position for same risk
        assert tight_stop_result.amount > wide_stop_result.amount
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_position_sizing.py -v`
Expected: FAIL with "No module named 'execution.position_sizing'"

- [ ] **Step 3: Implement position sizing**

```python
# execution/position_sizing.py
"""
Fee-aware position sizing for risk management.

Calculates position size based on:
- Account balance
- Risk per trade (% of balance willing to lose)
- Entry and stop prices
- Trading fees
- Maximum position cap
"""

from dataclasses import dataclass

from config.fee_model import FeeModelSettings


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    amount: float  # Position size in base currency
    notional: float  # Position value in quote currency
    risk_amount: float  # Actual risk in quote currency
    stop_distance_pct: float  # Distance to stop as percentage
    fees_included: float  # Total fees deducted from risk budget
    capped_by_max: bool  # Whether max position cap was applied
    reason: str  # Explanation of calculation


def calculate_position_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    fee_model: FeeModelSettings,
    max_position_pct: float = 0.02,
) -> PositionSizeResult:
    """
    Calculate fee-aware position size.

    The position size is calculated to risk exactly `risk_pct` of the balance,
    accounting for fees and the distance to the stop loss.

    Formula:
        risk_amount = balance * risk_pct
        stop_distance_pct = |entry - stop| / entry
        total_loss_pct = stop_distance_pct + round_trip_fees
        max_notional_from_risk = risk_amount / total_loss_pct
        max_notional_from_cap = balance * max_position_pct
        notional = min(max_notional_from_risk, max_notional_from_cap)
        amount = notional / entry_price

    Args:
        balance: Account balance in quote currency (e.g., USDT)
        risk_pct: Risk per trade as decimal (0.01 = 1%)
        entry_price: Entry price
        stop_price: Stop loss price
        fee_model: Fee model for cost calculation
        max_position_pct: Maximum position as fraction of balance (0.02 = 2%)

    Returns:
        PositionSizeResult with calculated position details

    Raises:
        ValueError: If risk_pct is negative
    """
    if risk_pct < 0:
        raise ValueError("risk_pct cannot be negative")

    # Handle edge cases
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

    if entry_price <= 0:
        return PositionSizeResult(
            amount=0.0,
            notional=0.0,
            risk_amount=0.0,
            stop_distance_pct=0.0,
            fees_included=0.0,
            capped_by_max=False,
            reason="Invalid entry price",
        )

    # Calculate stop distance
    stop_distance = abs(entry_price - stop_price)
    stop_distance_pct = stop_distance / entry_price

    if stop_distance_pct == 0:
        return PositionSizeResult(
            amount=0.0,
            notional=0.0,
            risk_amount=0.0,
            stop_distance_pct=0.0,
            fees_included=0.0,
            capped_by_max=False,
            reason="Zero stop distance (entry equals stop)",
        )

    # Calculate fees (as decimal, not percentage)
    # Use 1 holding period as conservative estimate for stop-loss scenarios
    fees_pct = fee_model.round_trip_cost_pct(holding_periods_8h=1) / 100

    # Total potential loss percentage includes stop distance plus fees
    total_loss_pct = stop_distance_pct + fees_pct

    # Risk amount in quote currency
    risk_amount = balance * risk_pct

    # Calculate position size from risk
    max_notional_from_risk = risk_amount / total_loss_pct

    # Apply position cap
    max_notional_from_cap = balance * max_position_pct

    # Use smaller of the two limits
    capped_by_max = max_notional_from_risk > max_notional_from_cap
    notional = min(max_notional_from_risk, max_notional_from_cap)

    # Calculate amount in base currency
    amount = notional / entry_price

    # Recalculate actual risk with final position
    actual_risk = notional * total_loss_pct

    reason = (
        f"Position sized for {risk_pct*100:.1f}% risk with {stop_distance_pct*100:.2f}% stop"
        + (f" (capped at {max_position_pct*100:.1f}% max)" if capped_by_max else "")
    )

    return PositionSizeResult(
        amount=amount,
        notional=notional,
        risk_amount=actual_risk,
        stop_distance_pct=stop_distance_pct * 100,  # Return as percentage
        fees_included=fees_pct * 100,  # Return as percentage
        capped_by_max=capped_by_max,
        reason=reason,
    )
```

- [ ] **Step 4: Run tests to verify position sizing**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_position_sizing.py -v`
Expected: All tests pass

- [ ] **Step 5: Update execution/__init__.py**

Add position sizing to exports:

```python
# Add to execution/__init__.py imports:
from execution.position_sizing import calculate_position_size, PositionSizeResult

# Add to __all__:
    "calculate_position_size",
    "PositionSizeResult",
```

- [ ] **Step 6: Commit**

```bash
git add execution/position_sizing.py tests/test_execution/test_position_sizing.py execution/__init__.py
git commit -m "feat(execution): add fee-aware position sizing with risk management"
```

---

## Task 6: Implement BinanceExecutionClient - Core

**Files:**
- Create: `/Users/javierlee/Trading Swarm/execution/binance_client.py`
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_binance_client.py`

- [ ] **Step 1: Write core client tests**

```python
# tests/test_execution/test_binance_client.py
"""Tests for BinanceExecutionClient."""

import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from execution.binance_client import BinanceExecutionClient
from execution.exceptions import (
    LiveTradingNotAllowedError,
    KillSwitchActiveError,
)
from execution.models import OrderResult, OrderStatus, Position
from config.fee_model import FeeModelSettings


class TestBinanceClientInit:
    """Test client initialization."""

    def test_default_testnet_true(self):
        """Client should default to testnet=True."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_binance.return_value = MagicMock()

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
            )

            assert client.testnet is True

    def test_live_trading_requires_env_var(self):
        """Live trading without ALLOW_LIVE_TRADING should raise."""
        # Ensure env var is not set
        os.environ.pop("ALLOW_LIVE_TRADING", None)

        with pytest.raises(LiveTradingNotAllowedError):
            BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=False,
            )

    def test_live_trading_allowed_with_env_var(self):
        """Live trading should work with ALLOW_LIVE_TRADING=true."""
        os.environ["ALLOW_LIVE_TRADING"] = "true"

        try:
            with patch("execution.binance_client.ccxt.binance") as mock_binance:
                mock_binance.return_value = MagicMock()

                client = BinanceExecutionClient(
                    api_key="test_key",
                    secret="test_secret",
                    mode="futures",
                    testnet=False,
                )

                assert client.testnet is False
        finally:
            del os.environ["ALLOW_LIVE_TRADING"]

    def test_sandbox_mode_enabled_for_testnet(self):
        """CCXT sandbox mode should be set for testnet."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
            )

            mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    def test_rate_limit_enabled(self):
        """CCXT rate limiting should be enabled."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
            )

            # Check that enableRateLimit was passed
            call_kwargs = mock_binance.call_args[0][0]
            assert call_kwargs["enableRateLimit"] is True


class TestBinanceClientOrders:
    """Test order operations."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_place_limit_order(self, client):
        """Test placing a limit order."""
        client._exchange.create_limit_order = AsyncMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "amount": 0.001,
            "price": 50000.0,
            "status": "open",
            "filled": 0.0,
            "remaining": 0.001,
            "cost": 0.0,
            "fee": {"cost": 0.0},
            "timestamp": 1712419200000,
        })

        result = await client.place_limit_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            price=50000.0,
        )

        assert isinstance(result, OrderResult)
        assert result.order_id == "12345"
        assert result.status == "open"

    @pytest.mark.asyncio
    async def test_place_market_order(self, client):
        """Test placing a market order."""
        client._exchange.create_market_order = AsyncMock(return_value={
            "id": "12346",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "market",
            "amount": 0.001,
            "price": None,
            "status": "closed",
            "filled": 0.001,
            "remaining": 0.0,
            "cost": 50.0,
            "average": 50000.0,
            "fee": {"cost": 0.025},
            "timestamp": 1712419200000,
        })

        result = await client.place_market_order(
            symbol="BTC/USDT",
            side="sell",
            amount=0.001,
        )

        assert isinstance(result, OrderResult)
        assert result.order_id == "12346"
        assert result.filled == 0.001

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test canceling an order."""
        client._exchange.cancel_order = AsyncMock(return_value={
            "id": "12345",
            "status": "canceled",
        })

        result = await client.cancel_order(
            order_id="12345",
            symbol="BTC/USDT",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_order_status(self, client):
        """Test getting order status."""
        client._exchange.fetch_order = AsyncMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT",
            "status": "closed",
            "filled": 0.001,
            "remaining": 0.0,
            "average": 50000.0,
        })

        status = await client.get_order_status(
            order_id="12345",
            symbol="BTC/USDT",
        )

        assert isinstance(status, OrderStatus)
        assert status.is_complete is True


class TestBinanceClientPositions:
    """Test position management."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_get_balance(self, client):
        """Test getting account balance."""
        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 1000.0, "used": 100.0, "total": 1100.0},
        })

        balance = await client.get_balance("USDT")

        assert balance == 1000.0  # Free balance

    @pytest.mark.asyncio
    async def test_get_open_positions(self, client):
        """Test getting open positions."""
        client._exchange.fetch_positions = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.001,
                "entryPrice": 50000.0,
                "markPrice": 51000.0,
                "unrealizedPnl": 1.0,
                "leverage": 1,
                "liquidationPrice": 40000.0,
            },
        ])

        positions = await client.get_open_positions()

        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].side == "long"

    @pytest.mark.asyncio
    async def test_close_position(self, client):
        """Test closing a position."""
        # Mock fetching the position first
        client._exchange.fetch_positions = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.001,
                "entryPrice": 50000.0,
                "markPrice": 51000.0,
                "unrealizedPnl": 1.0,
                "leverage": 1,
            },
        ])

        # Mock closing order
        client._exchange.create_market_order = AsyncMock(return_value={
            "id": "close_123",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "market",
            "amount": 0.001,
            "price": None,
            "status": "closed",
            "filled": 0.001,
            "remaining": 0.0,
            "cost": 51.0,
            "fee": {"cost": 0.025},
            "timestamp": 1712419200000,
        })

        result = await client.close_position("BTC/USDT")

        assert isinstance(result, OrderResult)
        assert result.side == "sell"  # Opposite of long position
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_binance_client.py -v`
Expected: FAIL with "No module named 'execution.binance_client'"

- [ ] **Step 3: Implement BinanceExecutionClient**

```python
# execution/binance_client.py
"""
Binance execution client for deploying trading signals.

Wraps CCXT with:
- Testnet-first safety (live trading requires explicit opt-in)
- Kill switch support
- Daily loss circuit breaker
- Position limits
- Order cooldowns
- Comprehensive logging
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt.async_support as ccxt
from loguru import logger

from config.fee_model import FeeModelSettings
from config.settings import settings

from execution.exceptions import (
    ExecutionError,
    KillSwitchActiveError,
    DailyLossLimitError,
    PositionLimitError,
    CooldownActiveError,
    DailyTradeCountError,
    InsufficientBalanceError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
    SignalRejectedError,
)
from execution.models import (
    OrderResult,
    OrderStatus,
    Position,
    TradeDecision,
    DailyStats,
    SignalInput,
)
from execution.state_manager import StateManager
from execution.position_sizing import calculate_position_size


class BinanceExecutionClient:
    """
    Production execution client for Binance.

    Features:
    - Testnet by default (live requires ALLOW_LIVE_TRADING=true)
    - Kill switch via STOP file
    - Daily loss circuit breaker
    - Position and trade count limits
    - Order cooldown
    - Fee-aware position sizing
    - Comprehensive order logging

    Usage:
        client = BinanceExecutionClient(
            api_key="...",
            secret="...",
            mode="futures",
            testnet=True,
        )

        # Check if a signal should be executed
        decision = await client.accept_signal(signal_json)

        if decision.execute:
            # Confirm with user before actually placing
            result = await client.place_limit_order(
                decision.symbol,
                decision.side,
                decision.amount,
                decision.price,
            )
    """

    def __init__(
        self,
        api_key: str,
        secret: str,
        mode: Literal["spot", "futures"],
        testnet: bool = True,
        state_dir: Path | None = None,
        max_daily_trades: int | None = None,
        max_daily_loss_pct: float | None = None,
        max_open_positions: int | None = None,
        max_position_pct: float | None = None,
        order_cooldown_seconds: int | None = None,
        min_confidence: float | None = None,
    ):
        """
        Initialize Binance execution client.

        Args:
            api_key: Binance API key
            secret: Binance API secret
            mode: Trading mode ('spot' or 'futures')
            testnet: Use testnet (MUST be True unless ALLOW_LIVE_TRADING=true)
            state_dir: Directory for state files (default: execution/state)
            max_daily_trades: Override max daily trades
            max_daily_loss_pct: Override max daily loss percentage
            max_open_positions: Override max open positions
            max_position_pct: Override max position percentage
            order_cooldown_seconds: Override order cooldown
            min_confidence: Override minimum signal confidence

        Raises:
            LiveTradingNotAllowedError: If testnet=False without ALLOW_LIVE_TRADING=true
        """
        # Safety check: live trading requires explicit opt-in
        if not testnet:
            allow_live = os.environ.get("ALLOW_LIVE_TRADING", "").lower() == "true"
            if not allow_live:
                raise LiveTradingNotAllowedError(
                    "Live trading requires ALLOW_LIVE_TRADING=true environment variable. "
                    "This is a safety measure to prevent accidental live trading."
                )

        self.testnet = testnet
        self.mode = mode

        # Initialize CCXT exchange
        exchange_config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": mode,  # 'spot' or 'futures'
            },
        }

        self._exchange = ccxt.binance(exchange_config)

        # Enable sandbox mode for testnet
        if testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("Binance client initialized in TESTNET mode")
        else:
            logger.warning("Binance client initialized in LIVE mode - real money at risk!")

        # Initialize state manager
        state_dir = state_dir or settings.execution.state_dir
        self._state = StateManager(state_dir)

        # Load safety limits (use overrides or settings)
        self.max_daily_trades = max_daily_trades or settings.execution.max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct or settings.execution.max_daily_loss_pct
        self.max_open_positions = max_open_positions or settings.execution.max_open_positions
        self.max_position_pct = max_position_pct or settings.execution.max_position_pct
        self.order_cooldown_seconds = order_cooldown_seconds or settings.execution.order_cooldown_seconds
        self.min_confidence = min_confidence or settings.execution.min_confidence

        logger.info(
            "Execution client configured",
            mode=mode,
            testnet=testnet,
            max_daily_trades=self.max_daily_trades,
            max_daily_loss_pct=self.max_daily_loss_pct,
            max_open_positions=self.max_open_positions,
        )

    async def close(self) -> None:
        """Close exchange connection."""
        await self._exchange.close()

    async def __aenter__(self) -> "BinanceExecutionClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()

    # =========================================================================
    # Order Methods
    # =========================================================================

    async def place_limit_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order quantity in base currency
            price: Limit price

        Returns:
            OrderResult with order details

        Raises:
            KillSwitchActiveError: If STOP file exists
            OrderRejectedError: If exchange rejects order
        """
        self._check_kill_switch()

        try:
            response = await self._exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
            )

            result = self._parse_order_response(response)

            # Log the order
            self._state.log_order({
                "order_id": result.order_id,
                "symbol": symbol,
                "side": side,
                "order_type": "limit",
                "amount": amount,
                "price": price,
                "status": result.status,
                "testnet": self.testnet,
            })

            return result

        except ccxt.BaseError as e:
            raise OrderRejectedError(str(e), exchange_error=type(e).__name__)

    async def place_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order quantity in base currency

        Returns:
            OrderResult with order details

        Raises:
            KillSwitchActiveError: If STOP file exists
            OrderRejectedError: If exchange rejects order
        """
        self._check_kill_switch()

        try:
            response = await self._exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
            )

            result = self._parse_order_response(response)

            # Log the order
            self._state.log_order({
                "order_id": result.order_id,
                "symbol": symbol,
                "side": side,
                "order_type": "market",
                "amount": amount,
                "price": response.get("average"),
                "filled": result.filled,
                "fee": result.fee,
                "status": result.status,
                "testnet": self.testnet,
            })

            return result

        except ccxt.BaseError as e:
            raise OrderRejectedError(str(e), exchange_error=type(e).__name__)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            True if cancelled successfully
        """
        try:
            await self._exchange.cancel_order(order_id, symbol)

            self._state.log_order({
                "action": "cancel",
                "order_id": order_id,
                "symbol": symbol,
                "testnet": self.testnet,
            })

            return True
        except ccxt.BaseError as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            OrderStatus with current state
        """
        response = await self._exchange.fetch_order(order_id, symbol)

        return OrderStatus(
            order_id=response["id"],
            symbol=response["symbol"],
            status=response["status"],
            filled=response.get("filled", 0.0),
            remaining=response.get("remaining", 0.0),
            average_price=response.get("average"),
        )

    # =========================================================================
    # Position Management
    # =========================================================================

    async def get_balance(self, asset: str = "USDT") -> float:
        """
        Get available balance for an asset.

        Args:
            asset: Asset symbol (default: 'USDT')

        Returns:
            Available (free) balance
        """
        balance = await self._exchange.fetch_balance()
        return balance.get(asset, {}).get("free", 0.0)

    async def get_open_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        if self.mode != "futures":
            # Spot doesn't have positions in the same sense
            return []

        positions = await self._exchange.fetch_positions()

        result = []
        for pos in positions:
            # Skip positions with zero amount
            contracts = float(pos.get("contracts", 0))
            if contracts == 0:
                continue

            result.append(Position(
                symbol=pos["symbol"].split(":")[0],  # Remove :USDT suffix
                side=pos.get("side", "long"),
                amount=contracts,
                entry_price=float(pos.get("entryPrice", 0)),
                mark_price=float(pos.get("markPrice", 0)),
                unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                leverage=int(pos.get("leverage", 1)),
                liquidation_price=pos.get("liquidationPrice"),
            ))

        return result

    async def close_position(self, symbol: str) -> OrderResult:
        """
        Close an open position with a market order.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            OrderResult of the closing order

        Raises:
            ExecutionError: If no position exists for symbol
        """
        positions = await self.get_open_positions()

        # Find position for symbol
        position = None
        for pos in positions:
            if pos.symbol == symbol or pos.symbol.startswith(symbol):
                position = pos
                break

        if position is None:
            raise ExecutionError(f"No open position for {symbol}")

        # Close with opposite side market order
        close_side = "sell" if position.side == "long" else "buy"

        return await self.place_market_order(
            symbol=symbol,
            side=close_side,
            amount=position.amount,
        )

    # =========================================================================
    # Signal Acceptance
    # =========================================================================

    async def accept_signal(
        self,
        signal: SignalInput | dict,
        fee_model: FeeModelSettings | None = None,
    ) -> TradeDecision:
        """
        Evaluate a signal and return a trade decision.

        This method does NOT place an order. It evaluates whether the signal
        should be executed based on:
        1. Kill switch status
        2. Daily trade count
        3. Daily loss limit
        4. Open position limit
        5. Order cooldown
        6. Signal confidence
        7. Expected return vs fee threshold

        Args:
            signal: Signal from signal generation pipeline
            fee_model: Fee model for profitability check (default: from settings)

        Returns:
            TradeDecision with execute flag and parameters
        """
        # Parse signal if dict
        if isinstance(signal, dict):
            signal = SignalInput.model_validate(signal)

        fee_model = fee_model or settings.fee_model

        # Check 1: Kill switch
        if self._state.is_kill_switch_active():
            return TradeDecision(
                execute=False,
                reason="Kill switch active - all trading halted",
            )

        # Get current balance for stats
        try:
            balance = await self.get_balance()
        except Exception as e:
            return TradeDecision(
                execute=False,
                reason=f"Failed to fetch balance: {e}",
            )

        # Check 2: Daily stats limits
        stats = self._state.get_daily_stats(starting_balance=balance)

        # Check 2a: Trade count
        if stats.trade_count >= self.max_daily_trades:
            return TradeDecision(
                execute=False,
                reason=f"Daily trade limit reached: {stats.trade_count}/{self.max_daily_trades}",
            )

        # Check 2b: Daily loss
        if stats.daily_loss_pct >= self.max_daily_loss_pct:
            # Activate kill switch for severe losses
            if stats.daily_loss_pct >= self.max_daily_loss_pct * 1.5:
                self._state.activate_kill_switch(
                    f"Automatic: Daily loss {stats.daily_loss_pct:.2f}% exceeded 1.5x limit"
                )
            return TradeDecision(
                execute=False,
                reason=f"Daily loss limit breached: {stats.daily_loss_pct:.2f}% >= {self.max_daily_loss_pct}%",
            )

        # Check 3: Position limit
        try:
            positions = await self.get_open_positions()
            if len(positions) >= self.max_open_positions:
                return TradeDecision(
                    execute=False,
                    reason=f"Position limit reached: {len(positions)}/{self.max_open_positions}",
                )
        except Exception as e:
            return TradeDecision(
                execute=False,
                reason=f"Failed to fetch positions: {e}",
            )

        # Check 4: Cooldown
        if stats.last_order_timestamp:
            seconds_since_last = (
                datetime.now(timezone.utc) - stats.last_order_timestamp
            ).total_seconds()
            if seconds_since_last < self.order_cooldown_seconds:
                remaining = self.order_cooldown_seconds - seconds_since_last
                return TradeDecision(
                    execute=False,
                    reason=f"Order cooldown active: {remaining:.1f}s remaining",
                )

        # Check 5: Signal confidence
        if signal.confidence < self.min_confidence:
            return TradeDecision(
                execute=False,
                reason=f"Signal confidence too low: {signal.confidence:.2f} < {self.min_confidence}",
            )

        # Check 6: Expected return vs fee threshold
        min_profitable = fee_model.minimum_profitable_return_pct(holding_periods_8h=1)
        if signal.expected_return_pct < min_profitable:
            return TradeDecision(
                execute=False,
                reason=f"Expected return {signal.expected_return_pct:.3f}% below fee threshold {min_profitable:.3f}%",
            )

        # All checks passed - calculate position size
        entry_price = signal.entry_price
        if entry_price is None:
            # Would need to fetch current price
            return TradeDecision(
                execute=False,
                reason="Signal missing entry_price - cannot calculate position size",
            )

        # Calculate stop price from stop_loss_pct
        if signal.direction == "long":
            stop_price = entry_price * (1 - signal.stop_loss_pct / 100)
        else:
            stop_price = entry_price * (1 + signal.stop_loss_pct / 100)

        # Calculate position size
        from execution.position_sizing import calculate_position_size

        size_result = calculate_position_size(
            balance=balance,
            risk_pct=0.01,  # 1% risk per trade
            entry_price=entry_price,
            stop_price=stop_price,
            fee_model=fee_model,
            max_position_pct=self.max_position_pct,
        )

        if size_result.amount <= 0:
            return TradeDecision(
                execute=False,
                reason=f"Position size calculation failed: {size_result.reason}",
            )

        # Calculate take profit price if provided
        take_profit_price = None
        if signal.take_profit_pct:
            if signal.direction == "long":
                take_profit_price = entry_price * (1 + signal.take_profit_pct / 100)
            else:
                take_profit_price = entry_price * (1 - signal.take_profit_pct / 100)

        # Build decision
        side = "buy" if signal.direction == "long" else "sell"

        return TradeDecision(
            execute=True,
            reason=f"All safety checks passed. {size_result.reason}",
            symbol=signal.symbol,
            side=side,
            amount=size_result.amount,
            price=entry_price,
            order_type="limit",
            stop_loss_price=stop_price,
            take_profit_price=take_profit_price,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _check_kill_switch(self) -> None:
        """Raise if kill switch is active."""
        if self._state.is_kill_switch_active():
            raise KillSwitchActiveError(
                "STOP file detected - all trading halted. "
                "Remove execution/state/STOP to resume."
            )

    def _parse_order_response(self, response: dict) -> OrderResult:
        """Parse CCXT order response into OrderResult."""
        fee = response.get("fee", {})
        fee_cost = fee.get("cost", 0.0) if isinstance(fee, dict) else 0.0

        return OrderResult(
            order_id=str(response["id"]),
            symbol=response["symbol"],
            side=response["side"],
            order_type=response["type"],
            amount=float(response.get("amount", 0)),
            price=response.get("price"),
            status=response["status"],
            filled=float(response.get("filled", 0)),
            remaining=float(response.get("remaining", 0)),
            cost=float(response.get("cost", 0)),
            fee=float(fee_cost),
            timestamp=datetime.fromtimestamp(
                response.get("timestamp", 0) / 1000,
                tz=timezone.utc
            ),
        )
```

- [ ] **Step 4: Run tests to verify client**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_binance_client.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add execution/binance_client.py tests/test_execution/test_binance_client.py
git commit -m "feat(execution): implement BinanceExecutionClient with order methods and position management"
```

---

## Task 7: Implement Safety Control Tests

**Files:**
- Test: `/Users/javierlee/Trading Swarm/tests/test_execution/test_safety_controls.py`

- [ ] **Step 1: Write comprehensive safety control tests**

```python
# tests/test_execution/test_safety_controls.py
"""Tests for execution safety controls."""

import os
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

from execution.binance_client import BinanceExecutionClient
from execution.models import SignalInput, DailyStats
from execution.exceptions import KillSwitchActiveError
from config.fee_model import FeeModelSettings


class TestKillSwitch:
    """Test kill switch functionality."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_limit_orders(self, client, state_dir):
        """Kill switch should block limit orders."""
        (state_dir / "STOP").write_text("Emergency stop")

        with pytest.raises(KillSwitchActiveError):
            await client.place_limit_order(
                symbol="BTC/USDT",
                side="buy",
                amount=0.001,
                price=50000.0,
            )

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_market_orders(self, client, state_dir):
        """Kill switch should block market orders."""
        (state_dir / "STOP").write_text("Emergency stop")

        with pytest.raises(KillSwitchActiveError):
            await client.place_market_order(
                symbol="BTC/USDT",
                side="sell",
                amount=0.001,
            )

    @pytest.mark.asyncio
    async def test_kill_switch_rejects_signals(self, client, state_dir):
        """Kill switch should reject all signals."""
        (state_dir / "STOP").touch()

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "kill switch" in decision.reason.lower()


class TestDailyLossLimit:
    """Test daily loss circuit breaker."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
                max_daily_loss_pct=2.0,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_daily_loss_limit_blocks_signals(self, client, state_dir):
        """Signals should be rejected when daily loss exceeds limit."""
        # Create stats with loss exceeding limit
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stats = DailyStats(
            date=today,
            trade_count=5,
            realized_pnl=-250.0,  # -2.5% of 10000
            starting_balance=10000.0,
        )

        import json
        with open(state_dir / "daily_stats.json", "w") as f:
            f.write(stats.model_dump_json())

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 9750.0}
        })

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "loss limit" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_severe_loss_activates_kill_switch(self, client, state_dir):
        """Severe losses should automatically activate kill switch."""
        # Create stats with loss exceeding 1.5x limit (3% when limit is 2%)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stats = DailyStats(
            date=today,
            trade_count=10,
            realized_pnl=-350.0,  # -3.5% of 10000 (>1.5x of 2%)
            starting_balance=10000.0,
        )

        import json
        with open(state_dir / "daily_stats.json", "w") as f:
            f.write(stats.model_dump_json())

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 9650.0}
        })

        decision = await client.accept_signal(signal)

        # Kill switch should be activated
        assert (state_dir / "STOP").exists()


class TestDailyTradeLimit:
    """Test daily trade count limit."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
                max_daily_trades=10,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_trade_limit_blocks_signals(self, client, state_dir):
        """Signals should be rejected when daily trade count reaches limit."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stats = DailyStats(
            date=today,
            trade_count=10,  # At limit
            realized_pnl=0.0,
            starting_balance=10000.0,
        )

        import json
        with open(state_dir / "daily_stats.json", "w") as f:
            f.write(stats.model_dump_json())

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "trade limit" in decision.reason.lower()


class TestPositionLimit:
    """Test open position limit."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
                max_open_positions=3,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_position_limit_blocks_signals(self, client):
        """Signals should be rejected when at position limit."""
        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })

        # Mock 3 open positions
        client._exchange.fetch_positions = AsyncMock(return_value=[
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.001,
             "entryPrice": 50000, "markPrice": 51000, "unrealizedPnl": 1, "leverage": 1},
            {"symbol": "ETH/USDT:USDT", "side": "long", "contracts": 0.01,
             "entryPrice": 3000, "markPrice": 3100, "unrealizedPnl": 1, "leverage": 1},
            {"symbol": "SOL/USDT:USDT", "side": "short", "contracts": 1,
             "entryPrice": 100, "markPrice": 98, "unrealizedPnl": 2, "leverage": 1},
        ])

        signal = SignalInput(
            symbol="AVAX/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=25.0,
        )

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "position limit" in decision.reason.lower()


class TestOrderCooldown:
    """Test order cooldown."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
                order_cooldown_seconds=60,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_cooldown_blocks_rapid_signals(self, client, state_dir):
        """Signals should be rejected during cooldown period."""
        # Create stats with recent order
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        stats = DailyStats(
            date=today,
            trade_count=1,
            realized_pnl=0.0,
            starting_balance=10000.0,
            last_order_timestamp=recent_time,
        )

        import json
        with open(state_dir / "daily_stats.json", "w") as f:
            f.write(stats.model_dump_json())

        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })
        client._exchange.fetch_positions = AsyncMock(return_value=[])

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "cooldown" in decision.reason.lower()


class TestSignalConfidence:
    """Test signal confidence threshold."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
                min_confidence=0.7,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self, client):
        """Low confidence signals should be rejected."""
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.5,  # Below 0.7 threshold
            expected_return_pct=0.5,
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })
        client._exchange.fetch_positions = AsyncMock(return_value=[])

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "confidence" in decision.reason.lower()


class TestFeeThreshold:
    """Test expected return vs fee threshold."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_unprofitable_signal_rejected(self, client):
        """Signals with expected return below fee threshold should be rejected."""
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.9,
            expected_return_pct=0.05,  # Below typical ~0.09% fee threshold
            stop_loss_pct=1.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })
        client._exchange.fetch_positions = AsyncMock(return_value=[])

        decision = await client.accept_signal(signal)

        assert decision.execute is False
        assert "fee threshold" in decision.reason.lower() or "return" in decision.reason.lower()


class TestSuccessfulSignalAcceptance:
    """Test that valid signals are accepted."""

    @pytest.fixture
    def state_dir(self):
        """Create temporary state directory."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def client(self, state_dir):
        """Create client with mocked exchange."""
        with patch("execution.binance_client.ccxt.binance") as mock_binance:
            mock_exchange = MagicMock()
            mock_binance.return_value = mock_exchange

            client = BinanceExecutionClient(
                api_key="test_key",
                secret="test_secret",
                mode="futures",
                testnet=True,
                state_dir=state_dir,
            )
            client._exchange = mock_exchange
            yield client

    @pytest.mark.asyncio
    async def test_valid_signal_accepted(self, client):
        """Valid signals passing all checks should be accepted."""
        signal = SignalInput(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
            expected_return_pct=0.5,
            stop_loss_pct=2.0,
            take_profit_pct=3.0,
            entry_price=50000.0,
        )

        client._exchange.fetch_balance = AsyncMock(return_value={
            "USDT": {"free": 10000.0}
        })
        client._exchange.fetch_positions = AsyncMock(return_value=[])

        decision = await client.accept_signal(signal)

        assert decision.execute is True
        assert decision.symbol == "BTC/USDT"
        assert decision.side == "buy"
        assert decision.amount > 0
        assert decision.price == 50000.0
        assert decision.stop_loss_price is not None
        assert decision.take_profit_price is not None
```

- [ ] **Step 2: Run safety control tests**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/test_safety_controls.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_execution/test_safety_controls.py
git commit -m "test(execution): add comprehensive safety control tests"
```

---

## Task 8: Create State Directory and Gitignore

**Files:**
- Create: `/Users/javierlee/Trading Swarm/execution/state/.gitignore`
- Create: `/Users/javierlee/Trading Swarm/execution/state/.gitkeep`

- [ ] **Step 1: Create state directory structure**

```bash
mkdir -p /Users/javierlee/Trading\ Swarm/execution/state
```

- [ ] **Step 2: Create .gitignore for state directory**

```gitignore
# execution/state/.gitignore
# Ignore all state files except this gitignore and gitkeep
*
!.gitignore
!.gitkeep
```

- [ ] **Step 3: Create .gitkeep to preserve directory**

```bash
touch /Users/javierlee/Trading\ Swarm/execution/state/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add execution/state/.gitignore execution/state/.gitkeep
git commit -m "chore(execution): add state directory with gitignore"
```

---

## Task 9: Update CLAUDE.md

**Files:**
- Modify: `/Users/javierlee/Trading Swarm/CLAUDE.md`

- [ ] **Step 1: Add execution layer documentation to CLAUDE.md**

Add after "Session 14" in the Completed section:

```markdown
- Session 15: Binance Execution Client (execution module, safety controls, fee-aware sizing)
```

Add new section after "Fee-Aware Training (Session 14)":

```markdown
### Execution Layer (Session 15)
- **Purpose:** Production execution layer for deploying signals to Binance
- **Safety-First Design:** Testnet by default, live trading requires ALLOW_LIVE_TRADING=true
- **Kill Switch:** STOP file in execution/state/ halts all trading immediately
- **Circuit Breakers:**
  - Daily loss limit (default 2%) - auto-activates kill switch at 1.5x
  - Daily trade count limit (default 10)
  - Max open positions (default 3)
  - Order cooldown (default 60s between orders)
  - Max position size (default 2% of portfolio)
- **Fee-Aware Sizing:** Position sizing accounts for fees, respects risk limits
- **Signal Flow:** accept_signal() evaluates and returns decision; does NOT auto-execute
- **State Persistence:** daily_stats.json, order_log.jsonl in execution/state/
- **Integration:** Uses existing FeeModelSettings from config/fee_model.py
```

Add to File Index:

```markdown
### Execution Layer
- `execution/__init__.py` - Package exports
- `execution/exceptions.py` - ExecutionError hierarchy
- `execution/models.py` - OrderResult, Position, TradeDecision, DailyStats, SignalInput
- `execution/state_manager.py` - Kill switch, daily stats, order logging
- `execution/position_sizing.py` - Fee-aware position sizing
- `execution/binance_client.py` - BinanceExecutionClient main class
```

Add to Tests section:

```markdown
- `tests/test_execution/` - 50+ tests (exceptions, models, settings, state, sizing, client, safety)
```

- [ ] **Step 2: Commit documentation update**

```bash
git add CLAUDE.md
git commit -m "docs: add Session 15 execution layer to CLAUDE.md"
```

---

## Task 10: Run Full Test Suite

**Files:** None (verification only)

- [ ] **Step 1: Run all execution tests**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/test_execution/ -v`
Expected: All tests pass

- [ ] **Step 2: Run full project test suite**

Run: `pytest /Users/javierlee/Trading\ Swarm/tests/ -v --ignore=/Users/javierlee/Trading\ Swarm/tests/test_training/test_process_lock.py`
Expected: All tests pass (excluding known platform-specific failures)

- [ ] **Step 3: Verify import works**

```python
# Quick import test
python -c "from execution import BinanceExecutionClient, TradeDecision, calculate_position_size; print('Imports OK')"
```
Expected: "Imports OK"

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(execution): complete Binance execution client implementation

- BinanceExecutionClient with testnet-first safety
- Kill switch, daily loss circuit breaker, position limits
- Fee-aware position sizing with risk management
- Comprehensive state persistence and order logging
- 50+ tests covering all safety controls

Session 15 complete."
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] BinanceExecutionClient class with __init__ parameters
- [x] Order methods (place_limit_order, place_market_order, cancel_order, get_order_status)
- [x] Position management (get_balance, get_open_positions, close_position)
- [x] Fee-aware position sizing with max_position_pct cap
- [x] Safety controls (kill switch, daily loss, trade count, positions, cooldown)
- [x] accept_signal() returning TradeDecision (not auto-executing)
- [x] ExecutionSettings in config/settings.py
- [x] Tests with mocked CCXT
- [x] Kill switch tests
- [x] Daily loss circuit breaker tests
- [x] Position sizing with fee deduction tests
- [x] Live trading requires ALLOW_LIVE_TRADING=true test

**2. Placeholder scan:** None found

**3. Type consistency:** Verified all model names, method signatures match across tasks

---

Plan complete and saved to `/Users/javierlee/Trading Swarm/docs/superpowers/plans/2026-04-06-binance-execution-client.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
