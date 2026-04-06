"""
Execution layer for order management and position tracking.

This module provides the core execution infrastructure for placing orders,
managing positions, enforcing trading limits and safeguards (kill switches,
daily loss limits, position limits, order cooldowns), and handling exchange
rejections with proper validation and error reporting.

The execution layer acts as the primary interface between the signal generation
layer and the exchange, handling all state management, risk controls, and
order lifecycle management.
"""

from execution.binance_client import BinanceExecutionClient
from execution.exceptions import (
    CooldownActiveError,
    DailyLossLimitError,
    DailyTradeCountError,
    ExecutionError,
    InsufficientBalanceError,
    KillSwitchActiveError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
    PositionLimitError,
    SignalRejectedError,
)
from execution.position_sizing import PositionSizeResult, calculate_position_size
from execution.state_manager import StateManager

__all__ = [
    "BinanceExecutionClient",
    "ExecutionError",
    "KillSwitchActiveError",
    "DailyLossLimitError",
    "PositionLimitError",
    "CooldownActiveError",
    "DailyTradeCountError",
    "InsufficientBalanceError",
    "LiveTradingNotAllowedError",
    "OrderRejectedError",
    "SignalRejectedError",
    "StateManager",
    "calculate_position_size",
    "PositionSizeResult",
]
