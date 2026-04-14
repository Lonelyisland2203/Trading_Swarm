"""
Exception hierarchy for execution layer.

Distinguishes between different categories of execution failures:
- Safety controls (kill switch, daily loss limit, position limit, cooldown)
- Balance/funding issues
- Exchange rejections
- Configuration errors

Each exception stores relevant context for debugging and error recovery.
"""


class ExecutionError(Exception):
    """Base exception for all execution-related errors."""

    pass


class KillSwitchActiveError(ExecutionError):
    """STOP file detected - trading halted for safety."""

    def __init__(self):
        """Initialize KillSwitchActiveError."""
        super().__init__("Kill switch active - STOP file detected. Trading halted.")


class DailyLossLimitError(ExecutionError):
    """Daily loss limit exceeded."""

    def __init__(self, current_loss_pct: float, limit_pct: float):
        """
        Initialize DailyLossLimitError.

        Args:
            current_loss_pct: Current daily loss as percentage of account.
            limit_pct: Maximum allowed loss percentage.
        """
        self.current_loss_pct = current_loss_pct
        self.limit_pct = limit_pct
        super().__init__(
            f"Daily loss limit exceeded: {current_loss_pct:.2f}% (limit: {limit_pct:.2f}%)"
        )


class PositionLimitError(ExecutionError):
    """Maximum number of concurrent positions reached."""

    def __init__(self, current_positions: int, max_positions: int):
        """
        Initialize PositionLimitError.

        Args:
            current_positions: Current number of open positions.
            max_positions: Maximum allowed concurrent positions.
        """
        self.current_positions = current_positions
        self.max_positions = max_positions
        super().__init__(
            f"Position limit reached: {current_positions}/{max_positions} positions open"
        )


class CooldownActiveError(ExecutionError):
    """Order cooldown period still active."""

    def __init__(self, seconds_remaining: float):
        """
        Initialize CooldownActiveError.

        Args:
            seconds_remaining: Seconds until cooldown expires.
        """
        self.seconds_remaining = seconds_remaining
        super().__init__(f"Order cooldown active: {seconds_remaining:.1f}s remaining")


class DailyTradeCountError(ExecutionError):
    """Daily trade count limit reached."""

    def __init__(self, current_count: int, max_count: int):
        """
        Initialize DailyTradeCountError.

        Args:
            current_count: Number of trades executed today.
            max_count: Maximum allowed trades per day.
        """
        self.current_count = current_count
        self.max_count = max_count
        super().__init__(f"Daily trade limit reached: {current_count}/{max_count} trades")


class InsufficientBalanceError(ExecutionError):
    """Insufficient balance to execute order."""

    def __init__(self, required: float, available: float, asset: str):
        """
        Initialize InsufficientBalanceError.

        Args:
            required: Amount required to execute order.
            available: Currently available balance.
            asset: Asset symbol (e.g., 'USDT', 'BTC').
        """
        self.required = required
        self.available = available
        self.asset = asset
        super().__init__(
            f"Insufficient {asset} balance: required {required:.8f}, available {available:.8f}"
        )


class LiveTradingNotAllowedError(ExecutionError):
    """Live trading attempted without proper configuration/permission."""

    def __init__(self):
        """Initialize LiveTradingNotAllowedError."""
        super().__init__("Live trading not allowed - verify configuration and permissions")


class OrderRejectedError(ExecutionError):
    """Order rejected by exchange."""

    def __init__(self, reason: str, exchange_error: Exception = None):
        """
        Initialize OrderRejectedError.

        Args:
            reason: Human-readable rejection reason.
            exchange_error: Original exchange error if available.
        """
        self.reason = reason
        self.exchange_error = exchange_error
        message = f"Order rejected: {reason}"
        if exchange_error:
            message += f" (Exchange error: {str(exchange_error)})"
        super().__init__(message)


class SignalRejectedError(ExecutionError):
    """Signal failed acceptance criteria and was rejected."""

    def __init__(self, reason: str):
        """
        Initialize SignalRejectedError.

        Args:
            reason: Explanation for why signal was rejected.
        """
        self.reason = reason
        super().__init__(f"Signal rejected: {reason}")
