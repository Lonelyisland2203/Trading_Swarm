"""
Backtesting configuration.

Defines configuration options for outcome computation and verification.
"""

from dataclasses import dataclass

from loguru import logger

from .constants import DEFAULT_TXN_COST_PCT


@dataclass(slots=True, frozen=True)
class BacktestConfig:
    """
    Backtest configuration with sensible defaults.
    
    Attributes:
        txn_cost_pct: Transaction cost per trade (default 0.1%)
        entry_on: Entry price selection ("next_open" for realistic, "close" for testing)
    """
    
    txn_cost_pct: float = DEFAULT_TXN_COST_PCT
    entry_on: str = "next_open"  # "next_open" | "close" (testing only)
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.txn_cost_pct < 0:
            raise ValueError("txn_cost_pct cannot be negative")
        
        if self.txn_cost_pct > 0.01:  # More than 1% is suspicious
            logger.warning(
                "Transaction cost >1% is unusually high",
                txn_cost_pct=self.txn_cost_pct,
            )
        
        if self.entry_on not in ("next_open", "close"):
            raise ValueError(
                f"entry_on must be 'next_open' or 'close', got: '{self.entry_on}'"
            )
