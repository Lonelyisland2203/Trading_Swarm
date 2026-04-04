"""
Outcome computation for backtesting.

Computes realized returns, MAE, and net returns after transaction costs.
"""

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True, frozen=True)
class VerifiedOutcome:
    """
    Verified outcome from backtesting a training example.
    
    Attributes:
        example_id: UUID of training example
        actual_direction: Realized direction ("HIGHER" | "LOWER" | "FLAT")
        realized_return: Log return from entry to exit
        max_adverse_excursion: Worst drawdown during holding period (negative)
        net_return: Return after transaction costs
        entry_price: Price used for entry
        exit_price: Price used for exit
        bars_held: Number of bars in holding period
    """
    
    example_id: str
    actual_direction: str
    realized_return: float
    max_adverse_excursion: float
    net_return: float
    entry_price: float
    exit_price: float
    bars_held: int


def compute_log_return(entry_price: float, exit_price: float) -> float:
    """
    Compute log return - additive and symmetric.
    
    Log returns have desirable properties for DPO training:
    - Additivity: log returns sum across periods
    - Symmetry: +10% and -10% log returns have same magnitude
    - Distribution: closer to normal distribution
    
    Args:
        entry_price: Entry price (must be positive)
        exit_price: Exit price (must be positive)
    
    Returns:
        Log return (can be negative)
    
    Raises:
        ValueError: If prices are non-positive
        
    Example:
        >>> compute_log_return(100.0, 110.0)  # 10% gain
        0.09531...  # ln(1.1) ≈ 0.0953
        >>> compute_log_return(100.0, 90.0)   # 10% loss  
        -0.10536...  # ln(0.9) ≈ -0.1054
    """
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError(
            f"Prices must be positive: entry={entry_price}, exit={exit_price}"
        )
    
    return math.log(exit_price / entry_price)


def compute_net_return(
    log_return: float,
    txn_cost_pct: float = 0.001,  # 0.1% default
    num_trades: int = 2,          # Entry + exit
) -> float:
    """
    Compute net return after transaction costs (still log scale).
    
    Transaction costs reduce return by approximately cost * num_trades
    for small costs. We use exact log arithmetic for precision.
    
    Args:
        log_return: Gross log return (before costs)
        txn_cost_pct: Transaction cost per trade (e.g., 0.001 = 0.1%)
        num_trades: Number of trades (default 2 for entry + exit)
    
    Returns:
        Net log return after transaction costs
        
    Example:
        >>> log_ret = compute_log_return(100.0, 105.0)  # ~0.0488
        >>> compute_net_return(log_ret, txn_cost_pct=0.001, num_trades=2)
        0.04660...  # Reduced by ~0.002 (2 * 0.1%)
    """
    # Convert cost to log: ln(1 - cost) ≈ -cost for small cost
    # For exact arithmetic: ln((1-cost)^num_trades)
    cost_multiplier = (1 - txn_cost_pct) ** num_trades
    cost_log = math.log(cost_multiplier)
    
    return log_return + cost_log


def compute_mae(
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
) -> float:
    """
    Compute Max Adverse Excursion - worst drawdown during holding period.
    
    MAE measures the worst unrealized loss experienced during the trade.
    This distinguishes signals that:
    - Reached target smoothly (low MAE)
    - Experienced wild swings (high MAE - would hit stop-loss)
    
    Convention: MAE is negative (adverse means against us).
    
    Args:
        df: OHLCV DataFrame for holding period
        direction: Signal direction ("HIGHER" | "LOWER")
        entry_price: Entry price for the trade
    
    Returns:
        MAE as percentage (negative or zero)
        
    Example:
        For HIGHER signal at entry=100:
        - If lowest low during holding = 95 → MAE = -0.05 (-5%)
        - If lowest low during holding = 102 → MAE = 0.0 (no adverse)
        
        For LOWER signal at entry=100:
        - If highest high during holding = 105 → MAE = -0.05 (-5%)
        - If highest high during holding = 98 → MAE = 0.0 (no adverse)
    """
    if df.empty:
        raise ValueError("Cannot compute MAE from empty DataFrame")
    
    if direction == "HIGHER":
        # For long: worst case is price dropping to lowest low
        worst_price = df["low"].min()
        mae = (worst_price - entry_price) / entry_price
    elif direction == "LOWER":
        # For short: worst case is price rising to highest high
        worst_price = df["high"].max()
        mae = (entry_price - worst_price) / entry_price
    else:
        raise ValueError(f"Invalid direction: '{direction}'. Must be 'HIGHER' or 'LOWER'")
    
    # MAE is conventionally negative (adverse movement)
    # If trade never went against us, MAE = 0.0
    return min(mae, 0.0)


def determine_direction(log_return: float, threshold: float = 0.0001) -> str:
    """
    Determine realized direction from log return.
    
    Uses small threshold to avoid classifying noise as directional move.
    
    Args:
        log_return: Realized log return
        threshold: Minimum absolute return to classify as directional (default 0.01%)
    
    Returns:
        "HIGHER" | "LOWER" | "FLAT"
        
    Example:
        >>> determine_direction(0.05)   # 5% gain
        'HIGHER'
        >>> determine_direction(-0.02)  # 2% loss
        'LOWER'
        >>> determine_direction(0.00005)  # 0.005% - noise
        'FLAT'
    """
    if log_return > threshold:
        return "HIGHER"
    elif log_return < -threshold:
        return "LOWER"
    else:
        return "FLAT"
