"""
GRPO training data types and walk-forward split functions.

Provides GRPOTrainingExample for representing training inputs and
temporal split utilities for walk-forward validation.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GRPOTrainingExample:
    """
    Single training example for GRPO.

    Attributes:
        market_snapshot: Input prompt with market data and indicators
        actual_direction: Ground truth direction ("LONG", "SHORT", "FLAT")
        gross_return_pct: Ground truth return for reward computation
        timestamp_ms: Timestamp for temporal ordering (milliseconds)
    """

    market_snapshot: str
    actual_direction: str
    gross_return_pct: float
    timestamp_ms: int
