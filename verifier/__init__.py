"""
Verifier Layer - Backtesting and ground truth computation.

Computes realized outcomes from training examples with guaranteed point-in-time safety.
"""

from .config import BacktestConfig
from .engine import verify_batch, verify_example
from .outcome import VerifiedOutcome, compute_log_return, compute_mae

__all__ = [
    "BacktestConfig",
    "VerifiedOutcome",
    "compute_log_return",
    "compute_mae",
    "verify_batch",
    "verify_example",
]
