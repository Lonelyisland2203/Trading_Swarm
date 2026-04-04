"""
Evaluation Layer - Signal quality assessment and performance metrics.

Computes IC, Sharpe, Sortino, Calmar, win rate, profit factor, and other metrics
from verified outcomes with FDR-corrected statistical significance testing.
"""

from .config import EvaluationConfig, SampleSizeRequirements
from .engine import EvaluationResult, evaluate_batch
from .metrics import MetricValue

__all__ = [
    "EvaluationConfig",
    "SampleSizeRequirements",
    "MetricValue",
    "EvaluationResult",
    "evaluate_batch",
]
