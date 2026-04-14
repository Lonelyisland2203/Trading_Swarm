"""
DPO fine-tuning and dataset management (Process B).

This module contains:
- Reward computation for training signal generation
- Walk-forward temporal splitting
- DPO training pipeline
- Adapter evaluation and promotion

IMPORTANT: Training-specific imports (dpo_trainer) require the training
dependencies to be installed. These are imported lazily to avoid import
errors in the inference process (Process A).
"""

from .reward_config import RewardScaling
from .reward_engine import (
    BatchDiagnostics,
    BatchRewardResult,
    ComputedReward,
    compute_reward,
    compute_rewards_for_batch,
)

# These are always available (no heavy dependencies)
__all__ = [
    # Reward computation
    "RewardScaling",
    "ComputedReward",
    "BatchDiagnostics",
    "BatchRewardResult",
    "compute_reward",
    "compute_rewards_for_batch",
]


def __getattr__(name: str):
    """
    Lazy import for training-specific components.

    This allows the module to be imported in Process A (inference) without
    requiring training dependencies. Training components are only loaded
    when explicitly accessed in Process B.
    """
    training_exports = {
        "DPOTrainingError",
        "InsufficientDataError",
        "VRAMError",
        "TrainingConfig",
        "TrainingResult",
        "train_dpo",
        "promote_adapter",
        "check_should_retrain",
        "run_preflight_checks",
    }

    if name in training_exports:
        try:
            from . import dpo_trainer

            return getattr(dpo_trainer, name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' - training dependencies not installed. "
                f"Install with: pip install -r requirements-training.txt\n"
                f"Original error: {e}"
            ) from e

    raise AttributeError(f"module 'training' has no attribute '{name}'")
