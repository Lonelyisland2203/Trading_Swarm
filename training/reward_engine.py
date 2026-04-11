"""
Reward computation engine for DPO training.

Computes scalar rewards from verified outcomes with graceful fallback for missing components.
"""

from dataclasses import dataclass
from datetime import datetime, UTC

from loguru import logger

from config.settings import settings
from swarm.training_capture import TrainingExample
from verifier.outcome import VerifiedOutcome
from .reward_components import (
    clip_reward,
    compute_directional_reward,
    compute_mae_reward,
    compute_return_reward,
)
from .reward_config import RewardScaling

# Reward computation version (increment on breaking changes)
REWARD_VERSION = "1.0.0"


@dataclass(slots=True, frozen=True)
class ComputedReward:
    """
    Complete reward computation record for DPO training.
    
    Stores final reward, component breakdown, weights/scaling used,
    and raw input values for reproducibility.
    
    Attributes:
        final_reward: Clipped reward in [-1, 1] for DPO training
        
        return_reward: Return component contribution
        directional_reward: Directional accuracy contribution
        mae_reward: MAE penalty contribution
        
        return_weight: Weight used for return component
        directional_weight: Weight used for directional component
        mae_weight: Weight used for MAE component
        
        return_scale: Scale factor used for return
        mae_scale: Scale factor used for MAE
        
        net_return: Net return after costs (used for reward)
        realized_return: Realized return before costs (for diagnostics)
        mae: Max adverse excursion (negative or zero)
        
        predicted_direction: Signal direction
        actual_direction: Realized direction
        confidence: Signal confidence
        
        components_used: Number of components in final reward (1-3)
        computation_timestamp: When reward was computed
        reward_version: Schema version for compatibility
    """
    
    # Final reward
    final_reward: float
    
    # Components
    return_reward: float
    directional_reward: float
    mae_reward: float
    
    # Weights used (for reproducibility)
    return_weight: float
    directional_weight: float
    mae_weight: float
    
    # Scaling used
    return_scale: float
    mae_scale: float
    
    # Raw inputs (for recomputation)
    net_return: float
    realized_return: float
    mae: float | None
    predicted_direction: str
    actual_direction: str
    confidence: float
    
    # Metadata
    components_used: int
    computation_timestamp: str
    market_regime: str  # From training example, for evaluation grouping
    reward_version: str = REWARD_VERSION


def compute_reward(
    verified_outcome: VerifiedOutcome,
    training_example: TrainingExample,
    scaling: RewardScaling = RewardScaling(),
) -> ComputedReward:
    """
    Compute DPO reward from verified outcome with fallback for missing components.
    
    Implements graceful degradation:
    1. Full reward (all 3 components) - default
    2. Return + directional (if MAE unavailable)
    3. Return + MAE (if direction is FLAT)
    4. Return only (if both directional and MAE unavailable)
    
    Weights are renormalized if components are missing to preserve total weight = 1.0.
    Final reward is clipped to [-1, 1].
    
    Args:
        verified_outcome: Verified outcome from backtesting
        training_example: Training example with signal and critique
        scaling: Scaling parameters for component normalization
    
    Returns:
        Complete reward record with all components and metadata
        
    Example:
        >>> outcome = VerifiedOutcome(
        ...     actual_direction="HIGHER",
        ...     realized_return=0.05,
        ...     net_return=0.048,
        ...     max_adverse_excursion=-0.02,
        ...     ...
        ... )
        >>> example = TrainingExample(
        ...     generator_signal={"direction": "HIGHER", "confidence": 0.8, ...},
        ...     ...
        ... )
        >>> reward = compute_reward(outcome, example)
        >>> print(f"Final reward: {reward.final_reward:.3f}")
        Final reward: 0.542
    """
    # Extract signal details
    # Direction/confidence may be at top level or nested under signal_data
    signal = training_example.generator_signal
    _signal_data = signal.get("signal_data", {})
    predicted_direction = signal.get("direction") or _signal_data.get("direction", "UNKNOWN")
    confidence = signal.get("confidence") or _signal_data.get("confidence", 0.5)
    
    # Get reward weights from config
    weights = settings.reward
    
    # Always compute return component
    return_reward = compute_return_reward(
        verified_outcome.net_return,
        scaling.return_scale,
    )
    
    active_components = [
        ("return", return_reward, weights.return_weight)
    ]
    
    # Directional component: skip if FLAT outcome
    if verified_outcome.actual_direction != "FLAT":
        directional_reward = compute_directional_reward(
            predicted_direction,
            verified_outcome.actual_direction,
            confidence,
        )
        active_components.append(
            ("directional", directional_reward, weights.directional_weight)
        )
    else:
        directional_reward = 0.0
        logger.debug(
            "Skipping directional component (FLAT outcome)",
            example_id=training_example.example_id,
        )
    
    # MAE component: skip if missing
    mae = verified_outcome.max_adverse_excursion
    if mae is not None:
        mae_reward = compute_mae_reward(mae, scaling.mae_scale)
        active_components.append(
            ("mae", mae_reward, weights.mae_weight)
        )
    else:
        mae_reward = 0.0
        logger.debug(
            "Skipping MAE component (unavailable)",
            example_id=training_example.example_id,
        )
    
    # Renormalize weights and compute final reward
    total_weight = sum(w for _, _, w in active_components)
    
    if total_weight == 0:
        logger.error(
            "All component weights are zero",
            example_id=training_example.example_id,
        )
        final_reward = 0.0
    else:
        final_reward = sum(
            r * (w / total_weight)
            for _, r, w in active_components
        )
        final_reward = clip_reward(final_reward)
    
    # Log component summary
    logger.debug(
        "Reward computed",
        example_id=training_example.example_id,
        final_reward=final_reward,
        components=len(active_components),
        return_contrib=return_reward * (weights.return_weight / total_weight),
        dir_contrib=directional_reward * (weights.directional_weight / total_weight) if verified_outcome.actual_direction != "FLAT" else 0.0,
        mae_contrib=mae_reward * (weights.mae_weight / total_weight) if mae is not None else 0.0,
    )
    
    return ComputedReward(
        final_reward=final_reward,
        return_reward=return_reward,
        directional_reward=directional_reward,
        mae_reward=mae_reward,
        return_weight=weights.return_weight,
        directional_weight=weights.directional_weight,
        mae_weight=weights.mae_weight,
        return_scale=scaling.return_scale,
        mae_scale=scaling.mae_scale,
        net_return=verified_outcome.net_return,
        realized_return=verified_outcome.realized_return,
        mae=mae,
        predicted_direction=predicted_direction,
        actual_direction=verified_outcome.actual_direction,
        confidence=confidence,
        components_used=len(active_components),
        computation_timestamp=datetime.now(UTC).isoformat(),
        market_regime=training_example.market_regime,
        reward_version=REWARD_VERSION,
    )


@dataclass(slots=True, frozen=True)
class BatchDiagnostics:
    """
    Batch-level reward distribution diagnostics.
    
    Computed for monitoring only - does not affect reward values.
    """
    
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    pct_positive: float
    pct_clipped: float
    total_examples: int


@dataclass(slots=True, frozen=True)
class BatchRewardResult:
    """Result from batch reward computation."""
    
    rewards: list[ComputedReward]
    diagnostics: BatchDiagnostics


def compute_rewards_for_batch(
    examples_and_outcomes: list[tuple[TrainingExample, VerifiedOutcome]],
    scaling: RewardScaling = RewardScaling(),
) -> BatchRewardResult:
    """
    Compute rewards for a batch with diagnostics.
    
    Rewards are computed per-example (deterministic, order-independent).
    Diagnostics are computed for the batch (monitoring only).
    
    Args:
        examples_and_outcomes: List of (TrainingExample, VerifiedOutcome) pairs
        scaling: Scaling parameters for component normalization
    
    Returns:
        Batch result with rewards and diagnostics
        
    Example:
        >>> pairs = [(example1, outcome1), (example2, outcome2), ...]
        >>> result = compute_rewards_for_batch(pairs)
        >>> print(f"Mean reward: {result.diagnostics.mean_reward:.3f}")
        >>> print(f"Positive: {result.diagnostics.pct_positive:.1%}")
    """
    if not examples_and_outcomes:
        logger.warning("Empty batch provided to compute_rewards_for_batch")
        return BatchRewardResult(
            rewards=[],
            diagnostics=BatchDiagnostics(
                mean_reward=0.0,
                std_reward=0.0,
                min_reward=0.0,
                max_reward=0.0,
                pct_positive=0.0,
                pct_clipped=0.0,
                total_examples=0,
            ),
        )
    
    # Compute per-example rewards
    rewards = [
        compute_reward(outcome, example, scaling)
        for example, outcome in examples_and_outcomes
    ]
    
    # Compute batch diagnostics
    reward_values = [r.final_reward for r in rewards]
    
    mean_reward = sum(reward_values) / len(reward_values)
    
    if len(reward_values) > 1:
        variance = sum((r - mean_reward) ** 2 for r in reward_values) / len(reward_values)
        std_reward = variance ** 0.5
    else:
        std_reward = 0.0
    
    min_reward = min(reward_values)
    max_reward = max(reward_values)
    
    pct_positive = sum(1 for r in reward_values if r > 0) / len(reward_values)
    pct_clipped = sum(1 for r in reward_values if abs(r) >= 0.99) / len(reward_values)
    
    diagnostics = BatchDiagnostics(
        mean_reward=mean_reward,
        std_reward=std_reward,
        min_reward=min_reward,
        max_reward=max_reward,
        pct_positive=pct_positive,
        pct_clipped=pct_clipped,
        total_examples=len(rewards),
    )
    
    logger.info(
        "Batch rewards computed",
        total=len(rewards),
        mean=mean_reward,
        std=std_reward,
        pct_positive=f"{pct_positive:.1%}",
        pct_clipped=f"{pct_clipped:.1%}",
    )
    
    return BatchRewardResult(rewards=rewards, diagnostics=diagnostics)
