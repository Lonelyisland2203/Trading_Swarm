"""
GRPO reward computation for training loop.

Computes rewards for generated completions using:
1. Decision reward (asymmetric based on direction correctness)
2. Structure reward (regex-based reasoning format validation)
3. Directional accuracy (binary correct/incorrect)

All components are clipped to [-1, 1] before combination.
"""

import re
from dataclasses import dataclass
from typing import Literal

from config.fee_model import FeeModelSettings
from training.grpo_config import GRPORewardConfig


# Regex patterns for structure validation
# Matches section headers like "## THESIS", "**THESIS**", "THESIS:", etc.
SECTION_PATTERNS = {
    "thesis": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?\s*THESIS\s*(?:\*{1,2})?[:\s]", re.IGNORECASE
    ),
    "evidence": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?\s*EVIDENCE\s*(?:\*{1,2})?[:\s]", re.IGNORECASE
    ),
    "risk": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?\s*RISK[S]?\s*(?:\*{1,2})?[:\s]", re.IGNORECASE
    ),
    "decision": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?\s*DECISION\s*(?:\*{1,2})?[:\s]", re.IGNORECASE
    ),
}


Direction = Literal["HIGHER", "LOWER", "LONG", "SHORT", "FLAT", "NEUTRAL"]


@dataclass(frozen=True, slots=True)
class GRPORewardResult:
    """
    Complete reward computation result for a single completion.

    Stores final reward and all component values for logging/debugging.
    """

    # Final combined reward (weighted sum of components, clipped)
    final_reward: float

    # Individual components (each already clipped to [-1, 1])
    decision_reward: float
    structure_reward: float
    directional_reward: float

    # Raw inputs
    predicted_direction: str
    actual_direction: str
    gross_return_pct: float
    net_return_pct: float

    # Metadata
    sections_found: tuple[str, ...]
    all_sections_present: bool


def clip_component(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clip a reward component to bounded range."""
    return max(min_val, min(max_val, value))


def normalize_direction(direction: str) -> str:
    """
    Normalize direction strings to canonical form.

    HIGHER/LONG -> LONG
    LOWER/SHORT -> SHORT
    FLAT/NEUTRAL -> FLAT
    """
    direction_upper = direction.upper().strip()
    if direction_upper in ("HIGHER", "LONG"):
        return "LONG"
    elif direction_upper in ("LOWER", "SHORT"):
        return "SHORT"
    elif direction_upper in ("FLAT", "NEUTRAL"):
        return "FLAT"
    else:
        # Unknown direction treated as FLAT (no directional bet)
        return "FLAT"


def compute_decision_reward(
    predicted_direction: str,
    actual_direction: str,
    net_return_pct: float,
    config: GRPORewardConfig,
) -> float:
    """
    Compute asymmetric decision reward based on prediction correctness.

    Reward matrix:
    - True bullish (predicted LONG, price up): +1.0 × net_return
    - True bearish (predicted SHORT, price down): +1.0 × net_return
    - False bullish (predicted LONG, price down): -1.5 × |net_return|
    - False bearish (predicted SHORT, price up): -0.8 × |net_return|

    Args:
        predicted_direction: Normalized direction ("LONG", "SHORT", "FLAT")
        actual_direction: Normalized direction ("LONG", "SHORT", "FLAT")
        net_return_pct: Net return after fees as percentage
        config: Reward configuration with asymmetry coefficients

    Returns:
        Decision reward clipped to [-1, 1]
    """
    pred = normalize_direction(predicted_direction)
    actual = normalize_direction(actual_direction)

    # FLAT predictions get zero reward (no position taken)
    if pred == "FLAT":
        return 0.0

    # Determine if prediction was correct
    # LONG is correct if actual was LONG (price went up)
    # SHORT is correct if actual was SHORT (price went down)
    abs_return = abs(net_return_pct)

    if pred == "LONG":
        if actual == "LONG":
            # True bullish: positive reward proportional to return
            reward = config.true_bullish_multiplier * net_return_pct
        elif actual == "SHORT":
            # False bullish: penalized more heavily
            reward = -config.false_bullish_penalty * abs_return
        else:
            # Actual was FLAT - small penalty for unnecessary position
            reward = -0.1 * abs_return
    elif pred == "SHORT":
        if actual == "SHORT":
            # True bearish: positive reward proportional to return magnitude
            # For shorts, we want positive reward when price went down
            # net_return_pct is already computed correctly for short positions
            reward = config.true_bearish_multiplier * abs_return
        elif actual == "LONG":
            # False bearish: penalized (but less than false bullish)
            reward = -config.false_bearish_penalty * abs_return
        else:
            # Actual was FLAT - small penalty for unnecessary position
            reward = -0.1 * abs_return
    else:
        # Should not reach here after normalization
        reward = 0.0

    return clip_component(reward, config.clip_min, config.clip_max)


def check_structure(completion: str) -> tuple[tuple[str, ...], bool]:
    """
    Check if completion contains all required reasoning sections in order.

    Required sections: THESIS, EVIDENCE, RISK, DECISION (in that order)

    Args:
        completion: Generated text to validate

    Returns:
        Tuple of (sections_found, all_present_in_order)
    """
    sections_found = []
    last_position = -1

    # Check each section in order
    for section_name in ["thesis", "evidence", "risk", "decision"]:
        pattern = SECTION_PATTERNS[section_name]
        match = pattern.search(completion)
        if match:
            # Check if this section comes after the previous one
            if match.start() > last_position:
                sections_found.append(section_name.upper())
                last_position = match.start()
            else:
                # Section found but out of order - don't count subsequent matches
                break

    all_present = len(sections_found) == 4 and sections_found == [
        "THESIS",
        "EVIDENCE",
        "RISK",
        "DECISION",
    ]
    return tuple(sections_found), all_present


def compute_structure_reward(
    completion: str,
    config: GRPORewardConfig,
) -> tuple[float, tuple[str, ...], bool]:
    """
    Compute structure reward for reasoning format compliance.

    Args:
        completion: Generated text to validate
        config: Reward configuration

    Returns:
        Tuple of (reward, sections_found, all_sections_present)
    """
    sections_found, all_present = check_structure(completion)

    if all_present:
        reward = config.structure_reward_value
    else:
        reward = 0.0

    return clip_component(reward, config.clip_min, config.clip_max), sections_found, all_present


def compute_directional_accuracy(
    predicted_direction: str,
    actual_direction: str,
    config: GRPORewardConfig,
) -> float:
    """
    Compute binary directional accuracy reward.

    Args:
        predicted_direction: Predicted direction
        actual_direction: Actual direction
        config: Reward configuration

    Returns:
        1.0 if correct, 0.0 if wrong, clipped to [-1, 1]
    """
    pred = normalize_direction(predicted_direction)
    actual = normalize_direction(actual_direction)

    # FLAT predictions or FLAT actuals get 0 (no directional signal)
    if pred == "FLAT" or actual == "FLAT":
        return 0.0

    # Binary accuracy
    if pred == actual:
        return 1.0
    else:
        return 0.0


def compute_net_return(
    gross_return_pct: float,
    holding_periods_8h: float = 1.0,
    fee_model: FeeModelSettings | None = None,
) -> float:
    """
    Compute net return after transaction costs.

    Args:
        gross_return_pct: Gross return as percentage
        holding_periods_8h: Number of 8-hour funding periods
        fee_model: Fee model settings (uses default if None)

    Returns:
        Net return as percentage after fees
    """
    if fee_model is None:
        fee_model = FeeModelSettings()

    return fee_model.net_return(gross_return_pct, holding_periods_8h)


def compute_grpo_reward(
    completion: str,
    predicted_direction: str,
    actual_direction: str,
    gross_return_pct: float,
    holding_periods_8h: float = 1.0,
    config: GRPORewardConfig | None = None,
    fee_model: FeeModelSettings | None = None,
) -> GRPORewardResult:
    """
    Compute complete GRPO reward for a single completion.

    Combined reward formula:
        reward = 0.6 × decision_reward + 0.2 × structure_reward + 0.2 × directional_accuracy

    All components are clipped to [-1, 1] before combination.
    Final reward is also clipped to [-1, 1].

    Args:
        completion: Generated text (for structure validation)
        predicted_direction: Predicted direction ("LONG", "SHORT", "HIGHER", "LOWER", etc.)
        actual_direction: Actual direction from market outcome
        gross_return_pct: Gross return as percentage
        holding_periods_8h: Number of 8-hour funding periods for fee calculation
        config: Reward configuration (uses default if None)
        fee_model: Fee model for net return calculation (uses default if None)

    Returns:
        GRPORewardResult with final reward and all components

    Example:
        >>> result = compute_grpo_reward(
        ...     completion="## THESIS\\nBullish...\\n## EVIDENCE\\n...\\n## RISK\\n...\\n## DECISION\\nLONG",
        ...     predicted_direction="LONG",
        ...     actual_direction="LONG",
        ...     gross_return_pct=0.5,
        ... )
        >>> result.final_reward > 0
        True
        >>> result.all_sections_present
        True
    """
    if config is None:
        config = GRPORewardConfig()

    if fee_model is None:
        fee_model = FeeModelSettings()

    # Compute net return
    net_return_pct = fee_model.net_return(gross_return_pct, holding_periods_8h)

    # Compute each component
    decision_reward = compute_decision_reward(
        predicted_direction=predicted_direction,
        actual_direction=actual_direction,
        net_return_pct=net_return_pct,
        config=config,
    )

    structure_reward, sections_found, all_sections_present = compute_structure_reward(
        completion=completion,
        config=config,
    )

    directional_reward = compute_directional_accuracy(
        predicted_direction=predicted_direction,
        actual_direction=actual_direction,
        config=config,
    )

    # Combine with weights
    combined = (
        config.decision_weight * decision_reward
        + config.structure_weight * structure_reward
        + config.directional_weight * directional_reward
    )

    # Final clipping
    final_reward = clip_component(combined, config.clip_min, config.clip_max)

    return GRPORewardResult(
        final_reward=final_reward,
        decision_reward=decision_reward,
        structure_reward=structure_reward,
        directional_reward=directional_reward,
        predicted_direction=normalize_direction(predicted_direction),
        actual_direction=normalize_direction(actual_direction),
        gross_return_pct=gross_return_pct,
        net_return_pct=net_return_pct,
        sections_found=sections_found,
        all_sections_present=all_sections_present,
    )


def compute_group_advantages(
    rewards: list[float],
    epsilon: float = 1e-8,
) -> list[float]:
    """
    Compute group-relative advantages for GRPO.

    advantage_i = (reward_i - mean(rewards)) / std(rewards)

    This is the core GRPO mechanism: no value network, just group statistics.

    Args:
        rewards: List of rewards for G completions
        epsilon: Small value for numerical stability

    Returns:
        List of advantages (same length as rewards)

    Example:
        >>> rewards = [0.5, -0.2, 0.3, 0.1]
        >>> advantages = compute_group_advantages(rewards)
        >>> abs(sum(advantages)) < 0.01  # Should sum to ~0
        True
    """
    if len(rewards) == 0:
        return []

    if len(rewards) == 1:
        # Single completion - advantage is 0 (no relative comparison)
        return [0.0]

    # Compute mean and std
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = (variance**0.5) + epsilon

    # Compute advantages
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    return advantages
