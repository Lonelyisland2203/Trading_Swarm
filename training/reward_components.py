"""
Individual reward component computation functions.

Each component produces a value approximately in [-1, 1] range before weighting.
"""


def compute_return_reward(net_return: float, scale: float) -> float:
    """
    Compute return component of reward.

    Scales net return (after transaction costs) to approximately [-1, 1] range.
    Clips to prevent outliers from dominating the reward signal.

    Args:
        net_return: Log return after transaction costs
        scale: Scaling factor (e.g., 10.0 if typical returns are ~10%)

    Returns:
        Reward component in [-1, 1] range

    Example:
        >>> compute_return_reward(0.05, scale=10.0)  # 5% return
        0.5
        >>> compute_return_reward(-0.02, scale=10.0)  # -2% return
        -0.2
        >>> compute_return_reward(0.15, scale=10.0)  # 15% return (outlier)
        1.0  # Clipped
    """
    scaled = net_return * scale
    return max(-1.0, min(1.0, scaled))


def compute_directional_reward(
    predicted_direction: str,
    actual_direction: str,
    confidence: float,
) -> float:
    """
    Compute directional accuracy component with confidence weighting.

    Incentivizes well-calibrated predictions:
    - High confidence + correct = high reward
    - High confidence + wrong = high penalty
    - Low confidence + wrong = low penalty (appropriately uncertain)

    FLAT outcomes provide no directional signal and return 0.

    Args:
        predicted_direction: Predicted direction ("HIGHER" | "LOWER")
        actual_direction: Realized direction ("HIGHER" | "LOWER" | "FLAT")
        confidence: Generator confidence in [0.5, 1.0]

    Returns:
        Reward component in [-1, 1] range

    Example:
        >>> compute_directional_reward("HIGHER", "HIGHER", confidence=0.9)
        0.8  # Correct with high confidence
        >>> compute_directional_reward("HIGHER", "LOWER", confidence=0.9)
        -0.8  # Wrong with high confidence (penalized)
        >>> compute_directional_reward("HIGHER", "LOWER", confidence=0.6)
        -0.2  # Wrong with low confidence (small penalty)
        >>> compute_directional_reward("HIGHER", "FLAT", confidence=0.9)
        0.0  # FLAT outcome - no signal
    """
    # FLAT outcomes provide no directional information
    if actual_direction == "FLAT":
        return 0.0

    # Scale confidence from [0.5, 1.0] to [0, 1.0]
    # This makes random guessing (0.5 confidence) worth 0
    scaled_confidence = (confidence - 0.5) * 2.0
    scaled_confidence = max(0.0, min(1.0, scaled_confidence))  # Safety clip

    # Determine correctness
    correct = predicted_direction == actual_direction
    correctness = 1.0 if correct else -1.0

    return correctness * scaled_confidence


def compute_mae_reward(mae: float, scale: float) -> float:
    """
    Compute MAE (Max Adverse Excursion) penalty component.

    MAE represents the worst drawdown experienced during the holding period.
    Lower MAE (closer to 0) is better - it means the signal reached the target
    without significant adverse movement.

    MAE is always <= 0 (convention: adverse means against us).

    Args:
        mae: Max adverse excursion as fraction (e.g., -0.05 for 5% drawdown)
        scale: Scaling factor (e.g., 10.0 if typical MAE is ~10%)

    Returns:
        Penalty component in [-1, 0] range

    Example:
        >>> compute_mae_reward(0.0, scale=10.0)  # No adverse excursion
        0.0
        >>> compute_mae_reward(-0.05, scale=10.0)  # 5% drawdown
        -0.5
        >>> compute_mae_reward(-0.10, scale=10.0)  # 10% drawdown
        -1.0  # Clipped
        >>> compute_mae_reward(-0.20, scale=10.0)  # 20% drawdown (outlier)
        -1.0  # Clipped
    """
    # MAE is negative or zero, so mae * scale is negative or zero
    penalty = mae * scale

    # Clip to [-1, 0] range (penalty cannot be positive)
    return max(-1.0, min(0.0, penalty))


def clip_reward(reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Clip final reward to bounded range.

    Prevents outliers from dominating DPO training gradients.

    Args:
        reward: Unclipped reward value
        min_val: Minimum allowed value (default -1.0)
        max_val: Maximum allowed value (default +1.0)

    Returns:
        Clipped reward in [min_val, max_val]

    Example:
        >>> clip_reward(0.5)
        0.5
        >>> clip_reward(1.5)
        1.0
        >>> clip_reward(-2.0)
        -1.0
    """
    return max(min_val, min(max_val, reward))
