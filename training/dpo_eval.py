"""
DPO training evaluation metrics.

Evaluates fine-tuned adapters on held-out test sets to determine if
the adapter improves performance and should be promoted to production.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger

from swarm.training_capture import TrainingExample
from training.reward_engine import ComputedReward
from verifier.outcome import VerifiedOutcome


@dataclass(frozen=True)
class AdapterEvaluation:
    """Evaluation results for a DPO adapter."""

    # Information coefficient (IC) metrics
    ic: float
    ic_pvalue: float
    return_weighted_ic: float

    # Calibration metrics
    brier_score: float
    mean_abs_calibration_error: float

    # Regime-stratified IC
    ic_by_regime: dict[str, float]

    # Sample statistics
    num_examples: int
    mean_reward: float
    std_reward: float

    # Comparison to baseline (if provided)
    ic_improvement: float | None = None
    brier_improvement: float | None = None


class EvaluationError(Exception):
    """Raised when evaluation fails."""

    pass


def evaluate_adapter(
    examples: List[TrainingExample],
    outcomes: List[VerifiedOutcome],
    rewards: List[ComputedReward],
    baseline_eval: AdapterEvaluation | None = None,
) -> AdapterEvaluation:
    """
    Evaluate adapter performance on test set.

    Computes:
    - Information coefficient (IC): Spearman correlation between predicted confidence and actual returns
    - Return-weighted IC: IC weighted by absolute returns (emphasizes large moves)
    - Brier score: Calibration quality of confidence scores
    - Mean absolute calibration error (MACE)
    - Regime-stratified IC

    Args:
        examples: Test set training examples (with generator signals)
        outcomes: Verified outcomes for test examples
        rewards: Computed rewards for test examples
        baseline_eval: Optional baseline evaluation to compute improvement metrics

    Returns:
        AdapterEvaluation with comprehensive metrics

    Raises:
        EvaluationError: If evaluation fails or test set is too small

    Example:
        >>> eval_result = evaluate_adapter(test_examples, test_outcomes, test_rewards)
        >>> if eval_result.ic > 0.05:
        ...     print("Strong predictive power!")
    """
    if not examples:
        raise EvaluationError("No test examples provided")

    if len(examples) != len(outcomes) or len(examples) != len(rewards):
        raise EvaluationError(
            f"Length mismatch: {len(examples)} examples, "
            f"{len(outcomes)} outcomes, {len(rewards)} rewards"
        )

    if len(examples) < 30:
        raise EvaluationError(f"Test set too small: {len(examples)} examples (need >= 30)")

    # Extract arrays
    predicted_confidences = np.array([ex.generator_signal["confidence"] for ex in examples])
    actual_returns = np.array([outcome.realized_return for outcome in outcomes])
    predicted_directions = np.array([ex.generator_signal["direction"] for ex in examples])
    actual_directions = np.array([outcome.actual_direction for outcome in outcomes])
    final_rewards = np.array([r.final_reward for r in rewards])
    regimes = np.array([ex.market_regime for ex in examples])

    # Compute information coefficient (IC)
    from scipy import stats

    ic, ic_pvalue = stats.spearmanr(predicted_confidences, actual_returns)

    # Compute return-weighted IC (emphasizes large moves)
    abs_returns = np.abs(actual_returns)
    if abs_returns.sum() > 0:
        weights = abs_returns / abs_returns.sum()
        # Weighted Spearman: sort by predicted confidence, weight by return magnitude
        sorted_indices = np.argsort(predicted_confidences)
        weighted_returns = actual_returns[sorted_indices] * weights[sorted_indices]
        return_weighted_ic = weighted_returns.sum() / weights.sum() if weights.sum() > 0 else 0.0
    else:
        return_weighted_ic = 0.0

    # Compute Brier score (calibration)
    # For directional predictions, treat as binary classification
    # Convert direction to binary: 1 if correct, 0 if incorrect
    direction_correct = (predicted_directions == actual_directions).astype(float)
    # Brier score: mean((confidence - actual)^2)
    brier_score = np.mean((predicted_confidences - direction_correct) ** 2)

    # Compute mean absolute calibration error (MACE)
    # Bin predictions by confidence, compare predicted vs actual accuracy
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mace_sum = 0.0
    mace_count = 0

    for i in range(n_bins):
        bin_mask = (predicted_confidences >= bin_edges[i]) & (
            predicted_confidences < bin_edges[i + 1]
        )
        if bin_mask.sum() > 0:
            bin_predicted = predicted_confidences[bin_mask].mean()
            bin_actual = direction_correct[bin_mask].mean()
            mace_sum += np.abs(bin_predicted - bin_actual)
            mace_count += 1

    mace = mace_sum / mace_count if mace_count > 0 else 0.0

    # Compute regime-stratified IC
    ic_by_regime = {}
    unique_regimes = np.unique(regimes)

    for regime in unique_regimes:
        regime_mask = regimes == regime
        if regime_mask.sum() >= 10:  # Need enough samples
            regime_ic, _ = stats.spearmanr(
                predicted_confidences[regime_mask], actual_returns[regime_mask]
            )
            ic_by_regime[regime] = float(regime_ic) if not np.isnan(regime_ic) else 0.0
        else:
            ic_by_regime[regime] = 0.0

    # Sample statistics
    num_examples = len(examples)
    mean_reward = float(final_rewards.mean())
    std_reward = float(final_rewards.std())

    # Compute improvement over baseline
    ic_improvement = None
    brier_improvement = None

    if baseline_eval is not None:
        ic_improvement = ic - baseline_eval.ic
        brier_improvement = baseline_eval.brier_score - brier_score  # Lower is better

    logger.info(
        "Adapter evaluation complete",
        ic=f"{ic:.4f}",
        ic_pvalue=f"{ic_pvalue:.4f}",
        brier_score=f"{brier_score:.4f}",
        num_examples=num_examples,
        ic_improvement=f"{ic_improvement:.4f}" if ic_improvement is not None else None,
    )

    return AdapterEvaluation(
        ic=float(ic),
        ic_pvalue=float(ic_pvalue),
        return_weighted_ic=float(return_weighted_ic),
        brier_score=float(brier_score),
        mean_abs_calibration_error=float(mace),
        ic_by_regime=ic_by_regime,
        num_examples=num_examples,
        mean_reward=mean_reward,
        std_reward=std_reward,
        ic_improvement=ic_improvement,
        brier_improvement=brier_improvement,
    )


def should_promote_adapter(
    candidate_eval: AdapterEvaluation,
    baseline_eval: AdapterEvaluation,
    min_ic_improvement: float = 0.02,
    min_brier_improvement: float = 0.01,
    min_test_samples: int = 100,
) -> tuple[bool, str]:
    """
    Determine if candidate adapter should be promoted to production.

    Promotion criteria:
    1. IC improvement > min_ic_improvement (absolute)
    2. Brier score improvement > min_brier_improvement (lower is better)
    3. Test set size >= min_test_samples
    4. IC is statistically significant (p < 0.05)

    Args:
        candidate_eval: Evaluation of candidate (fine-tuned) adapter
        baseline_eval: Evaluation of baseline (pre-DPO or current production) adapter
        min_ic_improvement: Minimum IC improvement required (default: 0.02)
        min_brier_improvement: Minimum Brier improvement required (default: 0.01)
        min_test_samples: Minimum test set size (default: 100)

    Returns:
        Tuple of (should_promote: bool, reason: str)

    Example:
        >>> should_promote, reason = should_promote_adapter(
        ...     candidate_eval,
        ...     baseline_eval,
        ...     min_ic_improvement=0.02,
        ... )
        >>> if should_promote:
        ...     print(f"Promoting adapter: {reason}")
    """
    if candidate_eval.num_examples < min_test_samples:
        return False, f"Test set too small: {candidate_eval.num_examples} < {min_test_samples}"

    if candidate_eval.ic_pvalue >= 0.05:
        return False, f"IC not significant: p={candidate_eval.ic_pvalue:.4f}"

    # Check IC improvement
    if candidate_eval.ic_improvement is None:
        return False, "No baseline for comparison"

    ic_improvement = candidate_eval.ic_improvement
    if ic_improvement < min_ic_improvement:
        return (
            False,
            f"IC improvement too small: {ic_improvement:.4f} < {min_ic_improvement:.4f}",
        )

    # Check Brier improvement
    if candidate_eval.brier_improvement is None:
        return False, "No baseline Brier score for comparison"

    brier_improvement = candidate_eval.brier_improvement
    if brier_improvement < min_brier_improvement:
        return (
            False,
            f"Brier improvement too small: {brier_improvement:.4f} < {min_brier_improvement:.4f}",
        )

    # All criteria met
    reason = (
        f"Promotion approved: IC+{ic_improvement:.4f}, "
        f"Brier-{brier_improvement:.4f}, "
        f"N={candidate_eval.num_examples}"
    )
    logger.info(reason)
    return True, reason


def compare_adapters(
    baseline_eval: AdapterEvaluation,
    candidate_eval: AdapterEvaluation,
) -> dict:
    """
    Generate detailed comparison between baseline and candidate adapters.

    Args:
        baseline_eval: Baseline adapter evaluation
        candidate_eval: Candidate adapter evaluation

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> comparison = compare_adapters(baseline_eval, candidate_eval)
        >>> print(f"IC delta: {comparison['ic_delta']:.4f}")
    """
    comparison = {
        "ic_delta": candidate_eval.ic - baseline_eval.ic,
        "ic_pct_change": (
            (candidate_eval.ic - baseline_eval.ic) / baseline_eval.ic * 100
            if baseline_eval.ic != 0
            else 0.0
        ),
        "brier_delta": candidate_eval.brier_score - baseline_eval.brier_score,
        "brier_pct_change": (
            (candidate_eval.brier_score - baseline_eval.brier_score)
            / baseline_eval.brier_score
            * 100
            if baseline_eval.brier_score != 0
            else 0.0
        ),
        "return_weighted_ic_delta": (
            candidate_eval.return_weighted_ic - baseline_eval.return_weighted_ic
        ),
        "mace_delta": (
            candidate_eval.mean_abs_calibration_error - baseline_eval.mean_abs_calibration_error
        ),
        "baseline": {
            "ic": baseline_eval.ic,
            "brier": baseline_eval.brier_score,
            "n": baseline_eval.num_examples,
        },
        "candidate": {
            "ic": candidate_eval.ic,
            "brier": candidate_eval.brier_score,
            "n": candidate_eval.num_examples,
        },
        "regime_ic_comparison": {},
    }

    # Regime IC comparison
    all_regimes = set(baseline_eval.ic_by_regime.keys()) | set(
        candidate_eval.ic_by_regime.keys()
    )
    for regime in all_regimes:
        baseline_ic = baseline_eval.ic_by_regime.get(regime, 0.0)
        candidate_ic = candidate_eval.ic_by_regime.get(regime, 0.0)
        comparison["regime_ic_comparison"][regime] = {
            "baseline": baseline_ic,
            "candidate": candidate_ic,
            "delta": candidate_ic - baseline_ic,
        }

    logger.info(
        "Adapter comparison",
        ic_delta=f"{comparison['ic_delta']:.4f}",
        brier_delta=f"{comparison['brier_delta']:.4f}",
    )

    return comparison
