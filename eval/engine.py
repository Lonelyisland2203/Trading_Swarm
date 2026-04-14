"""
Evaluation engine for signal quality assessment.

Main API: evaluate_batch() processes verified outcomes and rewards to compute
comprehensive performance metrics with statistical significance testing.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Sequence

import numpy as np
from loguru import logger
from scipy.stats import false_discovery_control

from training.reward_engine import ComputedReward
from verifier.outcome import VerifiedOutcome

from .config import EvaluationConfig, SampleSizeRequirements
from .metrics import (
    MetricValue,
    bootstrap_confidence_interval,
    compute_calmar_ratio,
    compute_information_coefficient,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_win_rate,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Complete evaluation results with statistical metadata.

    Contains overall, per-symbol, and per-regime metrics with multiple testing
    correction and confidence intervals.
    """

    # Core metrics
    overall_metrics: Mapping[str, MetricValue]
    per_symbol_metrics: Mapping[str, Mapping[str, MetricValue]]
    per_regime_metrics: Mapping[str, Mapping[str, MetricValue]]

    # Statistical testing
    fdr_adjusted_pvalues: Mapping[str, float]  # IC p-values after FDR correction
    significant_at_alpha: frozenset[str]  # Groups with significant IC

    # Metadata
    evaluation_timestamp: str
    sample_size_total: int
    config: EvaluationConfig


def _compute_metrics_for_group(
    outcomes: Sequence[VerifiedOutcome],
    rewards: Sequence[ComputedReward],
    config: EvaluationConfig,
    requirements: SampleSizeRequirements,
) -> dict[str, MetricValue]:
    """
    Compute all metrics for a single group (overall, per-symbol, or per-regime).

    Args:
        outcomes: Verified outcomes for this group
        rewards: Corresponding rewards
        config: Evaluation configuration
        requirements: Sample size requirements

    Returns:
        Dictionary of metric name to MetricValue
    """
    n = len(outcomes)
    confidence_level = requirements.get_confidence_level(n)

    if confidence_level is None:
        # Insufficient data
        return {}

    if confidence_level == "low":
        logger.warning(
            "Marginal sample size for metrics",
            sample_size=n,
            recommended=requirements.adequate,
        )

    # Extract arrays for computation
    returns = np.array([o.realized_return for o in outcomes])
    net_returns = np.array([o.net_return for o in outcomes])
    confidences = np.array([r.confidence for r in rewards])
    predicted_directions = np.array([r.predicted_direction for r in rewards])
    actual_directions = np.array([o.actual_direction for o in outcomes])

    metrics = {}

    # --- Information Coefficient (both methods) ---
    try:
        spearman_ic, spearman_p = compute_information_coefficient(
            confidences, net_returns, method="spearman"
        )
        metrics["ic_spearman"] = MetricValue(
            value=spearman_ic,
            sample_size=n,
            p_value=spearman_p,
            confidence_level=confidence_level,
        )

        pearson_ic, pearson_p = compute_information_coefficient(
            confidences, net_returns, method="pearson"
        )
        metrics["ic_pearson"] = MetricValue(
            value=pearson_ic,
            sample_size=n,
            p_value=pearson_p,
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute IC", error=str(e), sample_size=n)

    # --- Performance Ratios ---
    try:
        sharpe = compute_sharpe_ratio(net_returns, config.annualization_factor)
        # Bootstrap CI for Sharpe
        sharpe_ci = bootstrap_confidence_interval(
            net_returns,
            lambda x: compute_sharpe_ratio(x, config.annualization_factor),
            n_bootstrap=config.bootstrap_samples,
        )
        metrics["sharpe_ratio"] = MetricValue(
            value=sharpe,
            sample_size=n,
            ci_lower=sharpe_ci[0],
            ci_upper=sharpe_ci[1],
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute Sharpe ratio", error=str(e))

    try:
        sortino = compute_sortino_ratio(net_returns, config.annualization_factor)
        metrics["sortino_ratio"] = MetricValue(
            value=sortino,
            sample_size=n,
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute Sortino ratio", error=str(e))

    try:
        calmar = compute_calmar_ratio(net_returns, config.annualization_factor)
        metrics["calmar_ratio"] = MetricValue(
            value=calmar,
            sample_size=n,
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute Calmar ratio", error=str(e))

    # --- Max Drawdown ---
    try:
        cumulative = np.cumsum(net_returns)
        max_dd = compute_max_drawdown(cumulative)
        metrics["max_drawdown"] = MetricValue(
            value=max_dd,
            sample_size=n,
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute max drawdown", error=str(e))

    # --- Win Rate ---
    try:
        profitable_mask = net_returns > 0
        win_rate = compute_win_rate(profitable_mask)
        metrics["win_rate"] = MetricValue(
            value=win_rate,
            sample_size=n,
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute win rate", error=str(e))

    # --- Profit Factor ---
    try:
        profit_factor = compute_profit_factor(net_returns)
        metrics["profit_factor"] = MetricValue(
            value=profit_factor,
            sample_size=n,
            confidence_level=confidence_level,
        )
    except ValueError as e:
        logger.warning("Failed to compute profit factor", error=str(e))

    # --- Directional Accuracy ---
    try:
        # Only count non-FLAT outcomes for accuracy
        non_flat_mask = actual_directions != "FLAT"
        if np.sum(non_flat_mask) > 0:
            correct_mask = predicted_directions[non_flat_mask] == actual_directions[non_flat_mask]
            directional_accuracy = np.mean(correct_mask)
            metrics["directional_accuracy"] = MetricValue(
                value=float(directional_accuracy),
                sample_size=int(np.sum(non_flat_mask)),
                confidence_level=confidence_level,
            )
    except Exception as e:
        logger.warning("Failed to compute directional accuracy", error=str(e))

    # --- Mean Return ---
    mean_return = float(np.mean(net_returns))
    annualized_return = mean_return * config.annualization_factor
    metrics["mean_return"] = MetricValue(
        value=mean_return,
        sample_size=n,
        confidence_level=confidence_level,
    )
    metrics["annualized_return"] = MetricValue(
        value=annualized_return,
        sample_size=n,
        confidence_level=confidence_level,
    )

    return metrics


def evaluate_batch(
    verified_outcomes: Sequence[VerifiedOutcome],
    rewards: Sequence[ComputedReward],
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """
    Evaluate signal quality from verified outcomes and rewards.

    Computes comprehensive metrics overall, per-symbol, and per-regime with
    FDR-corrected statistical significance testing for IC.

    Args:
        verified_outcomes: Sequence of verified outcomes
        rewards: Corresponding rewards (must match length and order)
        config: Evaluation configuration (uses defaults if None)

    Returns:
        EvaluationResult with all computed metrics and statistical tests

    Raises:
        ValueError: If inputs are mismatched or empty
    """
    if len(verified_outcomes) != len(rewards):
        raise ValueError(
            f"Outcome and reward count mismatch: {len(verified_outcomes)} vs {len(rewards)}"
        )

    if len(verified_outcomes) == 0:
        raise ValueError("Cannot evaluate empty batch")

    if config is None:
        config = EvaluationConfig()

    requirements = SampleSizeRequirements(minimum=config.min_sample_size)

    # --- Overall Metrics ---
    overall_metrics = _compute_metrics_for_group(verified_outcomes, rewards, config, requirements)

    # --- Group by Symbol ---
    symbol_groups: dict[str, list[tuple[VerifiedOutcome, ComputedReward]]] = defaultdict(list)
    for outcome, reward in zip(verified_outcomes, rewards):
        symbol_groups[outcome.example_id.split("-")[0]].append(
            (outcome, reward)
        )  # Extract symbol from ID

    per_symbol_metrics = {}
    for symbol, pairs in symbol_groups.items():
        symbol_outcomes = [p[0] for p in pairs]
        symbol_rewards = [p[1] for p in pairs]
        per_symbol_metrics[symbol] = _compute_metrics_for_group(
            symbol_outcomes, symbol_rewards, config, requirements
        )

    # --- Group by Regime ---
    # Extract regime from training examples (stored in rewards metadata)
    regime_groups: dict[str, list[tuple[VerifiedOutcome, ComputedReward]]] = defaultdict(list)
    for outcome, reward in zip(verified_outcomes, rewards):
        # Assume regime stored in reward metadata (from training example)
        regime = getattr(reward, "market_regime", "UNKNOWN")
        regime_groups[regime].append((outcome, reward))

    per_regime_metrics = {}
    for regime, pairs in regime_groups.items():
        regime_outcomes = [p[0] for p in pairs]
        regime_rewards = [p[1] for p in pairs]
        per_regime_metrics[regime] = _compute_metrics_for_group(
            regime_outcomes, regime_rewards, config, requirements
        )

    # --- Multiple Testing Correction (FDR) for IC ---
    # Collect all IC p-values from per-symbol and per-regime groups
    ic_tests = {}

    for symbol, metrics in per_symbol_metrics.items():
        if "ic_spearman" in metrics and metrics["ic_spearman"].p_value is not None:
            ic_tests[f"symbol_{symbol}"] = metrics["ic_spearman"].p_value

    for regime, metrics in per_regime_metrics.items():
        if "ic_spearman" in metrics and metrics["ic_spearman"].p_value is not None:
            ic_tests[f"regime_{regime}"] = metrics["ic_spearman"].p_value

    # Apply BH-FDR correction
    if len(ic_tests) > 0:
        p_values_list = list(ic_tests.values())
        try:
            # scipy 1.11+ has false_discovery_control
            rejected = false_discovery_control(p_values_list, method="bh")
            fdr_adjusted = {key: p_val for key, p_val in zip(ic_tests.keys(), p_values_list)}
            significant_groups = frozenset(
                key for key, is_rejected in zip(ic_tests.keys(), rejected) if is_rejected
            )
        except Exception:
            # Fallback: use statsmodels if scipy version too old
            from statsmodels.stats.multitest import multipletests

            rejected, p_adj, _, _ = multipletests(
                p_values_list, alpha=config.fdr_alpha, method="fdr_bh"
            )
            fdr_adjusted = {key: p_val for key, p_val in zip(ic_tests.keys(), p_adj)}
            significant_groups = frozenset(
                key for key, is_rejected in zip(ic_tests.keys(), rejected) if is_rejected
            )
    else:
        fdr_adjusted = {}
        significant_groups = frozenset()

    logger.info(
        "Evaluation complete",
        total_samples=len(verified_outcomes),
        symbols=len(per_symbol_metrics),
        regimes=len(per_regime_metrics),
        significant_groups=len(significant_groups),
    )

    return EvaluationResult(
        overall_metrics=overall_metrics,
        per_symbol_metrics=per_symbol_metrics,
        per_regime_metrics=per_regime_metrics,
        fdr_adjusted_pvalues=fdr_adjusted,
        significant_at_alpha=significant_groups,
        evaluation_timestamp=datetime.now(UTC).isoformat(),
        sample_size_total=len(verified_outcomes),
        config=config,
    )
