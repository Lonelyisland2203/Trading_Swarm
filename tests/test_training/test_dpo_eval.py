"""Tests for DPO adapter evaluation."""

import pytest
import numpy as np

from swarm.training_capture import TrainingExample
from training.dpo_eval import (
    AdapterEvaluation,
    EvaluationError,
    compare_adapters,
    evaluate_adapter,
    should_promote_adapter,
)
from training.reward_engine import ComputedReward
from verifier.outcome import VerifiedOutcome


@pytest.fixture
def sample_test_set():
    """Create sample test set with examples, outcomes, and rewards."""

    def _make_test_set(num_samples: int = 100, ic_strength: float = 0.3):
        """
        Create test set with controlled IC.

        Args:
            num_samples: Number of test samples
            ic_strength: Strength of IC (0 = random, 1 = perfect correlation)
        """
        np.random.seed(42)  # Reproducible

        examples = []
        outcomes = []
        rewards = []

        # Generate returns first
        true_returns = np.random.normal(0, 0.03, num_samples)  # 3% volatility

        # Generate confidences directly correlated with returns
        # IC = correlation(confidence, return)
        # So we want: confidence ∝ ic_strength * return + noise
        noise = np.random.uniform(0.6, 0.8, num_samples)
        # Scale returns to [0,1] range and mix with noise
        returns_normalized = (true_returns - true_returns.min()) / (
            true_returns.max() - true_returns.min() + 1e-10
        )
        confidences = ic_strength * returns_normalized + (1 - ic_strength) * noise
        # Clip to valid confidence range
        confidences = np.clip(confidences, 0.55, 0.95)

        for i in range(num_samples):
            true_return = true_returns[i]
            confidence = confidences[i]

            # Determine direction - always predict HIGHER for positive returns
            # This simplifies the IC interpretation
            direction = "HIGHER" if true_return > 0 else "LOWER"
            actual_direction = "HIGHER" if true_return > 0 else "LOWER"

            # Create training example
            example = TrainingExample(
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000000000 + i * 3600000,
                market_regime="NEUTRAL",
                persona="MOMENTUM",
                context_id=f"ctx-{i}",
                task_prompt=f"Prompt {i}",
                generator_signal={
                    "direction": direction,
                    "confidence": confidence,
                    "reasoning": f"Reason {i}",
                    "persona": "MOMENTUM",
                },
            )
            examples.append(example)

            # Create verified outcome
            outcome = VerifiedOutcome(
                example_id=f"ex-{i}",
                actual_direction=actual_direction,
                realized_return=true_return,
                max_adverse_excursion=-abs(true_return) * 0.5,
                net_return=true_return - 0.002,
                entry_price=100.0,
                exit_price=100.0 * (1 + true_return),
                bars_held=24,
            )
            outcomes.append(outcome)

            # Create computed reward
            reward = ComputedReward(
                final_reward=true_return * 10,
                return_reward=true_return * 10,
                directional_reward=0.0,
                mae_reward=0.0,
                return_weight=1.0,
                directional_weight=0.0,
                mae_weight=0.0,
                return_scale=10.0,
                mae_scale=10.0,
                net_return=true_return - 0.002,
                realized_return=true_return,
                mae=-abs(true_return) * 0.5,
                predicted_direction=direction,
                actual_direction=actual_direction,
                confidence=confidence,
                components_used=1,
                computation_timestamp="2024-01-01T00:00:00Z",
                market_regime="NEUTRAL",
                reward_version="1.0.0",
            )
            rewards.append(reward)

        return examples, outcomes, rewards

    return _make_test_set


class TestEvaluateAdapter:
    """Test adapter evaluation."""

    def test_basic_evaluation(self, sample_test_set):
        """Test basic adapter evaluation."""
        examples, outcomes, rewards = sample_test_set(num_samples=100, ic_strength=0.3)

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # Should have positive IC (controlled by ic_strength)
        assert eval_result.ic > 0
        assert eval_result.ic_pvalue < 0.05  # Should be significant
        assert eval_result.num_examples == 100
        assert eval_result.brier_score >= 0.0
        assert eval_result.mean_abs_calibration_error >= 0.0

    def test_strong_ic(self, sample_test_set):
        """Test with strong IC correlation."""
        examples, outcomes, rewards = sample_test_set(num_samples=150, ic_strength=0.8)

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # Should have high IC
        assert eval_result.ic > 0.5
        assert eval_result.ic_pvalue < 0.001  # Very significant

    def test_weak_ic(self, sample_test_set):
        """Test with weak IC correlation."""
        examples, outcomes, rewards = sample_test_set(num_samples=100, ic_strength=0.0)

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # IC should be close to zero (random)
        assert abs(eval_result.ic) < 0.3

    def test_regime_stratified_ic(self, sample_test_set):
        """Test regime-stratified IC computation."""
        examples, outcomes, rewards = sample_test_set(num_samples=100)

        # Modify regimes
        for i, ex in enumerate(examples):
            if i < 50:
                ex.market_regime = "TRENDING"
            else:
                ex.market_regime = "MEAN_REVERTING"

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # Should have IC for both regimes
        assert "TRENDING" in eval_result.ic_by_regime
        assert "MEAN_REVERTING" in eval_result.ic_by_regime

    def test_insufficient_samples_raises(self, sample_test_set):
        """Test error when test set too small."""
        examples, outcomes, rewards = sample_test_set(num_samples=20)

        with pytest.raises(EvaluationError, match="Test set too small"):
            evaluate_adapter(examples, outcomes, rewards)

    def test_length_mismatch_raises(self, sample_test_set):
        """Test error when lengths don't match."""
        examples, outcomes, rewards = sample_test_set(num_samples=100)

        with pytest.raises(EvaluationError, match="Length mismatch"):
            evaluate_adapter(examples[:50], outcomes, rewards)

    def test_empty_examples_raises(self):
        """Test error on empty examples."""
        with pytest.raises(EvaluationError, match="No test examples"):
            evaluate_adapter([], [], [])

    def test_with_baseline_comparison(self, sample_test_set):
        """Test evaluation with baseline comparison."""
        # Baseline with IC=0.2
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(
            num_samples=100, ic_strength=0.2
        )
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        # Candidate with IC=0.4 (improved)
        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(
            num_samples=100, ic_strength=0.4
        )
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        # Should have improvement metrics
        assert candidate_eval.ic_improvement is not None
        assert candidate_eval.ic_improvement > 0
        assert candidate_eval.brier_improvement is not None

    def test_return_weighted_ic(self, sample_test_set):
        """Test return-weighted IC computation."""
        examples, outcomes, rewards = sample_test_set(num_samples=100, ic_strength=0.5)

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # Return-weighted IC should be computed
        assert eval_result.return_weighted_ic != 0.0

    def test_brier_score_bounded(self, sample_test_set):
        """Test Brier score is in valid range."""
        examples, outcomes, rewards = sample_test_set(num_samples=100)

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # Brier score should be between 0 and 1
        assert 0.0 <= eval_result.brier_score <= 1.0

    def test_calibration_error_bounded(self, sample_test_set):
        """Test MACE is in valid range."""
        examples, outcomes, rewards = sample_test_set(num_samples=100)

        eval_result = evaluate_adapter(examples, outcomes, rewards)

        # MACE should be between 0 and 1
        assert 0.0 <= eval_result.mean_abs_calibration_error <= 1.0


class TestShouldPromoteAdapter:
    """Test adapter promotion logic."""

    def test_promote_when_criteria_met(self, sample_test_set):
        """Test promotion when all criteria met."""
        # Baseline with IC=0.2
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(
            num_samples=100, ic_strength=0.2
        )
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        # Candidate with IC=0.4 (strong improvement)
        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(
            num_samples=150, ic_strength=0.5
        )
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        should_promote, reason = should_promote_adapter(
            candidate_eval, baseline_eval, min_ic_improvement=0.02, min_brier_improvement=-0.05  # Allow slight Brier degradation
        )

        # Debug output
        if not should_promote:
            print(f"\nPromotion rejected: {reason}")
            print(f"Candidate IC: {candidate_eval.ic}, Baseline IC: {baseline_eval.ic}")
            print(f"IC improvement: {candidate_eval.ic_improvement}")
            print(f"Brier improvement: {candidate_eval.brier_improvement}")

        assert should_promote, f"Promotion failed: {reason}"
        assert "approved" in reason.lower()

    def test_reject_insufficient_ic_improvement(self, sample_test_set):
        """Test rejection when IC improvement too small."""
        # Baseline and candidate nearly identical
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(
            num_samples=100, ic_strength=0.3
        )
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(
            num_samples=100, ic_strength=0.31
        )
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        should_promote, reason = should_promote_adapter(
            candidate_eval, baseline_eval, min_ic_improvement=0.1  # High threshold
        )

        assert not should_promote
        assert "IC improvement too small" in reason

    def test_reject_small_test_set(self, sample_test_set):
        """Test rejection when test set too small."""
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(num_samples=100)
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        # Small test set
        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(num_samples=50)
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        should_promote, reason = should_promote_adapter(
            candidate_eval, baseline_eval, min_test_samples=100
        )

        assert not should_promote
        assert "Test set too small" in reason

    def test_reject_insignificant_ic(self, sample_test_set):
        """Test rejection when IC not statistically significant."""
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(num_samples=100)
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        # Weak IC (not significant)
        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(
            num_samples=100, ic_strength=0.0
        )
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        should_promote, reason = should_promote_adapter(candidate_eval, baseline_eval)

        assert not should_promote
        assert "IC not significant" in reason

    def test_reject_without_baseline(self, sample_test_set):
        """Test rejection when no baseline provided."""
        examples, outcomes, rewards = sample_test_set(num_samples=100)

        # Candidate eval without baseline comparison
        candidate_eval = evaluate_adapter(examples, outcomes, rewards)

        # Create minimal baseline for comparison
        baseline_eval = AdapterEvaluation(
            ic=0.1,
            ic_pvalue=0.01,
            return_weighted_ic=0.08,
            brier_score=0.25,
            mean_abs_calibration_error=0.15,
            ic_by_regime={},
            num_examples=100,
            mean_reward=0.5,
            std_reward=0.3,
        )

        should_promote, reason = should_promote_adapter(candidate_eval, baseline_eval)

        assert not should_promote
        assert "No baseline for comparison" in reason


class TestCompareAdapters:
    """Test adapter comparison."""

    def test_basic_comparison(self, sample_test_set):
        """Test basic adapter comparison."""
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(
            num_samples=100, ic_strength=0.2
        )
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(
            num_samples=100, ic_strength=0.4
        )
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        comparison = compare_adapters(baseline_eval, candidate_eval)

        # Should have delta metrics
        assert "ic_delta" in comparison
        assert "brier_delta" in comparison
        assert comparison["ic_delta"] == candidate_eval.ic - baseline_eval.ic

    def test_percentage_changes(self, sample_test_set):
        """Test percentage change calculations."""
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(num_samples=100)
        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(num_samples=100)
        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        comparison = compare_adapters(baseline_eval, candidate_eval)

        # Should have percentage changes
        assert "ic_pct_change" in comparison
        assert "brier_pct_change" in comparison

    def test_regime_comparison(self, sample_test_set):
        """Test regime-specific IC comparison."""
        baseline_examples, baseline_outcomes, baseline_rewards = sample_test_set(num_samples=100)
        # Set regimes
        for i, ex in enumerate(baseline_examples):
            ex.market_regime = "TRENDING" if i < 50 else "MEAN_REVERTING"

        baseline_eval = evaluate_adapter(baseline_examples, baseline_outcomes, baseline_rewards)

        candidate_examples, candidate_outcomes, candidate_rewards = sample_test_set(num_samples=100)
        for i, ex in enumerate(candidate_examples):
            ex.market_regime = "TRENDING" if i < 50 else "MEAN_REVERTING"

        candidate_eval = evaluate_adapter(
            candidate_examples, candidate_outcomes, candidate_rewards, baseline_eval=baseline_eval
        )

        comparison = compare_adapters(baseline_eval, candidate_eval)

        # Should have regime comparison
        assert "regime_ic_comparison" in comparison
        assert "TRENDING" in comparison["regime_ic_comparison"]
        assert "MEAN_REVERTING" in comparison["regime_ic_comparison"]
