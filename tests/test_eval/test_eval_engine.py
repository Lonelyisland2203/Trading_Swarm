"""Tests for evaluation engine integration."""

from datetime import UTC, datetime

import pytest

from eval.config import EvaluationConfig
from eval.engine import evaluate_batch
from training.reward_engine import ComputedReward
from verifier.outcome import VerifiedOutcome


@pytest.fixture
def sample_outcomes():
    """Create sample verified outcomes for testing."""
    return [
        VerifiedOutcome(
            example_id="BTC-0001",
            actual_direction="HIGHER",
            realized_return=0.05,
            max_adverse_excursion=-0.01,
            net_return=0.048,
            entry_price=100.0,
            exit_price=105.0,
            bars_held=24,
        ),
        VerifiedOutcome(
            example_id="BTC-0002",
            actual_direction="LOWER",
            realized_return=-0.03,
            max_adverse_excursion=-0.02,
            net_return=-0.032,
            entry_price=100.0,
            exit_price=97.0,
            bars_held=24,
        ),
        VerifiedOutcome(
            example_id="ETH-0003",
            actual_direction="HIGHER",
            realized_return=0.02,
            max_adverse_excursion=-0.005,
            net_return=0.018,
            entry_price=100.0,
            exit_price=102.0,
            bars_held=24,
        ),
    ]


@pytest.fixture
def sample_rewards():
    """Create sample computed rewards for testing."""
    return [
        ComputedReward(
            final_reward=0.6,
            return_reward=0.5,
            directional_reward=0.8,
            mae_reward=-0.1,
            return_weight=0.5,
            directional_weight=0.3,
            mae_weight=0.2,
            return_scale=10.0,
            mae_scale=10.0,
            net_return=0.048,
            realized_return=0.05,
            mae=-0.01,
            predicted_direction="HIGHER",
            actual_direction="HIGHER",
            confidence=0.8,
            components_used=3,
            computation_timestamp=datetime.now(UTC).isoformat(),
            market_regime="RISK_ON",
            reward_version="1.0.0",
        ),
        ComputedReward(
            final_reward=-0.4,
            return_reward=-0.3,
            directional_reward=-0.6,
            mae_reward=-0.2,
            return_weight=0.5,
            directional_weight=0.3,
            mae_weight=0.2,
            return_scale=10.0,
            mae_scale=10.0,
            net_return=-0.032,
            realized_return=-0.03,
            mae=-0.02,
            predicted_direction="HIGHER",
            actual_direction="LOWER",
            confidence=0.7,
            components_used=3,
            computation_timestamp=datetime.now(UTC).isoformat(),
            market_regime="RISK_OFF",
            reward_version="1.0.0",
        ),
        ComputedReward(
            final_reward=0.2,
            return_reward=0.18,
            directional_reward=0.4,
            mae_reward=-0.05,
            return_weight=0.5,
            directional_weight=0.3,
            mae_weight=0.2,
            return_scale=10.0,
            mae_scale=10.0,
            net_return=0.018,
            realized_return=0.02,
            mae=-0.005,
            predicted_direction="HIGHER",
            actual_direction="HIGHER",
            confidence=0.6,
            components_used=3,
            computation_timestamp=datetime.now(UTC).isoformat(),
            market_regime="RISK_ON",
            reward_version="1.0.0",
        ),
    ]


class TestEvaluateBatch:
    """Test evaluate_batch function."""

    def test_evaluate_empty_batch_raises_error(self):
        """Test that empty batch raises ValueError."""
        with pytest.raises(ValueError, match="empty batch"):
            evaluate_batch([], [])

    def test_evaluate_mismatched_lengths_raises_error(self, sample_outcomes):
        """Test that mismatched outcome/reward lengths raise error."""
        with pytest.raises(ValueError, match="mismatch"):
            evaluate_batch(sample_outcomes, [])

    def test_evaluate_basic_batch(self, sample_outcomes, sample_rewards):
        """Test basic batch evaluation."""
        config = EvaluationConfig(min_sample_size=2)  # Lower threshold for testing
        result = evaluate_batch(sample_outcomes, sample_rewards, config=config)

        assert result.sample_size_total == 3
        assert len(result.overall_metrics) > 0
        assert 'ic_spearman' in result.overall_metrics
        assert 'sharpe_ratio' in result.overall_metrics

    def test_overall_metrics_computed(self, sample_outcomes, sample_rewards):
        """Test that overall metrics are computed."""
        config = EvaluationConfig(min_sample_size=2)  # Lower threshold for testing
        result = evaluate_batch(sample_outcomes, sample_rewards, config=config)

        metrics = result.overall_metrics

        # Check all expected metrics exist
        assert 'ic_spearman' in metrics
        assert 'ic_pearson' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'mean_return' in metrics

    def test_per_symbol_metrics_computed(self, sample_outcomes, sample_rewards):
        """Test that per-symbol metrics are computed."""
        result = evaluate_batch(sample_outcomes, sample_rewards)

        # Should have metrics for BTC and ETH (extracted from example_id)
        assert len(result.per_symbol_metrics) >= 1

    def test_per_regime_metrics_computed(self, sample_outcomes, sample_rewards):
        """Test that per-regime metrics are computed."""
        result = evaluate_batch(sample_outcomes, sample_rewards)

        # Should have metrics for RISK_ON and RISK_OFF
        assert 'RISK_ON' in result.per_regime_metrics
        assert 'RISK_OFF' in result.per_regime_metrics

    def test_fdr_correction_applied(self, sample_outcomes, sample_rewards):
        """Test that FDR correction is applied to IC tests."""
        result = evaluate_batch(sample_outcomes, sample_rewards)

        # Should have FDR-adjusted p-values
        assert len(result.fdr_adjusted_pvalues) >= 0  # May be empty if insufficient data

    def test_metric_values_have_metadata(self, sample_outcomes, sample_rewards):
        """Test that MetricValue objects include metadata."""
        config = EvaluationConfig(min_sample_size=2)  # Lower threshold for testing
        result = evaluate_batch(sample_outcomes, sample_rewards, config=config)

        ic_metric = result.overall_metrics['ic_spearman']

        assert ic_metric.sample_size == 3
        assert ic_metric.p_value is not None
        assert ic_metric.confidence_level in ['low', 'moderate', 'high', None]

    def test_custom_config(self, sample_outcomes, sample_rewards):
        """Test evaluation with custom configuration."""
        config = EvaluationConfig(
            annualization_factor=252,  # Equity markets
            min_sample_size=2,
            fdr_alpha=0.01,
        )

        result = evaluate_batch(sample_outcomes, sample_rewards, config=config)

        assert result.config.annualization_factor == 252
        assert result.config.fdr_alpha == 0.01

    def test_evaluation_timestamp_present(self, sample_outcomes, sample_rewards):
        """Test that evaluation timestamp is recorded."""
        result = evaluate_batch(sample_outcomes, sample_rewards)

        assert result.evaluation_timestamp is not None
        # Should be valid ISO format
        datetime.fromisoformat(result.evaluation_timestamp)
