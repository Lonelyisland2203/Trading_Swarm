"""Tests for evaluation configuration."""

import pytest

from eval.config import EvaluationConfig, SampleSizeRequirements


class TestSampleSizeRequirements:
    """Test sample size requirement thresholds."""

    def test_default_values(self):
        """Test default thresholds."""
        req = SampleSizeRequirements()
        assert req.minimum == 30
        assert req.marginal == 60
        assert req.adequate == 100
        assert req.robust == 250

    def test_custom_values(self):
        """Test custom threshold values."""
        req = SampleSizeRequirements(
            minimum=50,
            marginal=100,
            adequate=200,
            robust=500,
        )
        assert req.minimum == 50
        assert req.marginal == 100

    def test_get_confidence_level_insufficient(self):
        """Test confidence level for insufficient sample size."""
        req = SampleSizeRequirements()
        assert req.get_confidence_level(29) is None
        assert req.get_confidence_level(0) is None

    def test_get_confidence_level_low(self):
        """Test low confidence region."""
        req = SampleSizeRequirements()
        assert req.get_confidence_level(30) == "low"
        assert req.get_confidence_level(59) == "low"

    def test_get_confidence_level_moderate(self):
        """Test moderate confidence region."""
        req = SampleSizeRequirements()
        assert req.get_confidence_level(60) == "moderate"
        assert req.get_confidence_level(99) == "moderate"

    def test_get_confidence_level_high(self):
        """Test high confidence region."""
        req = SampleSizeRequirements()
        assert req.get_confidence_level(100) == "high"
        assert req.get_confidence_level(1000) == "high"

    def test_is_immutable(self):
        """Test that SampleSizeRequirements is frozen."""
        req = SampleSizeRequirements()
        with pytest.raises(AttributeError):
            req.minimum = 50  # type: ignore


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.annualization_factor == 365  # Crypto default
        assert config.min_sample_size == 30
        assert config.fdr_alpha == 0.05
        assert config.bootstrap_samples == 1000
        assert config.confidence_buckets == 5

    def test_custom_crypto_config(self):
        """Test custom configuration for crypto."""
        config = EvaluationConfig(
            annualization_factor=365,
            min_sample_size=50,
            fdr_alpha=0.01,
        )
        assert config.annualization_factor == 365
        assert config.min_sample_size == 50
        assert config.fdr_alpha == 0.01

    def test_equity_config(self):
        """Test configuration for equity markets."""
        config = EvaluationConfig(annualization_factor=252)
        assert config.annualization_factor == 252

    def test_rejects_negative_annualization(self):
        """Test that negative annualization factor is rejected."""
        with pytest.raises(ValueError, match="annualization_factor must be positive"):
            EvaluationConfig(annualization_factor=-1)

    def test_rejects_zero_annualization(self):
        """Test that zero annualization factor is rejected."""
        with pytest.raises(ValueError, match="annualization_factor must be positive"):
            EvaluationConfig(annualization_factor=0)

    def test_rejects_too_small_sample_size(self):
        """Test that sample size < 2 is rejected."""
        with pytest.raises(ValueError, match="min_sample_size must be >= 2"):
            EvaluationConfig(min_sample_size=1)

    def test_rejects_invalid_fdr_alpha(self):
        """Test that FDR alpha outside (0, 1) is rejected."""
        with pytest.raises(ValueError, match="fdr_alpha must be in"):
            EvaluationConfig(fdr_alpha=0.0)

        with pytest.raises(ValueError, match="fdr_alpha must be in"):
            EvaluationConfig(fdr_alpha=1.0)

        with pytest.raises(ValueError, match="fdr_alpha must be in"):
            EvaluationConfig(fdr_alpha=-0.05)

    def test_rejects_too_few_bootstrap_samples(self):
        """Test that bootstrap samples < 100 is rejected."""
        with pytest.raises(ValueError, match="bootstrap_samples must be >= 100"):
            EvaluationConfig(bootstrap_samples=99)

    def test_rejects_too_few_confidence_buckets(self):
        """Test that confidence buckets < 2 is rejected."""
        with pytest.raises(ValueError, match="confidence_buckets must be >= 2"):
            EvaluationConfig(confidence_buckets=1)

    def test_is_immutable(self):
        """Test that EvaluationConfig is frozen."""
        config = EvaluationConfig()
        with pytest.raises(AttributeError):
            config.annualization_factor = 252  # type: ignore
