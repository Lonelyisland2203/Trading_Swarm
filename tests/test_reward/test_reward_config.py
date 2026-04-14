"""Tests for reward configuration."""

import pytest

from training.reward_config import RewardScaling


class TestRewardScaling:
    """Test reward scaling configuration."""

    def test_default_values(self):
        """Test default scaling values."""
        scaling = RewardScaling()

        assert scaling.return_scale == 10.0
        assert scaling.mae_scale == 10.0

    def test_custom_values(self):
        """Test custom scaling values."""
        scaling = RewardScaling(return_scale=20.0, mae_scale=15.0)

        assert scaling.return_scale == 20.0
        assert scaling.mae_scale == 15.0

    def test_immutable(self):
        """Test that RewardScaling is immutable."""
        scaling = RewardScaling()

        with pytest.raises(AttributeError):
            scaling.return_scale = 5.0  # type: ignore

    def test_rejects_zero_return_scale(self):
        """Test that zero return_scale raises ValueError."""
        with pytest.raises(ValueError, match="return_scale must be positive"):
            RewardScaling(return_scale=0.0)

    def test_rejects_negative_return_scale(self):
        """Test that negative return_scale raises ValueError."""
        with pytest.raises(ValueError, match="return_scale must be positive"):
            RewardScaling(return_scale=-1.0)

    def test_rejects_zero_mae_scale(self):
        """Test that zero mae_scale raises ValueError."""
        with pytest.raises(ValueError, match="mae_scale must be positive"):
            RewardScaling(mae_scale=0.0)

    def test_rejects_negative_mae_scale(self):
        """Test that negative mae_scale raises ValueError."""
        with pytest.raises(ValueError, match="mae_scale must be positive"):
            RewardScaling(mae_scale=-5.0)
