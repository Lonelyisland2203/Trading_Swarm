"""Tests for individual reward component functions."""

import pytest

from training.reward_components import (
    clip_reward,
    compute_directional_reward,
    compute_mae_reward,
    compute_return_reward,
)


class TestComputeReturnReward:
    """Test return reward computation."""
    
    def test_positive_return_scaled(self):
        """Test that positive return is scaled correctly."""
        # 5% return with scale=10 → 0.5 reward
        reward = compute_return_reward(0.05, scale=10.0)
        assert reward == pytest.approx(0.5, abs=0.001)
    
    def test_negative_return_scaled(self):
        """Test that negative return is scaled correctly."""
        # -2% return with scale=10 → -0.2 reward
        reward = compute_return_reward(-0.02, scale=10.0)
        assert reward == pytest.approx(-0.2, abs=0.001)
    
    def test_zero_return(self):
        """Test that zero return gives zero reward."""
        reward = compute_return_reward(0.0, scale=10.0)
        assert reward == 0.0
    
    def test_positive_outlier_clipped(self):
        """Test that large positive return is clipped to 1.0."""
        # 15% return with scale=10 → 1.5 before clip → 1.0 after
        reward = compute_return_reward(0.15, scale=10.0)
        assert reward == 1.0
    
    def test_negative_outlier_clipped(self):
        """Test that large negative return is clipped to -1.0."""
        # -20% return with scale=10 → -2.0 before clip → -1.0 after
        reward = compute_return_reward(-0.20, scale=10.0)
        assert reward == -1.0
    
    def test_custom_scale_factor(self):
        """Test with custom scale factor."""
        # 10% return with scale=5 → 0.5 reward
        reward = compute_return_reward(0.10, scale=5.0)
        assert reward == pytest.approx(0.5, abs=0.001)
    
    def test_boundary_at_one(self):
        """Test exact boundary at scale threshold."""
        # 10% return with scale=10 → exactly 1.0
        reward = compute_return_reward(0.10, scale=10.0)
        assert reward == 1.0


class TestComputeDirectionalReward:
    """Test directional accuracy reward computation."""
    
    def test_correct_high_confidence(self):
        """Test correct prediction with high confidence."""
        # Correct + 90% confidence → 0.8 reward
        # scaled_confidence = (0.9 - 0.5) * 2 = 0.8
        reward = compute_directional_reward("HIGHER", "HIGHER", confidence=0.9)
        assert reward == pytest.approx(0.8, abs=0.001)
    
    def test_correct_low_confidence(self):
        """Test correct prediction with low confidence."""
        # Correct + 60% confidence → 0.2 reward
        # scaled_confidence = (0.6 - 0.5) * 2 = 0.2
        reward = compute_directional_reward("HIGHER", "HIGHER", confidence=0.6)
        assert reward == pytest.approx(0.2, abs=0.001)
    
    def test_wrong_high_confidence(self):
        """Test wrong prediction with high confidence (high penalty)."""
        # Wrong + 90% confidence → -0.8 penalty
        reward = compute_directional_reward("HIGHER", "LOWER", confidence=0.9)
        assert reward == pytest.approx(-0.8, abs=0.001)
    
    def test_wrong_low_confidence(self):
        """Test wrong prediction with low confidence (low penalty)."""
        # Wrong + 60% confidence → -0.2 penalty
        reward = compute_directional_reward("HIGHER", "LOWER", confidence=0.6)
        assert reward == pytest.approx(-0.2, abs=0.001)
    
    def test_flat_outcome_returns_zero(self):
        """Test that FLAT outcome returns zero regardless of confidence."""
        reward = compute_directional_reward("HIGHER", "FLAT", confidence=0.9)
        assert reward == 0.0
        
        reward = compute_directional_reward("LOWER", "FLAT", confidence=0.5)
        assert reward == 0.0
    
    def test_random_guess_confidence_returns_zero(self):
        """Test that 50% confidence (random guess) returns zero."""
        # scaled_confidence = (0.5 - 0.5) * 2 = 0
        reward = compute_directional_reward("HIGHER", "HIGHER", confidence=0.5)
        assert reward == 0.0
    
    def test_max_confidence_correct(self):
        """Test maximum confidence correct prediction."""
        # 100% confidence correct → +1.0 reward
        reward = compute_directional_reward("HIGHER", "HIGHER", confidence=1.0)
        assert reward == 1.0
    
    def test_max_confidence_wrong(self):
        """Test maximum confidence wrong prediction."""
        # 100% confidence wrong → -1.0 penalty
        reward = compute_directional_reward("HIGHER", "LOWER", confidence=1.0)
        assert reward == -1.0
    
    def test_confidence_below_05_clipped(self):
        """Test that confidence below 0.5 is clipped to 0 scaled_confidence."""
        # This shouldn't happen in practice, but test defensive clipping
        reward = compute_directional_reward("HIGHER", "HIGHER", confidence=0.3)
        assert reward == 0.0  # Clipped to 0


class TestComputeMAEReward:
    """Test MAE penalty computation."""
    
    def test_zero_mae_returns_zero(self):
        """Test that zero MAE (no adverse excursion) returns zero."""
        reward = compute_mae_reward(0.0, scale=10.0)
        assert reward == 0.0
    
    def test_small_mae_penalty(self):
        """Test small MAE produces proportional penalty."""
        # -5% MAE with scale=10 → -0.5 penalty
        reward = compute_mae_reward(-0.05, scale=10.0)
        assert reward == pytest.approx(-0.5, abs=0.001)
    
    def test_moderate_mae_penalty(self):
        """Test moderate MAE penalty."""
        # -10% MAE with scale=10 → -1.0 penalty
        reward = compute_mae_reward(-0.10, scale=10.0)
        assert reward == -1.0
    
    def test_large_mae_clipped(self):
        """Test that large MAE is clipped to -1.0."""
        # -20% MAE with scale=10 → -2.0 before clip → -1.0 after
        reward = compute_mae_reward(-0.20, scale=10.0)
        assert reward == -1.0
    
    def test_positive_mae_clipped_to_zero(self):
        """Test that positive MAE (shouldn't happen) is clipped to 0."""
        # Defensive: MAE should be <= 0 by convention
        reward = compute_mae_reward(0.05, scale=10.0)
        assert reward == 0.0
    
    def test_custom_scale_factor(self):
        """Test with custom scale factor."""
        # -10% MAE with scale=5 → -0.5 penalty
        reward = compute_mae_reward(-0.10, scale=5.0)
        assert reward == pytest.approx(-0.5, abs=0.001)
    
    def test_mae_reward_always_non_positive(self):
        """Test that MAE reward is always <= 0."""
        test_cases = [0.0, -0.01, -0.05, -0.10, -0.50, -1.0]
        
        for mae in test_cases:
            reward = compute_mae_reward(mae, scale=10.0)
            assert reward <= 0.0


class TestClipReward:
    """Test reward clipping function."""
    
    def test_value_within_range_unchanged(self):
        """Test that values within range are unchanged."""
        assert clip_reward(0.5) == 0.5
        assert clip_reward(-0.3) == -0.3
        assert clip_reward(0.0) == 0.0
    
    def test_value_above_max_clipped(self):
        """Test that values above max are clipped."""
        assert clip_reward(1.5) == 1.0
        assert clip_reward(2.0) == 1.0
    
    def test_value_below_min_clipped(self):
        """Test that values below min are clipped."""
        assert clip_reward(-1.5) == -1.0
        assert clip_reward(-2.0) == -1.0
    
    def test_exact_boundaries(self):
        """Test exact boundary values."""
        assert clip_reward(1.0) == 1.0
        assert clip_reward(-1.0) == -1.0
    
    def test_custom_bounds(self):
        """Test with custom min/max bounds."""
        assert clip_reward(0.5, min_val=-0.5, max_val=0.5) == 0.5
        assert clip_reward(1.0, min_val=-0.5, max_val=0.5) == 0.5
        assert clip_reward(-1.0, min_val=-0.5, max_val=0.5) == -0.5
