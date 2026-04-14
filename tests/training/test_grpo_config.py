"""
Tests for GRPO configuration module.
"""

import pytest
from pathlib import Path

from training.grpo_config import (
    GRPORewardConfig,
    GRPOLoRAConfig,
    GRPOTrainingConfig,
    load_grpo_config,
)


class TestGRPORewardConfig:
    """Tests for GRPORewardConfig."""

    def test_default_weights_sum_to_one(self):
        """Default reward weights should sum to 1.0."""
        config = GRPORewardConfig()
        total = config.decision_weight + config.structure_weight + config.directional_weight
        assert abs(total - 1.0) < 1e-6

    def test_default_values(self):
        """Verify default configuration values."""
        config = GRPORewardConfig()
        assert config.decision_weight == 0.6
        assert config.structure_weight == 0.2
        assert config.directional_weight == 0.2
        assert config.false_bullish_penalty == 1.5
        assert config.false_bearish_penalty == 0.8
        assert config.true_bullish_multiplier == 1.0
        assert config.true_bearish_multiplier == 1.0
        assert config.clip_min == -1.0
        assert config.clip_max == 1.0

    def test_custom_weights_valid(self):
        """Custom weights that sum to 1.0 should be accepted."""
        config = GRPORewardConfig(
            decision_weight=0.5,
            structure_weight=0.3,
            directional_weight=0.2,
        )
        assert config.decision_weight == 0.5
        assert config.structure_weight == 0.3
        assert config.directional_weight == 0.2

    def test_custom_weights_invalid_sum(self):
        """Weights that don't sum to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            GRPORewardConfig(
                decision_weight=0.5,
                structure_weight=0.3,
                directional_weight=0.3,  # Sum = 1.1
            )

    def test_custom_asymmetry_coefficients(self):
        """Custom asymmetry coefficients should be accepted."""
        config = GRPORewardConfig(
            false_bullish_penalty=2.0,
            false_bearish_penalty=0.5,
        )
        assert config.false_bullish_penalty == 2.0
        assert config.false_bearish_penalty == 0.5

    def test_frozen_dataclass(self):
        """Config should be immutable (frozen)."""
        config = GRPORewardConfig()
        with pytest.raises(AttributeError):
            config.decision_weight = 0.8  # type: ignore


class TestGRPOLoRAConfig:
    """Tests for GRPOLoRAConfig."""

    def test_default_values(self):
        """Verify default LoRA configuration."""
        config = GRPOLoRAConfig()
        assert config.rank == 32
        assert config.alpha == 64
        assert config.dropout == 0.05
        assert config.bias == "none"
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules
        assert "gate_proj" in config.target_modules
        assert "up_proj" in config.target_modules
        assert "down_proj" in config.target_modules

    def test_custom_rank_alpha(self):
        """Custom rank and alpha should be accepted."""
        config = GRPOLoRAConfig(rank=16, alpha=32)
        assert config.rank == 16
        assert config.alpha == 32


class TestGRPOTrainingConfig:
    """Tests for GRPOTrainingConfig."""

    def test_default_values(self):
        """Verify default training configuration."""
        config = GRPOTrainingConfig()
        assert config.group_size == 4
        assert config.kl_penalty_beta == 0.04
        assert config.clip_epsilon == 0.2
        assert config.learning_rate == 2e-5
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 16
        assert config.max_steps == 5000
        assert config.checkpoint_interval_steps == 500

    def test_effective_batch_size(self):
        """Effective batch size should be batch_size * grad_accum."""
        config = GRPOTrainingConfig(batch_size=2, gradient_accumulation_steps=8)
        assert config.effective_batch_size == 16

    def test_nested_configs(self):
        """Nested LoRA and reward configs should be accessible."""
        config = GRPOTrainingConfig()
        assert config.lora.rank == 32
        assert config.reward.decision_weight == 0.6

    def test_custom_nested_configs(self):
        """Custom nested configs should work."""
        reward_config = GRPORewardConfig(
            decision_weight=0.5,
            structure_weight=0.25,
            directional_weight=0.25,
        )
        config = GRPOTrainingConfig(reward=reward_config)
        assert config.reward.decision_weight == 0.5

    def test_path_defaults(self):
        """Default paths should be Path objects."""
        config = GRPOTrainingConfig()
        assert isinstance(config.checkpoint_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.sft_adapter_path, Path)


class TestLoadGRPOConfig:
    """Tests for load_grpo_config function."""

    def test_no_overrides(self):
        """Loading with no overrides returns defaults."""
        config = load_grpo_config()
        assert config.group_size == 4
        assert config.learning_rate == 2e-5

    def test_simple_overrides(self):
        """Simple parameter overrides should work."""
        config = load_grpo_config({"max_steps": 1000, "learning_rate": 1e-5})
        assert config.max_steps == 1000
        assert config.learning_rate == 1e-5

    def test_nested_reward_overrides(self):
        """Nested reward parameter overrides should work."""
        config = load_grpo_config(
            {
                "reward.false_bullish_penalty": 2.0,
                "reward.false_bearish_penalty": 0.6,
            }
        )
        assert config.reward.false_bullish_penalty == 2.0
        assert config.reward.false_bearish_penalty == 0.6

    def test_nested_lora_overrides(self):
        """Nested LoRA parameter overrides should work."""
        config = load_grpo_config(
            {
                "lora.rank": 16,
                "lora.alpha": 32,
            }
        )
        assert config.lora.rank == 16
        assert config.lora.alpha == 32

    def test_mixed_overrides(self):
        """Mixed flat and nested overrides should work."""
        config = load_grpo_config(
            {
                "max_steps": 2000,
                "reward.decision_weight": 0.7,
                "reward.structure_weight": 0.15,
                "reward.directional_weight": 0.15,
                "lora.rank": 64,
            }
        )
        assert config.max_steps == 2000
        assert config.reward.decision_weight == 0.7
        assert config.lora.rank == 64
