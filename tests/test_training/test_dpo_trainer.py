"""Tests for DPO training pipeline.

These tests focus on:
1. Pre-flight checks (data, temporal, VRAM validation)
2. Configuration construction
3. Promotion cooldown logic
4. Integration flow (with mocked ML components)

Note: Actual model loading/training tests require GPU and are marked as slow.

IMPORTANT: These tests require training dependencies (torch, transformers, etc).
Run with the training environment:
    pip install -r requirements-training.txt
    pytest tests/test_training/test_dpo_trainer.py -v
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Check if training dependencies are available BEFORE any training imports
try:
    import torch
    import transformers
    import peft
    import trl
    import datasets

    TRAINING_DEPS_AVAILABLE = True
except ImportError:
    TRAINING_DEPS_AVAILABLE = False

# Collect.ignore if dependencies not available
if not TRAINING_DEPS_AVAILABLE:
    # This prevents pytest from trying to collect tests from this module
    collect_ignore = [__file__]
    pytest.skip(
        "Training dependencies not installed (torch, transformers, peft, trl, datasets)",
        allow_module_level=True,
    )

# Only import training modules if dependencies are available
from training.dpo_export import PreferencePair
from training.dpo_trainer import (
    InsufficientDataError,
    TrainingConfig,
    TrainingResult,
    VRAMError,
    _check_promotion_cooldown,
    _create_dpo_config,
    _create_lora_config,
    _create_quantization_config,
    check_should_retrain,
    run_preflight_checks,
    PROMOTION_COOLDOWN_HOURS,
    MAX_PROMOTIONS_PER_WEEK,
)
from training.vram_check import VRAMStatus


@pytest.fixture
def sample_preference_pair():
    """Factory for creating test preference pairs."""

    def _make_pair(timestamp_ms: int, context_id: str = "ctx-1") -> PreferencePair:
        return PreferencePair(
            prompt="Analyze BTC/USDT chart",
            context_id=context_id,
            chosen_reasoning="Strong upward momentum with RSI confirmation",
            chosen_direction="HIGHER",
            chosen_confidence=0.85,
            chosen_reward=0.75,
            chosen_example_id=f"ex-chosen-{timestamp_ms}",
            chosen_persona="MOMENTUM",
            rejected_reasoning="Overbought conditions suggest reversal",
            rejected_direction="LOWER",
            rejected_confidence=0.65,
            rejected_reward=0.35,
            rejected_example_id=f"ex-rejected-{timestamp_ms}",
            rejected_persona="CONTRARIAN",
            reward_delta=0.40,
            symbol="BTC/USDT",
            timestamp_ms=timestamp_ms,
            market_regime="NEUTRAL",
        )

    return _make_pair


@pytest.fixture
def many_preference_pairs(sample_preference_pair):
    """Create enough preference pairs for training (700 pairs)."""
    base_time = 1609459200000  # 2021-01-01
    hour_ms = 3600 * 1000

    pairs = []
    for i in range(700):
        pairs.append(
            sample_preference_pair(
                timestamp_ms=base_time + (i * hour_ms),
                context_id=f"ctx-{i}",
            )
        )

    return pairs


@pytest.fixture
def few_preference_pairs(sample_preference_pair):
    """Create insufficient preference pairs (100 pairs)."""
    base_time = 1609459200000
    hour_ms = 3600 * 1000

    pairs = []
    for i in range(100):
        pairs.append(
            sample_preference_pair(
                timestamp_ms=base_time + (i * hour_ms),
                context_id=f"ctx-{i}",
            )
        )

    return pairs


@pytest.fixture
def mock_vram_available():
    """Mock VRAM check returning sufficient VRAM."""
    return VRAMStatus(
        total_mb=16384,
        used_mb=2048,
        free_mb=14336,
        gpu_name="NVIDIA RTX 5070 Ti",
        can_train=True,
        reason="Sufficient VRAM available: 14.0 GB free",
    )


@pytest.fixture
def mock_vram_insufficient():
    """Mock VRAM check returning insufficient VRAM."""
    return VRAMStatus(
        total_mb=8192,
        used_mb=7168,
        free_mb=1024,
        gpu_name="NVIDIA RTX 3060",
        can_train=False,
        reason="Insufficient VRAM: 1.0 GB free, need 9.0 GB minimum",
    )


@pytest.fixture
def temp_promotions_log(tmp_path, monkeypatch):
    """Create temporary promotions log file."""
    log_file = tmp_path / "promotions.jsonl"
    monkeypatch.setattr(
        "training.dpo_trainer.PROMOTIONS_LOG_FILE",
        log_file,
    )
    return log_file


class TestTrainingConfig:
    """Test TrainingConfig construction."""

    def test_from_settings(self, env_vars):
        """Test config is constructed from settings."""
        config = TrainingConfig.from_settings()

        assert config.base_model_id == "Qwen/Qwen3-8B"
        assert config.lora_rank == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.05
        assert config.dpo_beta == 0.1
        assert config.learning_rate == 5e-6
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 16

    def test_lora_target_modules(self, env_vars):
        """Test LoRA targets attention + MLP modules."""
        config = TrainingConfig.from_settings()

        expected_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP
        ]
        assert config.lora_target_modules == expected_modules


class TestQuantizationConfig:
    """Test BitsAndBytesConfig construction."""

    def test_4bit_nf4_config(self):
        """Test 4-bit NF4 quantization config."""
        bnb_config = _create_quantization_config()

        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
        assert bnb_config.bnb_4bit_use_double_quant is True


class TestLoraConfig:
    """Test LoraConfig construction."""

    def test_lora_config_from_training_config(self, env_vars):
        """Test LoRA config matches training config."""
        training_config = TrainingConfig.from_settings()
        lora_config = _create_lora_config(training_config)

        assert lora_config.r == 32
        assert lora_config.lora_alpha == 64
        assert lora_config.lora_dropout == 0.05
        assert lora_config.bias == "none"
        assert lora_config.task_type.value == "CAUSAL_LM"


class TestDPOConfig:
    """Test DPOConfig construction."""

    def test_dpo_config_from_training_config(self, env_vars, tmp_path):
        """Test DPO config has correct settings."""
        training_config = TrainingConfig.from_settings()
        dpo_config = _create_dpo_config(training_config, tmp_path / "output")

        assert dpo_config.per_device_train_batch_size == 1
        assert dpo_config.gradient_accumulation_steps == 16
        assert dpo_config.learning_rate == 5e-6
        assert dpo_config.beta == 0.1
        assert dpo_config.loss_type == "sigmoid"
        assert dpo_config.precompute_ref_log_probs is True  # Critical for VRAM
        assert dpo_config.gradient_checkpointing is True
        assert dpo_config.bf16 is True


class TestPreflightChecks:
    """Test pre-flight validation checks."""

    def test_insufficient_data_raises(self, few_preference_pairs):
        """Test InsufficientDataError when not enough pairs."""
        with pytest.raises(InsufficientDataError) as exc_info:
            run_preflight_checks(few_preference_pairs)

        assert "Insufficient preference pairs" in str(exc_info.value)

    def test_sufficient_data_passes(self, many_preference_pairs, mock_vram_available):
        """Test preflight passes with sufficient data and VRAM."""
        with patch(
            "training.dpo_trainer.check_vram_availability",
            return_value=mock_vram_available,
        ):
            split, vram_status = run_preflight_checks(many_preference_pairs)

            assert len(split.train_pairs) == 500
            assert len(split.test_pairs) == 100
            assert vram_status.can_train is True

    def test_insufficient_vram_raises(self, many_preference_pairs, mock_vram_insufficient):
        """Test VRAMError when insufficient VRAM."""
        with patch(
            "training.dpo_trainer.check_vram_availability",
            return_value=mock_vram_insufficient,
        ):
            with pytest.raises(VRAMError) as exc_info:
                run_preflight_checks(many_preference_pairs)

            assert "Insufficient VRAM" in str(exc_info.value)

    def test_temporal_split_is_validated(self, many_preference_pairs, mock_vram_available):
        """Test that temporal split validation is performed."""
        with patch(
            "training.dpo_trainer.check_vram_availability",
            return_value=mock_vram_available,
        ):
            split, _ = run_preflight_checks(many_preference_pairs)

            # Verify temporal ordering
            assert split.test_start_ms > split.train_end_ms
            for pair in split.replay_pairs:
                assert pair.timestamp_ms < split.train_start_ms


class TestPromotionCooldown:
    """Test promotion rate limiting."""

    def test_no_previous_promotions(self, temp_promotions_log):
        """Test promotion allowed when no history."""
        can_promote, reason = _check_promotion_cooldown()

        assert can_promote is True
        assert "No previous promotions" in reason

    def test_cooldown_blocks_promotion(self, temp_promotions_log):
        """Test cooldown blocks recent promotions."""
        # Write recent promotion
        recent_record = {
            "timestamp": time.time() - 3600,  # 1 hour ago
            "adapter_path": "/path/to/adapter",
            "ic": 0.07,
        }
        with open(temp_promotions_log, "w") as f:
            f.write(json.dumps(recent_record) + "\n")

        can_promote, reason = _check_promotion_cooldown()

        assert can_promote is False
        assert "Cooldown active" in reason

    def test_cooldown_allows_after_period(self, temp_promotions_log):
        """Test promotion allowed after cooldown expires."""
        # Write old promotion
        old_record = {
            "timestamp": time.time() - (PROMOTION_COOLDOWN_HOURS + 1) * 3600,
            "adapter_path": "/path/to/adapter",
            "ic": 0.07,
        }
        with open(temp_promotions_log, "w") as f:
            f.write(json.dumps(old_record) + "\n")

        can_promote, reason = _check_promotion_cooldown()

        assert can_promote is True

    def test_weekly_limit_blocks_promotion(self, temp_promotions_log):
        """Test weekly rate limit blocks excess promotions."""
        # Write MAX_PROMOTIONS_PER_WEEK promotions in past week
        records = []
        for i in range(MAX_PROMOTIONS_PER_WEEK):
            # Spread across the week, but all outside cooldown
            timestamp = time.time() - (PROMOTION_COOLDOWN_HOURS + 24 + i) * 3600
            records.append(
                {
                    "timestamp": timestamp,
                    "adapter_path": f"/path/to/adapter-{i}",
                    "ic": 0.07,
                }
            )

        with open(temp_promotions_log, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        can_promote, reason = _check_promotion_cooldown()

        assert can_promote is False
        assert "Weekly limit" in reason


class TestCheckShouldRetrain:
    """Test retraining trigger logic."""

    def test_insufficient_total_pairs(self):
        """Test no retrain when total pairs below threshold."""
        # Default: train_window=500, test_window=100 -> need 600
        should_retrain, reason = check_should_retrain(
            current_pair_count=400,
            last_training_pair_count=0,
        )

        assert should_retrain is False
        assert "Insufficient total pairs" in reason

    def test_insufficient_new_pairs(self):
        """Test no retrain when new pairs below threshold."""
        # Default: retrain_threshold=250
        should_retrain, reason = check_should_retrain(
            current_pair_count=700,
            last_training_pair_count=550,  # Only 150 new
        )

        assert should_retrain is False
        assert "Insufficient new pairs" in reason

    def test_retrain_triggered(self):
        """Test retrain triggers with sufficient new data."""
        should_retrain, reason = check_should_retrain(
            current_pair_count=900,
            last_training_pair_count=600,  # 300 new pairs
        )

        assert should_retrain is True
        assert "Retrain triggered" in reason


class TestTrainingResult:
    """Test TrainingResult serialization."""

    def test_to_dict_success(self):
        """Test successful result serialization."""
        result = TrainingResult(
            success=True,
            adapter_path=Path("/models/adapter-123"),
            evaluation=None,
            promoted=False,
            promotion_reason="Awaiting evaluation",
            training_time_seconds=1234.5,
            num_training_pairs=575,
            num_test_pairs=100,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["adapter_path"] == "/models/adapter-123"
        assert result_dict["training_time_seconds"] == 1234.5
        assert result_dict["error"] is None

    def test_to_dict_failure(self):
        """Test failed result serialization."""
        result = TrainingResult(
            success=False,
            adapter_path=None,
            evaluation=None,
            promoted=False,
            promotion_reason="VRAM insufficient",
            training_time_seconds=5.0,
            num_training_pairs=0,
            num_test_pairs=0,
            error="VRAMError: Only 4GB available",
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["adapter_path"] is None
        assert "VRAMError" in result_dict["error"]


class TestTrainDPOIntegration:
    """Integration tests for train_dpo function.

    These tests mock the heavy ML components and verify the flow.
    """

    def test_preflight_failure_returns_early(self, few_preference_pairs):
        """Test that preflight failure returns error result without acquiring lock."""
        from training.dpo_trainer import train_dpo

        result = train_dpo(few_preference_pairs)

        assert result.success is False
        assert "Insufficient" in result.error
        assert result.adapter_path is None

    def test_lock_unavailable_returns_error(self, many_preference_pairs, mock_vram_available):
        """Test that lock unavailability returns error result."""
        from training.dpo_trainer import train_dpo

        with (
            patch(
                "training.dpo_trainer.check_vram_availability",
                return_value=mock_vram_available,
            ),
            patch(
                "training.dpo_trainer.check_can_train",
                return_value=(False, "Inference process is running"),
            ),
        ):
            result = train_dpo(many_preference_pairs)

            assert result.success is False
            assert "Inference process" in result.error


# Tests that require GPU are marked for optional execution
@pytest.mark.slow
@pytest.mark.gpu
class TestTrainDPOWithGPU:
    """GPU-dependent integration tests.

    Run with: pytest -m "slow and gpu" --no-header
    """

    def test_model_loading(self, env_vars):
        """Test actual model loading with quantization."""
        pytest.skip("Requires GPU and model download")

    def test_full_training_run(self, many_preference_pairs, env_vars):
        """Test full training pipeline on GPU."""
        pytest.skip("Requires GPU and significant time")
