"""Tests for GRPO trainer."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from training.grpo_trainer import (
    parse_direction,
    GRPOStepResult,
    GRPOTrainingResult,
    run_grpo_preflight,
    log_vram_usage,
    compute_kl_divergence,
)
from training.grpo_data import GRPOTrainingExample


class TestParseDirection:
    """Tests for parse_direction function."""

    def test_parse_long_uppercase(self) -> None:
        """Test parsing LONG from uppercase."""
        completion = "## THESIS\nBullish\n## EVIDENCE\nRSI\n## RISK\nVol\n## DECISION\nLONG"
        assert parse_direction(completion) == "LONG"

    def test_parse_long_lowercase(self) -> None:
        """Test parsing long from lowercase."""
        completion = "## DECISION\nlong"
        assert parse_direction(completion) == "LONG"

    def test_parse_short(self) -> None:
        """Test parsing SHORT."""
        completion = "## DECISION\nSHORT"
        assert parse_direction(completion) == "SHORT"

    def test_parse_higher_maps_to_long(self) -> None:
        """Test that HIGHER maps to LONG."""
        completion = "## DECISION\nHIGHER"
        assert parse_direction(completion) == "LONG"

    def test_parse_lower_maps_to_short(self) -> None:
        """Test that LOWER maps to SHORT."""
        completion = "## DECISION\nlower"
        assert parse_direction(completion) == "SHORT"

    def test_parse_flat(self) -> None:
        """Test parsing FLAT."""
        completion = "## DECISION\nFLAT"
        assert parse_direction(completion) == "FLAT"

    def test_parse_neutral_maps_to_flat(self) -> None:
        """Test that NEUTRAL maps to FLAT."""
        completion = "## DECISION\nneutral"
        assert parse_direction(completion) == "FLAT"

    def test_corrupt_completion_defaults_flat(self) -> None:
        """Test that unparseable completion defaults to FLAT."""
        completion = "This is garbage text with no structure"
        assert parse_direction(completion) == "FLAT"

    def test_missing_decision_section_defaults_flat(self) -> None:
        """Test that missing DECISION section defaults to FLAT."""
        completion = "## THESIS\nBullish\n## EVIDENCE\nRSI"
        assert parse_direction(completion) == "FLAT"

    def test_direction_in_middle_of_text(self) -> None:
        """Test parsing direction when surrounded by other text."""
        completion = "## DECISION\nBased on analysis, I recommend going LONG with caution."
        assert parse_direction(completion) == "LONG"


class TestGRPOStepResult:
    """Tests for GRPOStepResult dataclass."""

    def test_create_step_result(self) -> None:
        """Test creating a step result."""
        result = GRPOStepResult(
            step=100,
            mean_reward=0.23,
            mean_advantage=0.0,
            kl_divergence=0.012,
            loss=0.45,
            vram_mb=10240,
        )
        assert result.step == 100
        assert result.mean_reward == 0.23
        assert result.kl_divergence == 0.012

    def test_step_result_to_dict(self) -> None:
        """Test converting step result to dictionary."""
        result = GRPOStepResult(
            step=100,
            mean_reward=0.23,
            mean_advantage=0.0,
            kl_divergence=0.012,
            loss=0.45,
            vram_mb=10240,
        )
        d = result.to_dict()
        assert d["step"] == 100
        assert d["mean_reward"] == 0.23
        assert "timestamp" in d


class TestGRPOTrainingResult:
    """Tests for GRPOTrainingResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful training result."""
        result = GRPOTrainingResult(
            success=True,
            adapter_path=Path("adapters/grpo_latest"),
            steps_completed=5000,
            final_metrics={"mean_reward": 0.25, "kl": 0.01},
            error=None,
        )
        assert result.success is True
        assert result.adapter_path == Path("adapters/grpo_latest")
        assert result.steps_completed == 5000

    def test_create_failure_result(self) -> None:
        """Test creating a failed training result."""
        result = GRPOTrainingResult(
            success=False,
            adapter_path=None,
            steps_completed=0,
            final_metrics={},
            error="Preflight failed: lock unavailable",
        )
        assert result.success is False
        assert result.adapter_path is None
        assert "lock" in result.error


class TestPreflightChecks:
    """Tests for GRPO preflight checks."""

    @pytest.fixture
    def sample_examples(self) -> list[GRPOTrainingExample]:
        """Create sample training examples."""
        return [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG",
                gross_return_pct=0.1,
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(10)
        ]

    def test_preflight_empty_examples_fails(self) -> None:
        """Test that empty examples list fails preflight."""
        ok, reason = run_grpo_preflight([])
        assert ok is False
        assert "empty" in reason.lower()

    def test_preflight_unsorted_examples_fails(self) -> None:
        """Test that unsorted examples fail preflight."""
        examples = [
            GRPOTrainingExample(
                market_snapshot="second",
                actual_direction="LONG",
                gross_return_pct=0.1,
                timestamp_ms=2000,
            ),
            GRPOTrainingExample(
                market_snapshot="first",
                actual_direction="SHORT",
                gross_return_pct=-0.1,
                timestamp_ms=1000,
            ),
        ]
        ok, reason = run_grpo_preflight(examples)
        assert ok is False
        assert "sorted" in reason.lower() or "temporal" in reason.lower()

    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_insufficient_vram_fails(
        self,
        mock_vram: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that insufficient VRAM fails preflight."""
        mock_vram.return_value = MagicMock(can_train=False, reason="Only 4GB free")
        ok, reason = run_grpo_preflight(sample_examples)
        assert ok is False
        assert "VRAM" in reason or "4GB" in reason

    @patch("training.grpo_trainer.check_can_train")
    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_lock_unavailable_fails(
        self,
        mock_vram: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that unavailable lock fails preflight."""
        mock_vram.return_value = MagicMock(can_train=True, reason="OK")
        mock_lock.return_value = (False, "Another training process is running")
        ok, reason = run_grpo_preflight(sample_examples)
        assert ok is False
        assert "lock" in reason.lower() or "training" in reason.lower()

    @patch("training.grpo_trainer.check_can_train")
    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_stop_file_fails(
        self,
        mock_vram: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
        tmp_path: Path,
    ) -> None:
        """Test that STOP file fails preflight."""
        mock_vram.return_value = MagicMock(can_train=True, reason="OK")
        mock_lock.return_value = (True, "Ready")

        # Create STOP file
        stop_dir = tmp_path / "execution" / "state"
        stop_dir.mkdir(parents=True)
        (stop_dir / "STOP").touch()

        with patch("training.grpo_trainer.STOP_FILE_PATH", stop_dir / "STOP"):
            ok, reason = run_grpo_preflight(sample_examples)
            assert ok is False
            assert "STOP" in reason

    @patch("training.grpo_trainer.check_can_train")
    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_success(
        self,
        mock_vram: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test successful preflight."""
        mock_vram.return_value = MagicMock(can_train=True, reason="OK", free_mb=12288)
        mock_lock.return_value = (True, "Ready")

        with patch("training.grpo_trainer.STOP_FILE_PATH", Path("/nonexistent/STOP")):
            ok, reason = run_grpo_preflight(sample_examples)
            assert ok is True
            assert "ready" in reason.lower()


class TestVRAMMonitoring:
    """Tests for VRAM monitoring."""

    @patch("training.grpo_trainer.torch")
    def test_log_vram_returns_usage(self, mock_torch: MagicMock) -> None:
        """Test that log_vram returns VRAM usage in MB."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        vram_mb = log_vram_usage(step=100)
        assert vram_mb == 10 * 1024  # 10GB in MB

    @patch("training.grpo_trainer.torch")
    @patch("training.grpo_trainer.logger")
    def test_vram_warning_above_14gb(self, mock_logger: MagicMock, mock_torch: MagicMock) -> None:
        """Test warning logged when VRAM exceeds 14GB."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 15 * 1024 * 1024 * 1024  # 15GB

        log_vram_usage(step=100)
        mock_logger.warning.assert_called_once()
        assert (
            "14GB" in str(mock_logger.warning.call_args)
            or "exceeded" in str(mock_logger.warning.call_args).lower()
        )

    @patch("training.grpo_trainer.torch")
    @patch("training.grpo_trainer.logger")
    def test_no_warning_under_14gb(self, mock_logger: MagicMock, mock_torch: MagicMock) -> None:
        """Test no warning when VRAM is under 14GB."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        log_vram_usage(step=100)
        mock_logger.warning.assert_not_called()

    @patch("training.grpo_trainer.torch")
    def test_vram_no_cuda_returns_zero(self, mock_torch: MagicMock) -> None:
        """Test that no CUDA returns 0."""
        mock_torch.cuda.is_available.return_value = False

        vram_mb = log_vram_usage(step=100)
        assert vram_mb == 0


class TestCheckpointing:
    """Tests for GRPO checkpointing."""

    def test_config_hash_deterministic(self) -> None:
        """Test that config hash is deterministic."""
        from training.grpo_trainer import compute_config_hash
        from training.grpo_config import GRPOTrainingConfig

        config = GRPOTrainingConfig()
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated hash

    def test_config_hash_changes_with_params(self) -> None:
        """Test that config hash changes when params change."""
        from training.grpo_trainer import compute_config_hash
        from training.grpo_config import load_grpo_config

        config1 = load_grpo_config()
        config2 = load_grpo_config({"learning_rate": 1e-4})
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 != hash2

    @patch("training.grpo_trainer.time.time")
    def test_save_checkpoint_creates_metadata(self, mock_time: MagicMock, tmp_path: Path) -> None:
        """Test that checkpoint saves metadata file."""
        import json
        from training.grpo_trainer import save_grpo_checkpoint
        from training.grpo_config import GRPOTrainingConfig

        mock_time.return_value = 1700000000.0

        # Mock model
        mock_model = MagicMock()

        checkpoint_dir = tmp_path / "checkpoint-500"
        config = GRPOTrainingConfig()
        metrics = {"mean_reward": 0.25, "kl": 0.01, "loss": 0.5}

        save_grpo_checkpoint(
            model=mock_model,
            checkpoint_dir=checkpoint_dir,
            step=500,
            config=config,
            metrics=metrics,
        )

        # Check model.save_pretrained was called
        mock_model.save_pretrained.assert_called_once_with(str(checkpoint_dir))

        # Check metadata file exists
        metadata_path = checkpoint_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata contents
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["step"] == 500
        assert metadata["mean_reward"] == 0.25
        assert metadata["timestamp_ms"] == 1700000000000
        assert "config_hash" in metadata


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_identical_distributions_is_zero(self) -> None:
        """Test that KL divergence of identical distributions is 0."""
        policy_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        ref_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        assert abs(kl) < 1e-6

    def test_kl_always_non_negative(self) -> None:
        """Test that KL divergence is non-negative."""
        policy_logprobs = torch.tensor([-1.0, -1.5, -2.0])
        ref_logprobs = torch.tensor([-2.0, -2.5, -3.0])
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        assert kl >= 0

    def test_kl_divergence_value(self) -> None:
        """Test KL divergence with known values."""
        # KL = sum(policy_prob * (policy_logprob - ref_logprob))
        # For log probs, this is sum(exp(policy_logprob) * (policy_logprob - ref_logprob))
        # Simplified: mean(policy_logprob - ref_logprob) when using log space approximation
        policy_logprobs = torch.tensor([-1.0, -2.0])
        ref_logprobs = torch.tensor([-1.5, -2.5])
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        # Expected: mean([0.5, 0.5]) = 0.5
        assert abs(kl - 0.5) < 1e-6
