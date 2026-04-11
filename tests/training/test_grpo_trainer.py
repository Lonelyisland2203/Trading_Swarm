"""Tests for GRPO trainer."""

from pathlib import Path

from training.grpo_trainer import (
    parse_direction,
    GRPOStepResult,
    GRPOTrainingResult,
)


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
