"""Tests for run_grpo_training.py pipeline CLI."""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_grpo_training import (
    PipelineConfig,
    PhaseResult,
    parse_args,
    print_plan,
    phase_sft_data_generation,
    phase_sft_training,
    phase_grpo_training,
    phase_evaluation,
    phase_promotion,
    DEFAULT_SFT_DATA_PATH,
    DEFAULT_SFT_ADAPTER_PATH,
    DEFAULT_GRPO_DATA_PATH,
)


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_create_default_config(self) -> None:
        """Test creating config with defaults."""
        config = PipelineConfig(
            sft_data_path=DEFAULT_SFT_DATA_PATH,
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=DEFAULT_SFT_ADAPTER_PATH,
            retrain_sft=False,
            grpo_data_path=DEFAULT_GRPO_DATA_PATH,
            max_steps=None,
            grpo_config_path=None,
            dry_run=False,
        )
        assert config.sft_data_path == DEFAULT_SFT_DATA_PATH
        assert config.sft_data_limit is None
        assert config.regenerate_sft_data is False

    def test_create_config_with_overrides(self) -> None:
        """Test creating config with custom values."""
        config = PipelineConfig(
            sft_data_path=Path("custom/sft.jsonl"),
            sft_data_limit=100,
            regenerate_sft_data=True,
            sft_adapter_path=Path("custom/adapter"),
            retrain_sft=True,
            grpo_data_path=Path("custom/grpo.jsonl"),
            max_steps=1000,
            grpo_config_path=Path("custom/config.py"),
            dry_run=True,
        )
        assert config.sft_data_path == Path("custom/sft.jsonl")
        assert config.sft_data_limit == 100
        assert config.regenerate_sft_data is True
        assert config.max_steps == 1000
        assert config.dry_run is True


class TestPhaseResult:
    """Tests for PhaseResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful phase result."""
        result = PhaseResult(
            phase="sft_data_generation",
            success=True,
            duration_seconds=120.5,
            message="Generated 500 examples",
            artifacts={"count": 500},
        )
        assert result.phase == "sft_data_generation"
        assert result.success is True
        assert result.duration_seconds == 120.5
        assert result.artifacts["count"] == 500

    def test_create_failure_result(self) -> None:
        """Test creating a failed phase result."""
        result = PhaseResult(
            phase="grpo_training",
            success=False,
            duration_seconds=10.0,
            message="Failed: VRAM insufficient",
            artifacts={},
        )
        assert result.phase == "grpo_training"
        assert result.success is False
        assert "VRAM" in result.message


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_parse_default_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing with no arguments."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py"])
        args = parse_args()

        assert args.limit is None
        assert args.max_steps is None
        assert args.regenerate is False
        assert args.retrain_sft is False
        assert args.dry_run is False
        assert args.config is None
        assert args.sft_data_path == DEFAULT_SFT_DATA_PATH
        assert args.sft_adapter_path == DEFAULT_SFT_ADAPTER_PATH
        assert args.grpo_data_path == DEFAULT_GRPO_DATA_PATH

    def test_parse_limit_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing --limit flag."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--limit", "50"])
        args = parse_args()
        assert args.limit == 50

    def test_parse_max_steps_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing --max-steps flag."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--max-steps", "1000"])
        args = parse_args()
        assert args.max_steps == 1000

    def test_parse_regenerate_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing --regenerate flag."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--regenerate"])
        args = parse_args()
        assert args.regenerate is True

    def test_parse_retrain_sft_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing --retrain-sft flag."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--retrain-sft"])
        args = parse_args()
        assert args.retrain_sft is True

    def test_parse_dry_run_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing --dry-run flag."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--dry-run"])
        args = parse_args()
        assert args.dry_run is True

    def test_parse_config_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing --config flag."""
        monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--config", "custom_config.py"])
        args = parse_args()
        assert args.config == Path("custom_config.py")

    def test_parse_custom_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing custom path arguments."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_grpo_training.py",
                "--sft-data-path",
                "custom/sft.jsonl",
                "--sft-adapter-path",
                "custom/adapter",
                "--grpo-data-path",
                "custom/grpo.jsonl",
            ],
        )
        args = parse_args()
        assert args.sft_data_path == Path("custom/sft.jsonl")
        assert args.sft_adapter_path == Path("custom/adapter")
        assert args.grpo_data_path == Path("custom/grpo.jsonl")

    def test_parse_multiple_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing multiple flags together."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_grpo_training.py",
                "--limit",
                "100",
                "--max-steps",
                "2000",
                "--regenerate",
                "--retrain-sft",
                "--dry-run",
            ],
        )
        args = parse_args()
        assert args.limit == 100
        assert args.max_steps == 2000
        assert args.regenerate is True
        assert args.retrain_sft is True
        assert args.dry_run is True


class TestPhaseSFTDataGeneration:
    """Tests for SFT data generation phase."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> PipelineConfig:
        """Create test config with temp paths."""
        return PipelineConfig(
            sft_data_path=tmp_path / "sft_data.jsonl",
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=None,
            grpo_config_path=None,
            dry_run=False,
        )

    def test_skips_when_data_exists(self, config: PipelineConfig) -> None:
        """Test that phase skips when data exists and regenerate not set."""
        # Create existing data file
        config.sft_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.sft_data_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"example": i}) + "\n")

        result = phase_sft_data_generation(config)

        assert result.success is True
        assert "Skipped" in result.message
        assert result.artifacts["count"] == 10

    def test_regenerates_when_flag_set(self, config: PipelineConfig) -> None:
        """Test that phase regenerates when --regenerate is set."""
        # Create existing data file
        config.sft_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.sft_data_path, "w") as f:
            f.write(json.dumps({"example": 0}) + "\n")

        # Set regenerate flag
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=True,  # Changed
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=config.retrain_sft,
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,  # Use dry run to avoid actual generation
        )

        result = phase_sft_data_generation(config)

        assert result.success is True
        assert "DRY RUN" in result.message

    def test_dry_run_returns_success(self, config: PipelineConfig) -> None:
        """Test that dry run returns success without executing."""
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=config.regenerate_sft_data,
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=config.retrain_sft,
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,
        )

        result = phase_sft_data_generation(config)

        assert result.success is True
        assert "DRY RUN" in result.message


class TestPhaseSFTTraining:
    """Tests for SFT training phase."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> PipelineConfig:
        """Create test config with temp paths."""
        return PipelineConfig(
            sft_data_path=tmp_path / "sft_data.jsonl",
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=None,
            grpo_config_path=None,
            dry_run=False,
        )

    def test_skips_when_adapter_exists(self, config: PipelineConfig) -> None:
        """Test that phase skips when adapter exists and retrain not set."""
        # Create existing adapter directory
        config.sft_adapter_path.mkdir(parents=True, exist_ok=True)

        result = phase_sft_training(config)

        assert result.success is True
        assert "Skipped" in result.message

    def test_retrains_when_flag_set(self, config: PipelineConfig) -> None:
        """Test that phase retrains when --retrain-sft is set."""
        # Create existing adapter directory
        config.sft_adapter_path.mkdir(parents=True, exist_ok=True)

        # Set retrain flag
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=config.regenerate_sft_data,
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=True,  # Changed
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,  # Use dry run to avoid actual training
        )

        result = phase_sft_training(config)

        assert result.success is True
        assert "DRY RUN" in result.message

    def test_dry_run_returns_success(self, config: PipelineConfig) -> None:
        """Test that dry run returns success without executing."""
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=config.regenerate_sft_data,
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=config.retrain_sft,
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,
        )

        result = phase_sft_training(config)

        assert result.success is True
        assert "DRY RUN" in result.message


class TestPhaseGRPOTraining:
    """Tests for GRPO training phase."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> PipelineConfig:
        """Create test config with temp paths."""
        return PipelineConfig(
            sft_data_path=tmp_path / "sft_data.jsonl",
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=100,
            grpo_config_path=None,
            dry_run=False,
        )

    def test_fails_when_data_missing(self, config: PipelineConfig) -> None:
        """Test that phase fails when GRPO data is missing."""
        result = phase_grpo_training(config)

        assert result.success is False
        assert "not found" in result.message

    def test_dry_run_returns_success(self, config: PipelineConfig) -> None:
        """Test that dry run returns success without executing."""
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=config.regenerate_sft_data,
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=config.retrain_sft,
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,
        )

        result = phase_grpo_training(config)

        assert result.success is True
        assert "DRY RUN" in result.message

    def test_fails_when_data_empty(self, config: PipelineConfig, tmp_path: Path) -> None:
        """Test that phase fails when GRPO data file is empty."""
        # Create empty data file
        config.grpo_data_path.parent.mkdir(parents=True, exist_ok=True)
        config.grpo_data_path.touch()

        result = phase_grpo_training(config)

        assert result.success is False
        assert "No GRPO training examples" in result.message


class TestPhaseEvaluation:
    """Tests for evaluation phase."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> PipelineConfig:
        """Create test config with temp paths."""
        return PipelineConfig(
            sft_data_path=tmp_path / "sft_data.jsonl",
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=None,
            grpo_config_path=None,
            dry_run=False,
        )

    def test_dry_run_returns_success(self, config: PipelineConfig, tmp_path: Path) -> None:
        """Test that dry run returns success without executing."""
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=config.regenerate_sft_data,
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=config.retrain_sft,
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,
        )

        result = phase_evaluation(config, tmp_path / "adapter")

        assert result.success is True
        assert "DRY RUN" in result.message


class TestPhasePromotion:
    """Tests for promotion phase."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> PipelineConfig:
        """Create test config with temp paths."""
        return PipelineConfig(
            sft_data_path=tmp_path / "sft_data.jsonl",
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=None,
            grpo_config_path=None,
            dry_run=False,
        )

    def test_promotion_fails_when_criteria_not_met(
        self, config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Test that promotion fails when criteria not met."""
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)

        evaluation_artifacts = {
            "passes_promotion": False,
            "promotion_reason": "IC 0.03 < 0.05; Brier 0.30 > 0.25",
        }

        result = phase_promotion(config, adapter_path, evaluation_artifacts)

        assert result.success is False
        assert result.artifacts["promoted"] is False

    def test_promotion_succeeds_when_criteria_met(
        self, config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Test that promotion succeeds when all criteria met."""
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        # Create a dummy file in the adapter directory
        (adapter_path / "adapter_model.bin").touch()

        evaluation_artifacts = {
            "passes_promotion": True,
            "promotion_reason": "All criteria passed",
            "ic": 0.08,
            "brier_score": 0.22,
            "structure_compliance": 0.95,
        }

        result = phase_promotion(config, adapter_path, evaluation_artifacts)

        assert result.success is True
        assert result.artifacts["promoted"] is True
        assert "promoted_path" in result.artifacts

        # Check that promoted directory exists
        promoted_path = Path(result.artifacts["promoted_path"])
        assert promoted_path.exists()
        assert promoted_path.suffix == ".promoted"

    def test_promotion_updates_metadata(self, config: PipelineConfig, tmp_path: Path) -> None:
        """Test that promotion adds metadata to adapter."""
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Create initial metadata
        with open(adapter_path / "metadata.json", "w") as f:
            json.dump({"step": 5000, "config_hash": "abc123"}, f)

        evaluation_artifacts = {
            "passes_promotion": True,
            "promotion_reason": "All criteria passed",
            "ic": 0.08,
            "brier_score": 0.22,
            "structure_compliance": 0.95,
        }

        result = phase_promotion(config, adapter_path, evaluation_artifacts)

        # Check promoted metadata
        promoted_path = Path(result.artifacts["promoted_path"])
        with open(promoted_path / "metadata.json") as f:
            metadata = json.load(f)

        assert "promoted_at" in metadata
        assert "promotion_metrics" in metadata
        assert metadata["promotion_metrics"]["ic"] == 0.08

    def test_dry_run_returns_success(self, config: PipelineConfig, tmp_path: Path) -> None:
        """Test that dry run returns success without executing."""
        config = PipelineConfig(
            sft_data_path=config.sft_data_path,
            sft_data_limit=config.sft_data_limit,
            regenerate_sft_data=config.regenerate_sft_data,
            sft_adapter_path=config.sft_adapter_path,
            retrain_sft=config.retrain_sft,
            grpo_data_path=config.grpo_data_path,
            max_steps=config.max_steps,
            grpo_config_path=config.grpo_config_path,
            dry_run=True,
        )

        result = phase_promotion(config, tmp_path / "adapter", {})

        assert result.success is True
        assert "DRY RUN" in result.message


class TestPrintPlan:
    """Tests for print_plan function."""

    def test_print_plan_outputs_all_phases(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Test that print_plan outputs all phases."""
        config = PipelineConfig(
            sft_data_path=tmp_path / "sft_data.jsonl",
            sft_data_limit=50,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=1000,
            grpo_config_path=None,
            dry_run=True,
        )

        print_plan(config)

        captured = capsys.readouterr()
        assert "SFT Data Generation" in captured.out
        assert "SFT Training" in captured.out
        assert "GRPO Training" in captured.out
        assert "Evaluation" in captured.out
        assert "Promotion Gate" in captured.out

    def test_print_plan_shows_skip_status(
        self, capsys: pytest.CaptureFixture, tmp_path: Path
    ) -> None:
        """Test that print_plan shows skip status correctly."""
        # Create existing data
        sft_data_path = tmp_path / "sft_data.jsonl"
        sft_data_path.touch()

        config = PipelineConfig(
            sft_data_path=sft_data_path,
            sft_data_limit=None,
            regenerate_sft_data=False,
            sft_adapter_path=tmp_path / "sft_adapter",
            retrain_sft=False,
            grpo_data_path=tmp_path / "grpo_data.jsonl",
            max_steps=None,
            grpo_config_path=None,
            dry_run=True,
        )

        print_plan(config)

        captured = capsys.readouterr()
        # Should show skip as True for SFT data generation
        assert "Skip: True" in captured.out
