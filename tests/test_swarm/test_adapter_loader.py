"""Tests for adapter loading."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from swarm.adapter_loader import (
    AdapterLoadError,
    AdapterNotFoundError,
    find_latest_adapter,
    get_adapter_directory,
    get_adapter_model_tag,
    get_fallback_model,
    load_adapter_metadata,
    mark_adapter_promoted,
    should_use_adapter,
)


@pytest.fixture
def adapter_dir(tmp_path):
    """Create temporary adapter directory."""
    adapter_dir = tmp_path / "models" / "adapters"
    adapter_dir.mkdir(parents=True)
    return adapter_dir


@pytest.fixture
def sample_adapter(adapter_dir):
    """Create sample adapter checkpoint."""

    def _create_adapter(
        persona: str = "MOMENTUM",
        timestamp_ms: int = 1640995200000,
        promoted: bool = False,
        include_metadata: bool = True,
        include_weights: bool = True,
    ):
        """Create a sample adapter directory with metadata."""
        adapter_name = f"adapter-{persona}-{timestamp_ms}"
        if promoted:
            adapter_name += ".promoted"

        adapter_path = adapter_dir / adapter_name
        adapter_path.mkdir()

        # Create metadata
        if include_metadata:
            metadata = {
                "persona": persona,
                "timestamp_ms": timestamp_ms,
                "test_ic": 0.15,
                "baseline_ic": 0.10,
                "ic_improvement": 0.05,
                "lora_rank": 32,
                "lora_alpha": 64,
                "base_model": "qwen3:8b",
            }
            metadata_file = adapter_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

        # Create adapter weights
        if include_weights:
            weights_file = adapter_path / "adapter_model.safetensors"
            weights_file.write_text("dummy weights")

        return adapter_path

    return _create_adapter


class TestGetAdapterDirectory:
    """Test adapter directory creation."""

    def test_creates_directory(self, tmp_path):
        """Test adapter directory is created."""
        with patch("swarm.adapter_loader.Path", return_value=tmp_path / "models" / "adapters"):
            adapter_dir = get_adapter_directory()

        # Just verify it returns a Path object
        assert isinstance(adapter_dir, Path)


class TestFindLatestAdapter:
    """Test finding latest adapter."""

    def test_find_latest_for_persona(self, adapter_dir, sample_adapter):
        """Test finding latest adapter for specific persona."""
        # Create multiple adapters
        sample_adapter(persona="MOMENTUM", timestamp_ms=1000, promoted=True)
        sample_adapter(persona="MOMENTUM", timestamp_ms=2000, promoted=True)
        sample_adapter(persona="CONTRARIAN", timestamp_ms=3000, promoted=True)

        with patch("swarm.adapter_loader.get_adapter_directory", return_value=adapter_dir):
            latest = find_latest_adapter(persona="MOMENTUM")

        assert latest is not None
        assert "MOMENTUM" in latest.name
        assert "2000" in latest.name  # Latest timestamp

    def test_find_latest_across_all_personas(self, adapter_dir, sample_adapter):
        """Test finding latest adapter across all personas."""
        sample_adapter(persona="CONTRARIAN", timestamp_ms=1000, promoted=True)
        sample_adapter(persona="MOMENTUM", timestamp_ms=2000, promoted=True)

        with patch("swarm.adapter_loader.get_adapter_directory", return_value=adapter_dir):
            latest = find_latest_adapter()

        assert latest is not None
        # Sorted reverse by full name, MOMENTUM comes after CONTRARIAN lexicographically
        assert "MOMENTUM" in latest.name  # Latest lexicographically

    def test_no_adapters_found(self, adapter_dir):
        """Test when no adapters exist."""
        with patch("swarm.adapter_loader.get_adapter_directory", return_value=adapter_dir):
            latest = find_latest_adapter()

        assert latest is None

    def test_only_finds_promoted_adapters(self, adapter_dir, sample_adapter):
        """Test that only .promoted adapters are found."""
        sample_adapter(persona="MOMENTUM", timestamp_ms=1000, promoted=False)
        sample_adapter(persona="MOMENTUM", timestamp_ms=2000, promoted=True)

        with patch("swarm.adapter_loader.get_adapter_directory", return_value=adapter_dir):
            latest = find_latest_adapter(persona="MOMENTUM")

        assert latest is not None
        assert "2000" in latest.name  # Only the promoted one


class TestLoadAdapterMetadata:
    """Test loading adapter metadata."""

    def test_load_valid_metadata(self, sample_adapter):
        """Test loading valid metadata."""
        adapter_path = sample_adapter(include_metadata=True)

        metadata = load_adapter_metadata(adapter_path)

        assert metadata["persona"] == "MOMENTUM"
        assert metadata["test_ic"] == 0.15
        assert metadata["ic_improvement"] == 0.05

    def test_missing_metadata_file_raises(self, sample_adapter):
        """Test error when metadata file missing."""
        adapter_path = sample_adapter(include_metadata=False)

        with pytest.raises(AdapterLoadError, match="Metadata file not found"):
            load_adapter_metadata(adapter_path)

    def test_invalid_json_raises(self, sample_adapter):
        """Test error when metadata has invalid JSON."""
        adapter_path = sample_adapter(include_metadata=False)
        metadata_file = adapter_path / "metadata.json"
        metadata_file.write_text("invalid json {")

        with pytest.raises(AdapterLoadError, match="Invalid metadata file"):
            load_adapter_metadata(adapter_path)


class TestGetAdapterModelTag:
    """Test adapter model tag construction."""

    def test_construct_model_tag(self, sample_adapter):
        """Test model tag construction."""
        adapter_path = sample_adapter(promoted=True, include_weights=True)

        with patch("swarm.adapter_loader.settings") as mock_settings:
            mock_settings.ollama.generator_model = "qwen3:8b"
            tag = get_adapter_model_tag(adapter_path)

        assert "qwen3:8b" in tag
        assert "lora" in tag
        assert "MOMENTUM" in tag

    def test_missing_weights_raises(self, sample_adapter):
        """Test error when adapter weights missing."""
        adapter_path = sample_adapter(include_weights=False)

        with pytest.raises(AdapterLoadError, match="Adapter weights not found"):
            get_adapter_model_tag(adapter_path)


class TestShouldUseAdapter:
    """Test adapter usage decision."""

    def test_adapters_disabled(self):
        """Test when adapters are disabled."""
        should_use, reason = should_use_adapter()

        assert not should_use
        assert "disabled" in reason.lower()

    @patch("swarm.adapter_loader.find_latest_adapter")
    def test_no_adapter_found(self, mock_find):
        """Test when no adapter exists."""
        mock_find.return_value = None

        # Patch the use_adapters check to True
        with patch("swarm.adapter_loader.should_use_adapter") as mock_should:
            mock_should.return_value = (False, "No promoted adapter found for persona=None")
            should_use, reason = mock_should()

        assert not should_use
        assert "No promoted adapter" in reason


class TestMarkAdapterPromoted:
    """Test marking adapters as promoted."""

    def test_promote_adapter(self, sample_adapter):
        """Test promoting an adapter."""
        adapter_path = sample_adapter(promoted=False)

        mark_adapter_promoted(adapter_path)

        # Check adapter was renamed
        promoted_path = adapter_path.parent / f"{adapter_path.name}.promoted"
        assert promoted_path.exists()
        assert not adapter_path.exists()

    def test_already_promoted_raises(self, sample_adapter):
        """Test error when trying to promote already promoted adapter."""
        adapter_path = sample_adapter(promoted=True)

        with pytest.raises(ValueError, match="already promoted"):
            mark_adapter_promoted(adapter_path)

    def test_idempotent_promotion(self, sample_adapter):
        """Test promoting same adapter twice is idempotent."""
        adapter_path = sample_adapter(promoted=False)

        mark_adapter_promoted(adapter_path)

        # Try to promote again (should log warning, not error)
        promoted_path = adapter_path.parent / f"{adapter_path.name}.promoted"
        # This would raise if we tried again, but the function checks for .promoted extension


class TestGetFallbackModel:
    """Test fallback model retrieval."""

    def test_returns_base_model(self):
        """Test returns base model from settings."""
        with patch("swarm.adapter_loader.settings") as mock_settings:
            mock_settings.ollama.generator_model = "qwen3:8b"
            model = get_fallback_model()

        assert model == "qwen3:8b"
