"""Tests for configuration module."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from config.settings import (
    OllamaSettings,
    SwarmSettings,
    RewardWeights,
    MarketDataSettings,
    AppSettings,
    validate_ollama_models,
)


class TestOllamaSettings:
    """Test Ollama configuration."""

    def test_default_values(self):
        """Test default Ollama settings."""
        settings = OllamaSettings()
        assert settings.base_url == "http://localhost:11434"
        assert settings.generator_model == "qwen3:8b"
        assert settings.critic_model == "deepseek-r1:14b"
        assert settings.keep_alive == 0
        assert settings.timeout == 300

    def test_keep_alive_validation(self):
        """Test that keep_alive must be 0."""
        with pytest.raises(ValidationError) as exc_info:
            OllamaSettings(keep_alive=60)

        error = exc_info.value.errors()[0]
        assert "keep_alive" in error["loc"]
        assert "must be 0" in error["msg"].lower()

    def test_timeout_bounds(self):
        """Test timeout validation bounds."""
        # Too low
        with pytest.raises(ValidationError):
            OllamaSettings(timeout=10)

        # Too high
        with pytest.raises(ValidationError):
            OllamaSettings(timeout=1000)

        # Valid
        settings = OllamaSettings(timeout=120)
        assert settings.timeout == 120


class TestSwarmSettings:
    """Test swarm configuration."""

    def test_default_values(self):
        """Test default swarm settings."""
        settings = SwarmSettings()
        assert settings.generator_personas == 3
        assert settings.critique_enabled is True
        assert settings.dpo_batch_size == 4
        assert settings.training_enabled is False
        assert settings.concurrency == 2

    def test_persona_bounds(self):
        """Test generator personas validation."""
        # Too low
        with pytest.raises(ValidationError):
            SwarmSettings(generator_personas=0)

        # Too high
        with pytest.raises(ValidationError):
            SwarmSettings(generator_personas=10)

        # Valid
        settings = SwarmSettings(generator_personas=5)
        assert settings.generator_personas == 5


class TestRewardWeights:
    """Test reward weights configuration and validation."""

    def test_default_values_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        weights = RewardWeights()
        total = weights.return_weight + weights.directional_weight + weights.mae_weight
        assert abs(total - 1.0) < 1e-6

    def test_invalid_sum_raises_error(self):
        """Test that weights not summing to 1.0 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            RewardWeights(
                return_weight=0.6,  # This makes total > 1.0
                directional_weight=0.3,
                mae_weight=0.2
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_custom_valid_weights(self):
        """Test custom weights that sum to 1.0."""
        weights = RewardWeights(
            return_weight=0.5,
            directional_weight=0.3,
            mae_weight=0.2
        )
        total = weights.return_weight + weights.directional_weight + weights.mae_weight
        assert abs(total - 1.0) < 1e-6

    def test_negative_weights_rejected(self):
        """Test that negative weights are rejected."""
        with pytest.raises(ValidationError):
            RewardWeights(return_weight=-0.1)


class TestMarketDataSettings:
    """Test market data configuration."""

    def test_default_values(self):
        """Test default market data settings."""
        settings = MarketDataSettings()
        assert settings.exchange == "binance"
        assert "BTC/USDT" in settings.symbols
        assert settings.timeframe == "1h"
        assert settings.lookback_bars == 100

    def test_symbols_parsing_from_string(self):
        """Test parsing comma-separated symbols."""
        settings = MarketDataSettings(symbols="BTC/USDT, ETH/USDT, SOL/USDT")
        assert len(settings.symbols) == 3
        assert "BTC/USDT" in settings.symbols
        assert "SOL/USDT" in settings.symbols

    def test_lookback_bars_bounds(self):
        """Test lookback bars validation."""
        # Too low
        with pytest.raises(ValidationError):
            MarketDataSettings(lookback_bars=10)

        # Too high
        with pytest.raises(ValidationError):
            MarketDataSettings(lookback_bars=2000)

        # Valid
        settings = MarketDataSettings(lookback_bars=200)
        assert settings.lookback_bars == 200


class TestAppSettings:
    """Test main application settings."""

    def test_settings_load_from_env(self, env_vars):
        """Test settings load from environment variables."""
        # Force reload settings with env vars
        settings = AppSettings()

        assert settings.ollama.base_url == "http://localhost:11434"
        assert settings.ollama.generator_model == "qwen3:8b"
        assert settings.swarm.generator_personas == 3
        assert settings.reward.return_weight == 0.50  # Default value from new schema
        assert len(settings.market_data.symbols) == 2

    def test_directories_created(self, tmp_path, monkeypatch):
        """Test that required directories are created."""
        monkeypatch.setenv("MODEL_SAVE_DIR", str(tmp_path / "models"))
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / ".cache"))
        monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "data_cache"))
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "0")

        settings = AppSettings()

        assert settings.model_save_dir.exists()
        assert settings.output_dir.exists()
        assert settings.cache_dir.exists()
        assert settings.market_data.cache_dir.exists()

    def test_nested_settings_validation(self, env_vars):
        """Test that nested settings are properly validated."""
        # This should pass with valid env vars
        settings = AppSettings()

        # Verify nested validation works
        assert settings.ollama.keep_alive == 0
        total_weights = (
            settings.reward.return_weight +
            settings.reward.directional_weight +
            settings.reward.mae_weight
        )
        assert abs(total_weights - 1.0) < 1e-6


@pytest.mark.asyncio
class TestValidateOllamaModels:
    """Test Ollama model validation function."""

    async def test_both_models_available(self, mock_ollama_client, monkeypatch, env_vars):
        """Test successful validation when both models are available."""
        import aiohttp
        monkeypatch.setattr(aiohttp, "ClientSession", lambda: mock_ollama_client)

        result = await validate_ollama_models()

        assert result["error"] is None
        assert "qwen3:8b" in result["available"]
        assert "deepseek-r1:14b" in result["available"]
        assert len(result["missing"]) == 0

    async def test_missing_models(self, monkeypatch, env_vars):
        """Test validation when models are missing."""
        from unittest.mock import AsyncMock, MagicMock

        # Mock empty model list
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"models": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        import aiohttp
        monkeypatch.setattr(aiohttp, "ClientSession", lambda: mock_session)

        result = await validate_ollama_models()

        assert result["error"] is None
        assert len(result["available"]) == 0
        assert "qwen3:8b" in result["missing"]
        assert "deepseek-r1:14b" in result["missing"]

    async def test_ollama_connection_error(self, monkeypatch, env_vars):
        """Test handling of Ollama connection errors."""
        from unittest.mock import AsyncMock, MagicMock
        import aiohttp

        # Mock connection error
        def mock_get(*args, **kwargs):
            raise aiohttp.ClientError("Connection refused")

        mock_session = MagicMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        monkeypatch.setattr(aiohttp, "ClientSession", lambda: mock_session)

        result = await validate_ollama_models()

        assert result["error"] is not None
        assert "Failed to connect" in result["error"]
