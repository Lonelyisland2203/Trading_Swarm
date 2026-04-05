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
from config.fee_model import FeeModelSettings


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


class TestFeeModelSettings:
    """Test fee model configuration."""

    def test_default_values(self):
        """Test default fee model settings."""
        settings = FeeModelSettings()
        assert settings.maker_fee_pct == 0.02
        assert settings.taker_fee_pct == 0.05
        assert settings.bnb_discount_enabled is True
        assert settings.bnb_discount_pct == 10.0
        assert settings.funding_rate_pct == 0.01
        assert settings.funding_interval_hours == 8
        assert settings.slippage_pct == 0.02

    def test_round_trip_cost_pct_default(self):
        """Test round_trip_cost_pct with default settings."""
        settings = FeeModelSettings()
        # Default: entry=maker 0.02%, exit=taker 0.05%, funding 0.01% per 8h, slippage 0.02%
        # With BNB discount: (0.02 + 0.05) * 0.9 + 0.01 + 0.02 = 0.063 + 0.01 + 0.02 = 0.093%
        cost = settings.round_trip_cost_pct(holding_periods_8h=1)
        assert abs(cost - 0.093) < 0.001

    def test_round_trip_cost_no_bnb_discount(self):
        """Test round_trip with BNB discount disabled."""
        fee_model = FeeModelSettings(bnb_discount_enabled=False)
        cost = fee_model.round_trip_cost_pct(holding_periods_8h=1)
        # (0.02 + 0.05) + 0.01 + 0.02 = 0.10%
        assert abs(cost - 0.10) < 1e-9

    def test_round_trip_cost_multiple_funding_periods(self):
        """Test round_trip with extended holding period."""
        fee_model = FeeModelSettings()
        cost = fee_model.round_trip_cost_pct(holding_periods_8h=3)
        # (0.02 + 0.05) * 0.9 + 0.01*3 + 0.02 = 0.063 + 0.03 + 0.02 = 0.113%
        assert abs(cost - 0.113) < 1e-9

    def test_round_trip_cost_zero_holding(self):
        """Test round_trip with immediate close (no funding)."""
        fee_model = FeeModelSettings()
        cost = fee_model.round_trip_cost_pct(holding_periods_8h=0)
        # (0.02 + 0.05) * 0.9 + 0 + 0.02 = 0.083%
        assert abs(cost - 0.083) < 1e-9

    def test_negative_fees_rejected(self):
        """Test that negative fees raise validation error."""
        with pytest.raises(ValidationError):
            FeeModelSettings(maker_fee_pct=-0.01)

    def test_negative_taker_fees_rejected(self):
        """Test that negative taker fees raise validation error."""
        with pytest.raises(ValidationError):
            FeeModelSettings(taker_fee_pct=-0.01)

    def test_excessive_fees_rejected(self):
        """Test that unrealistic fees are rejected."""
        with pytest.raises(ValidationError):
            FeeModelSettings(taker_fee_pct=1.1)  # > 0.5

    def test_excessive_maker_fees_rejected(self):
        """Test that unrealistic maker fees are rejected."""
        with pytest.raises(ValidationError):
            FeeModelSettings(maker_fee_pct=0.6)  # > 0.5

    def test_funding_rate_bounds(self):
        """Test funding rate validation bounds."""
        # Negative funding rate should fail
        with pytest.raises(ValidationError):
            FeeModelSettings(funding_rate_pct=-0.01)

        # Excessive funding rate should fail
        with pytest.raises(ValidationError):
            FeeModelSettings(funding_rate_pct=1.5)  # > 1.0

        # Valid rate
        fee_model = FeeModelSettings(funding_rate_pct=0.05)
        assert fee_model.funding_rate_pct == 0.05

    def test_slippage_bounds(self):
        """Test slippage validation bounds."""
        # Negative slippage should fail
        with pytest.raises(ValidationError):
            FeeModelSettings(slippage_pct=-0.01)

        # Excessive slippage should fail
        with pytest.raises(ValidationError):
            FeeModelSettings(slippage_pct=1.5)  # > 1.0

        # Valid slippage
        fee_model = FeeModelSettings(slippage_pct=0.05)
        assert fee_model.slippage_pct == 0.05

    def test_bnb_discount_bounds(self):
        """Test BNB discount percentage validation."""
        # Negative discount should fail
        with pytest.raises(ValidationError):
            FeeModelSettings(bnb_discount_pct=-10.0)

        # Excessive discount should fail
        with pytest.raises(ValidationError):
            FeeModelSettings(bnb_discount_pct=150.0)  # > 100.0

        # Valid discount
        fee_model = FeeModelSettings(bnb_discount_pct=15.0)
        assert fee_model.bnb_discount_pct == 15.0

    def test_funding_interval_bounds(self):
        """Test funding interval hours validation."""
        # Too low
        with pytest.raises(ValidationError):
            FeeModelSettings(funding_interval_hours=0)

        # Too high
        with pytest.raises(ValidationError):
            FeeModelSettings(funding_interval_hours=25)

        # Valid values
        fee_model = FeeModelSettings(funding_interval_hours=12)
        assert fee_model.funding_interval_hours == 12

    def test_net_return_positive_gross(self):
        """Test net_return with gross return exceeding fees."""
        fee_model = FeeModelSettings()
        # Gross +0.15%, fees ~0.083% → net should be positive
        net = fee_model.net_return(gross_return_pct=0.15, holding_periods_8h=0)
        assert net > 0
        assert abs(net - 0.067) < 1e-9  # 0.15 - 0.083 = 0.067%

    def test_minimum_profitable_return_pct_equals_cost(self):
        """Test minimum_profitable_return_pct equals round_trip_cost."""
        fee_model = FeeModelSettings()
        min_return = fee_model.minimum_profitable_return_pct(holding_periods_8h=3)
        expected_cost = fee_model.round_trip_cost_pct(holding_periods_8h=3)
        assert abs(min_return - expected_cost) < 1e-9

    def test_round_trip_maker_entry_maker_exit(self):
        """Test round_trip with both entry and exit as limit orders."""
        fee_model = FeeModelSettings(
            entry_order_type="maker",
            exit_order_type="maker",
            bnb_discount=True,
        )
        cost = fee_model.round_trip_cost_pct(holding_periods_8h=0)
        # Entry: 0.02 * 0.9 = 0.018%, Exit: 0.02 * 0.9 = 0.018%
        # Slippage: 0.02%, Funding: 0%
        # Total: 0.036 + 0.02 = 0.056%
        assert abs(cost - 0.056) < 1e-9

    def test_round_trip_taker_entry_taker_exit(self):
        """Test round_trip with both entry and exit as market orders."""
        fee_model = FeeModelSettings(
            entry_order_type="taker",
            exit_order_type="taker",
            bnb_discount=True,
        )
        cost = fee_model.round_trip_cost_pct(holding_periods_8h=0)
        # Entry: 0.05 * 0.9 = 0.045%, Exit: 0.05 * 0.9 = 0.045%
        # Slippage: 0.02%, Funding: 0%
        # Total: 0.09 + 0.02 = 0.11%
        assert abs(cost - 0.11) < 1e-9

    def test_round_trip_taker_entry_maker_exit(self):
        """Test round_trip with market entry and limit exit."""
        fee_model = FeeModelSettings(
            entry_order_type="taker",
            exit_order_type="maker",
            bnb_discount=True,
        )
        cost = fee_model.round_trip_cost_pct(holding_periods_8h=0)
        # Entry: 0.05 * 0.9 = 0.045%, Exit: 0.02 * 0.9 = 0.018%
        # Slippage: 0.02%, Funding: 0%
        # Total: 0.063 + 0.02 = 0.083%
        assert abs(cost - 0.083) < 1e-9

    def test_funding_cost_variations(self):
        """Test funding costs with different holding periods."""
        fee_model = FeeModelSettings()

        # 0.5 periods
        cost_half = fee_model.round_trip_cost_pct(holding_periods_8h=0.5)
        expected_half = 0.083 + (0.01 * 0.5)  # Base + half funding
        assert abs(cost_half - expected_half) < 1e-9

        # 2.5 periods
        cost_two_half = fee_model.round_trip_cost_pct(holding_periods_8h=2.5)
        expected_two_half = 0.083 + (0.01 * 2.5)  # Base + 2.5x funding
        assert abs(cost_two_half - expected_two_half) < 1e-9

        # 10 periods (long hold)
        cost_ten = fee_model.round_trip_cost_pct(holding_periods_8h=10)
        expected_ten = 0.083 + (0.01 * 10)  # Base + 10x funding
        assert abs(cost_ten - expected_ten) < 1e-9

    def test_funding_disabled(self):
        """Test round_trip with funding cost disabled."""
        fee_model = FeeModelSettings(include_funding=False)

        # Should get same cost regardless of holding period
        cost_0 = fee_model.round_trip_cost_pct(holding_periods_8h=0)
        cost_5 = fee_model.round_trip_cost_pct(holding_periods_8h=5)

        assert abs(cost_0 - 0.083) < 1e-9  # Only base fees + slippage
        assert abs(cost_5 - 0.083) < 1e-9  # Same even with 5 periods
        assert abs(cost_0 - cost_5) < 1e-9  # Both equal

    def test_invalid_order_type_rejected(self):
        """Test that invalid order types raise validation error."""
        with pytest.raises(ValidationError):
            FeeModelSettings(entry_order_type="invalid")
        with pytest.raises(ValidationError):
            FeeModelSettings(exit_order_type="MAKER")  # case-sensitive

    def test_negative_holding_periods_rejected(self):
        """Test that negative holding periods raise ValueError."""
        fee_model = FeeModelSettings()
        with pytest.raises(ValueError, match="holding_periods_8h must be non-negative"):
            fee_model.round_trip_cost_pct(holding_periods_8h=-1.0)
