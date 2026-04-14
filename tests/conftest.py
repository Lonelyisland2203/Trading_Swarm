"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response for model listing."""
    return {
        "models": [
            {
                "name": "qwen3:8b",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 4800000000,
            },
            {
                "name": "deepseek-r1:14b",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 8200000000,
            },
        ]
    }


@pytest.fixture
def mock_ollama_client(mock_ollama_response):
    """Mock aiohttp client session for Ollama API calls."""
    from unittest.mock import MagicMock

    # Create a proper async context manager for the response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=mock_ollama_response)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    # Create session that returns the context manager response
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    return mock_session


@pytest.fixture
def env_vars(monkeypatch, tmp_path):
    """Set up test environment variables."""
    test_env = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_GENERATOR_MODEL": "qwen3:8b",
        "OLLAMA_CRITIC_MODEL": "deepseek-r1:14b",
        "OLLAMA_KEEP_ALIVE": "0",
        "OLLAMA_TIMEOUT": "300",
        "GENERATOR_PERSONAS": "3",
        "CRITIQUE_ENABLED": "true",
        "DPO_BATCH_SIZE": "4",
        "TRAINING_ENABLED": "false",
        "CONCURRENCY": "2",
        "REWARD_RETURN_WEIGHT": "0.50",
        "REWARD_DIRECTIONAL_WEIGHT": "0.30",
        "REWARD_MAE_WEIGHT": "0.20",
        "EXCHANGE": "binance",
        "SYMBOLS": "BTC/USDT,ETH/USDT",
        "TIMEFRAME": "1h",
        "LOOKBACK_BARS": "100",
        "DATA_CACHE_DIR": str(tmp_path / "cache"),
        "DATA_CACHE_SIZE_LIMIT": "1073741824",
        "MODEL_SAVE_DIR": str(tmp_path / "models"),
        "OUTPUT_DIR": str(tmp_path / "outputs"),
        "CACHE_DIR": str(tmp_path / ".cache"),
        "LOG_LEVEL": "INFO",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    return test_env


@pytest.fixture
def mock_market_data_service():
    """Factory for mock market data service."""

    def _create_mock(return_df):
        mock_service = MagicMock()
        mock_service.get_ohlcv_as_of = AsyncMock(return_value=return_df)
        return mock_service

    return _create_mock


@pytest.fixture
def mock_market_data_service_exception():
    """Factory for mock market data service that raises exception."""

    def _create_mock(exception):
        mock_service = MagicMock()
        mock_service.get_ohlcv_as_of = AsyncMock(side_effect=exception)
        return mock_service

    return _create_mock
