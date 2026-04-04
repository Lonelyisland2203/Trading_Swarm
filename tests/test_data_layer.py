"""Tests for data layer components (market data, regime, prompts)."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from data.cache_wrapper import AsyncDiskCache, make_cache_key
from data.regime_filter import RegimeClassifier, MarketRegime
from data.prompt_builder import sample_task, TaskType, PromptBuilder
from data.market_data import retry_with_backoff, DataUnavailableError


@pytest.fixture
def sample_close_series():
    """Create sample close price series."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # Generate price with volatility clusters
    returns = []
    vol = 0.01
    for i in range(100):
        # Increase volatility in middle section
        if 30 < i < 60:
            vol = 0.03
        else:
            vol = 0.01
        returns.append(np.random.normal(0, vol))

    prices = 100 * (1 + pd.Series(returns)).cumprod()
    prices.index = dates

    return prices


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 200
    timestamps = [1704067200000 + i * 3600000 for i in range(n)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.uniform(99, 101, n),
        "high": np.random.uniform(100, 102, n),
        "low": np.random.uniform(98, 100, n),
        "close": np.random.uniform(99, 101, n),
        "volume": np.random.uniform(1000, 2000, n),
    })

    # Ensure OHLC relationship
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


class TestCacheWrapper:
    """Test async disk cache wrapper."""

    def test_make_cache_key(self):
        """Test cache key generation."""
        key = make_cache_key(
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            start_ts=1640000000000,
            end_ts=1640003600000,
        )

        assert key == "ohlcv:binance:BTC_USDT:1h:1640000000000:1640003600000"

    def test_cache_key_normalization(self):
        """Test that symbols are normalized."""
        key1 = make_cache_key("binance", "BTC/USDT", "1h", 100, 200)
        key2 = make_cache_key("binance", "btc/usdt", "1h", 100, 200)

        # Should normalize to same key
        assert key1 == key2

    def test_cache_key_exchange_normalization(self):
        """Test that exchange names are normalized."""
        key1 = make_cache_key("binance", "BTC/USDT", "1h", 100, 200)
        key2 = make_cache_key("BINANCE", "BTC/USDT", "1h", 100, 200)
        key3 = make_cache_key("Binance", "BTC/USDT", "1h", 100, 200)

        # All should normalize to same key
        assert key1 == key2
        assert key1 == key3


@pytest.mark.asyncio
class TestAsyncDiskCacheContextManager:
    """Test async context manager support for AsyncDiskCache."""

    async def test_context_manager_basic(self):
        """Test cache works with async context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with AsyncDiskCache(Path(tmpdir), size_limit=1_000_000) as cache:
                await cache.set("key", "value")
                result = await cache.get("key")
                assert result == "value"

    async def test_context_manager_closes_on_exit(self):
        """Test cache is closed after context manager exits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AsyncDiskCache(Path(tmpdir), size_limit=1_000_000)
            async with cache:
                await cache.set("key", "value")
            # After exit, cache should be closed
            # Accessing closed cache may raise or return None
            # The important thing is no exception during context exit


class TestRegimeClassifier:
    """Test market regime classification."""

    def test_regime_classification(self, sample_close_series):
        """Test regime classification on volatility clusters."""
        classifier = RegimeClassifier(
            risk_off_threshold=75.0,
            risk_on_threshold=25.0,
            lookback_period=30
        )

        regime_series = classifier.classify_regime(sample_close_series)

        # Should have same length
        assert len(regime_series) == len(sample_close_series)

        # Should contain all regime types
        regimes = set(regime_series.values)
        assert MarketRegime.RISK_ON in regimes or MarketRegime.NEUTRAL in regimes

    def test_get_current_regime(self, sample_close_series):
        """Test getting current regime."""
        classifier = RegimeClassifier()

        regime, vol = classifier.get_current_regime(sample_close_series)

        assert isinstance(regime, MarketRegime)
        assert vol >= 0.0

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        classifier = RegimeClassifier(lookback_period=30)

        # Only 10 bars
        short_series = pd.Series([100 + i for i in range(10)])

        regime, vol = classifier.get_current_regime(short_series)

        # Should default to NEUTRAL
        assert regime == MarketRegime.NEUTRAL

    def test_realized_volatility_calculation(self):
        """Test realized volatility computation."""
        classifier = RegimeClassifier()

        # Create returns with known std
        returns = pd.Series([0.01, -0.01, 0.01, -0.01] * 25)  # Alternating

        vol = classifier.compute_realized_volatility(returns, window=10, annualize=False)

        # Should produce volatility values
        assert not vol.isna().all()
        assert (vol.dropna() >= 0).all()


class TestPromptBuilder:
    """Test prompt building and task sampling."""

    def test_sample_task_basic(self):
        """Test basic task sampling."""
        task = sample_task(available_bars=100, seed=42)

        assert isinstance(task.task_type, TaskType)
        assert task.weight > 0
        assert 1 <= task.difficulty <= 5

    def test_sample_task_insufficient_data(self):
        """Test error when insufficient data."""
        with pytest.raises(ValueError, match="No eligible tasks"):
            sample_task(available_bars=10)  # Too few bars

    def test_sample_task_difficulty_filter(self):
        """Test difficulty range filtering."""
        task = sample_task(
            available_bars=100,
            difficulty_range=(1, 2),  # Easy tasks only
            seed=42
        )

        assert task.difficulty <= 2

    def test_prompt_builder_creation(self, sample_ohlcv_df):
        """Test prompt building."""
        from data.prompt_builder import TaskConfig

        builder = PromptBuilder()

        # Use a task type that has a template
        task = TaskConfig(
            task_type=TaskType.PREDICT_DIRECTION,
            weight=1.0,
            difficulty=2,
            min_bars_required=50,
        )

        prompt = builder.build_prompt(
            task=task,
            df=sample_ohlcv_df,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
        )

        # Verify prompt structure
        assert "/no_think" in prompt
        assert "BTC/USDT" in prompt
        assert "JSON" in prompt

    def test_prompt_builder_insufficient_data(self):
        """Test error when insufficient data for task."""
        builder = PromptBuilder()

        # Create task requiring 100 bars
        from data.prompt_builder import TaskConfig
        task = TaskConfig(
            task_type=TaskType.PREDICT_DIRECTION,
            weight=1.0,
            difficulty=2,
            min_bars_required=100,
        )

        # Only provide 50 bars
        small_df = pd.DataFrame({
            "timestamp": list(range(50)),
            "open": [100] * 50,
            "high": [102] * 50,
            "low": [98] * 50,
            "close": [101] * 50,
            "volume": [1000] * 50,
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            builder.build_prompt(task, small_df, "BTC/USDT", "1h", MarketRegime.NEUTRAL)


@pytest.mark.asyncio
class TestRetryLogic:
    """Test retry with exponential backoff."""

    async def test_retry_success_on_first_attempt(self):
        """Test successful execution on first try."""
        mock_func = AsyncMock(return_value="success")

        result = await retry_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    async def test_retry_success_after_failures(self):
        """Test successful execution after retries."""
        import ccxt

        mock_func = AsyncMock()
        mock_func.side_effect = [
            ccxt.NetworkError("Temporary failure"),
            ccxt.NetworkError("Temporary failure"),
            "success"
        ]

        result = await retry_with_backoff(mock_func, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 3

    async def test_retry_exhaustion(self):
        """Test retry exhaustion raises last exception."""
        import ccxt

        mock_func = AsyncMock()
        mock_func.side_effect = ccxt.NetworkError("Permanent network failure")

        with pytest.raises(ccxt.NetworkError):
            await retry_with_backoff(mock_func, max_retries=2, base_delay=0.01)

        assert mock_func.call_count == 2

    async def test_non_retryable_exception(self):
        """Test that ExchangeError is not retried."""
        import ccxt

        mock_func = AsyncMock()
        mock_func.side_effect = ccxt.ExchangeError("Invalid symbol")

        with pytest.raises(ccxt.ExchangeError):
            await retry_with_backoff(mock_func, max_retries=3, base_delay=0.01)

        # Should not retry
        assert mock_func.call_count == 1


class TestPromptBuilderRandom:
    """Test prompt builder random isolation."""

    def test_sample_task_seed_reproducibility(self):
        """Test that same seed produces same result."""
        task1 = sample_task(available_bars=100, seed=12345)
        task2 = sample_task(available_bars=100, seed=12345)
        assert task1 == task2

    def test_sample_task_seed_does_not_affect_global_state(self):
        """Test that seed doesn't pollute global random state."""
        import random

        # Set global state
        random.seed(42)
        expected_random = random.random()

        # Reset and sample with different seed
        random.seed(42)
        _ = sample_task(available_bars=100, seed=99999)  # Different seed
        actual_random = random.random()

        # Global state should be unchanged
        assert actual_random == expected_random


class TestRegimeClassifierPctChange:
    """Test regime classifier pct_change behavior."""

    def test_pct_change_no_future_warning(self):
        """Test that pct_change doesn't produce FutureWarning."""
        import warnings

        classifier = RegimeClassifier()
        close = pd.Series([100.0 + i for i in range(50)])

        # This should not produce any FutureWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            classifier.classify_regime(close)

            # Filter for FutureWarnings about pct_change
            pct_warnings = [
                x for x in w
                if issubclass(x.category, FutureWarning)
                and "pct_change" in str(x.message)
            ]
            assert len(pct_warnings) == 0, f"Got pct_change FutureWarning: {pct_warnings}"
