"""Tests for verification engine with mocked market data."""

import pandas as pd
import pytest

from swarm.training_capture import TrainingExample
from verifier.config import BacktestConfig
from verifier.engine import verify_batch, verify_example


class MockMarketData:
    """Mock market data provider for testing."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
    ) -> pd.DataFrame:
        """Return mock data."""
        # Return last N bars
        return self.data.tail(lookback_bars).copy()
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Return bar duration in milliseconds."""
        mapping = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        return mapping.get(timeframe, 3_600_000)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    # 50 hourly bars starting at timestamp 1000
    timestamps = [1000 + i * 3_600_000 for i in range(50)]
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0 + i * 0.5 for i in range(50)],
        "high": [102.0 + i * 0.5 for i in range(50)],
        "low": [98.0 + i * 0.5 for i in range(50)],
        "close": [101.0 + i * 0.5 for i in range(50)],
        "volume": [1000.0] * 50,
    })


@pytest.fixture
def sample_example():
    """Create sample training example."""
    return TrainingExample(
        example_id="test-123",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1000 + 25 * 3_600_000,  # Bar 25
        market_regime="NEUTRAL",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
        },
    )


class TestVerifyExample:
    """Test single example verification."""
    
    @pytest.mark.asyncio
    async def test_verify_successful_example(self, sample_ohlcv, sample_example):
        """Test verifying an example with sufficient data."""
        market_data = MockMarketData(sample_ohlcv)
        
        outcome = await verify_example(
            sample_example,
            market_data,
            config=BacktestConfig(),
        )
        
        assert outcome is not None
        assert outcome.example_id == "test-123"
        assert outcome.actual_direction in ("HIGHER", "LOWER", "FLAT")
        assert isinstance(outcome.realized_return, float)
        assert isinstance(outcome.max_adverse_excursion, float)
        assert outcome.max_adverse_excursion <= 0.0  # MAE is non-positive
        assert outcome.bars_held == 24  # Default horizon for 1h
    
    @pytest.mark.asyncio
    async def test_verify_computes_entry_at_next_open(self, sample_ohlcv, sample_example):
        """Test that entry price is next bar open."""
        market_data = MockMarketData(sample_ohlcv)
        
        outcome = await verify_example(
            sample_example,
            market_data,
            config=BacktestConfig(entry_on="next_open"),
        )
        
        assert outcome is not None
        
        # Entry should be bar 26 open (bar after signal at bar 25)
        expected_entry = 100.0 + 26 * 0.5
        assert outcome.entry_price == pytest.approx(expected_entry, abs=0.01)
    
    @pytest.mark.asyncio
    async def test_verify_uses_timeframe_horizon(self, sample_ohlcv):
        """Test that verification uses correct horizon for timeframe."""
        # For 1h timeframe, horizon is 24 bars
        example = TrainingExample(
            example_id="test-horizon",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp_ms=1000,
            market_regime="NEUTRAL",
            generator_signal={"direction": "HIGHER", "confidence": 0.8},
        )
        
        market_data = MockMarketData(sample_ohlcv)
        outcome = await verify_example(example, market_data)
        
        assert outcome is not None
        assert outcome.bars_held == 24
    
    @pytest.mark.asyncio
    async def test_verify_returns_none_for_insufficient_data(self, sample_example):
        """Test that insufficient forward data returns None."""
        # Create data with only 5 bars (need 24)
        short_data = pd.DataFrame({
            "timestamp": [1000 + i * 3_600_000 for i in range(5)],
            "open": [100.0] * 5,
            "high": [102.0] * 5,
            "low": [98.0] * 5,
            "close": [101.0] * 5,
            "volume": [1000.0] * 5,
        })
        
        market_data = MockMarketData(short_data)
        outcome = await verify_example(sample_example, market_data)
        
        assert outcome is None  # Should return None due to insufficient data
    
    @pytest.mark.asyncio
    async def test_verify_computes_mae_for_higher_signal(self, sample_example):
        """Test that MAE is computed correctly for HIGHER signal."""
        # Create data where price drops before rising
        data = pd.DataFrame({
            "timestamp": [1000 + i * 3_600_000 for i in range(50)],
            # Price pattern: starts at 100, drops to 95, then rises to 110
            "open": [100.0] + [95.0] * 10 + [100.0 + i * 0.5 for i in range(39)],
            "high": [102.0] + [96.0] * 10 + [102.0 + i * 0.5 for i in range(39)],
            "low": [98.0] + [94.0] * 10 + [98.0 + i * 0.5 for i in range(39)],
            "close": [100.0] + [95.0] * 10 + [100.0 + i * 0.5 for i in range(39)],
            "volume": [1000.0] * 50,
        })
        
        market_data = MockMarketData(data)
        outcome = await verify_example(sample_example, market_data)
        
        assert outcome is not None
        # MAE should be negative (price dropped during holding)
        assert outcome.max_adverse_excursion < 0
    
    @pytest.mark.asyncio
    async def test_verify_applies_transaction_costs(self, sample_ohlcv, sample_example):
        """Test that net return accounts for transaction costs."""
        market_data = MockMarketData(sample_ohlcv)
        
        outcome = await verify_example(
            sample_example,
            market_data,
            config=BacktestConfig(txn_cost_pct=0.001),
        )
        
        assert outcome is not None
        # Net return should be less than realized return (costs reduce profit)
        assert outcome.net_return < outcome.realized_return
    
    @pytest.mark.asyncio
    async def test_verify_unknown_timeframe_returns_none(self, sample_ohlcv):
        """Test that unknown timeframe returns None."""
        example = TrainingExample(
            example_id="test",
            symbol="BTC/USDT",
            timeframe="2h",  # Not in HORIZON_BARS
            timestamp_ms=1000,
            market_regime="NEUTRAL",
            generator_signal={"direction": "HIGHER", "confidence": 0.8},
        )
        
        market_data = MockMarketData(sample_ohlcv)
        outcome = await verify_example(example, market_data)
        
        assert outcome is None


class TestVerifyBatch:
    """Test batch verification."""
    
    @pytest.mark.asyncio
    async def test_verify_empty_batch(self):
        """Test verifying empty list returns empty results."""
        market_data = MockMarketData(pd.DataFrame())
        
        outcomes = await verify_batch([], market_data)
        
        assert outcomes == []
    
    @pytest.mark.asyncio
    async def test_verify_batch_groups_by_symbol_timeframe(self, sample_ohlcv):
        """Test that batch processing groups by (symbol, timeframe)."""
        examples = [
            TrainingExample(
                example_id=f"test-{i}",
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000 + i * 3_600_000,
                market_regime="NEUTRAL",
                generator_signal={"direction": "HIGHER", "confidence": 0.8},
            )
            for i in range(5)
        ]
        
        market_data = MockMarketData(sample_ohlcv)
        outcomes = await verify_batch(examples, market_data)
        
        # All should succeed (sufficient data)
        assert len(outcomes) == 5
        assert all(o.example_id.startswith("test-") for o in outcomes)
    
    @pytest.mark.asyncio
    async def test_verify_batch_excludes_failed_examples(self, sample_ohlcv):
        """Test that batch excludes examples that fail verification."""
        examples = [
            # Valid example
            TrainingExample(
                example_id="valid",
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000,
                market_regime="NEUTRAL",
                generator_signal={"direction": "HIGHER", "confidence": 0.8},
            ),
            # Invalid timeframe
            TrainingExample(
                example_id="invalid",
                symbol="BTC/USDT",
                timeframe="2h",  # Unknown
                timestamp_ms=1000,
                market_regime="NEUTRAL",
                generator_signal={"direction": "HIGHER", "confidence": 0.8},
            ),
        ]
        
        market_data = MockMarketData(sample_ohlcv)
        outcomes = await verify_batch(examples, market_data)
        
        # Only the valid example should succeed
        assert len(outcomes) == 1
        assert outcomes[0].example_id == "valid"
    
    @pytest.mark.asyncio
    async def test_verify_batch_with_custom_config(self, sample_ohlcv):
        """Test batch verification with custom config."""
        examples = [
            TrainingExample(
                example_id="test",
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000,
                market_regime="NEUTRAL",
                generator_signal={"direction": "HIGHER", "confidence": 0.8},
            ),
        ]
        
        market_data = MockMarketData(sample_ohlcv)
        
        # Custom high transaction cost
        config = BacktestConfig(txn_cost_pct=0.01)
        outcomes = await verify_batch(examples, market_data, config=config)
        
        assert len(outcomes) == 1
        # High transaction cost should significantly reduce net return
        outcome = outcomes[0]
        assert outcome.realized_return - outcome.net_return > 0.01  # Significant difference
    
    @pytest.mark.asyncio
    async def test_verify_batch_respects_batch_size(self, sample_ohlcv):
        """Test that batch_size parameter is respected."""
        # Create 10 examples
        examples = [
            TrainingExample(
                example_id=f"test-{i}",
                symbol="BTC/USDT",
                timeframe="1h",
                timestamp_ms=1000 + i * 3_600_000,
                market_regime="NEUTRAL",
                generator_signal={"direction": "HIGHER", "confidence": 0.8},
            )
            for i in range(10)
        ]
        
        market_data = MockMarketData(sample_ohlcv)
        
        # Process with small batch size
        outcomes = await verify_batch(examples, market_data, batch_size=3)
        
        # All should still succeed
        assert len(outcomes) == 10
