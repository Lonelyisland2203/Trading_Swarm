"""Integration tests for dataset generation workflow."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from generate_training_dataset import (
    DatasetConfig,
    phase1_prepare_contexts,
    phase3_postprocess,
    parse_args,
)
from data.prompt_builder import TaskType


def create_ohlcv_dataframe(num_bars: int = 150, base_price: float = 50000.0) -> pd.DataFrame:
    """
    Create a realistic OHLCV DataFrame for testing.

    Args:
        num_bars: Number of bars to generate
        base_price: Starting price level

    Returns:
        DataFrame with timestamp, open, high, low, close, volume columns
    """
    np.random.seed(42)  # Reproducible test data

    # Generate realistic price movements using random walk
    returns = np.random.normal(0, 0.02, num_bars)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices with realistic intrabar movement
    close = prices
    open_prices = np.roll(close, 1)
    open_prices[0] = base_price

    # High/low based on open/close with some noise
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.005, num_bars)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.005, num_bars)))

    # Generate timestamps (1 hour apart, starting from a fixed point)
    base_timestamp_ms = 1704067200000  # 2024-01-01 00:00:00 UTC
    timestamps = base_timestamp_ms + np.arange(num_bars) * 3600000  # 1 hour intervals

    # Generate volume
    volume = np.random.lognormal(10, 1, num_bars)

    return pd.DataFrame({
        "timestamp": timestamps.astype(np.int64),
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    return DatasetConfig(
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        window_count=2,
        window_stride_bars=100,
        lookback_bars=100,
        task_types=[TaskType.PREDICT_DIRECTION, TaskType.ASSESS_MOMENTUM],
        output_dir=tmp_path / "test_dataset",
        resume_from=None,
    )


@pytest.fixture
def sample_examples(tmp_path):
    """Create sample JSONL file with training examples."""
    output_dir = tmp_path / "test_dataset"
    output_dir.mkdir()
    examples_file = output_dir / "examples.jsonl"

    # Create sample examples (3 contexts × 5 personas = 15 examples)
    examples = []
    for ctx_idx in range(3):
        for persona_idx, persona in enumerate(
            ["contrarian", "momentum", "mean_reversion", "breakout", "conservative"]
        ):
            ex = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "timestamp_ms": 1704067200000 + ctx_idx * 3600000,
                "market_regime": "NEUTRAL",
                "persona": persona,
                "context_id": f"context_{ctx_idx}",
                "generator_signal": {
                    "task_type": "predict_direction",
                    "signal_data": {"direction": "HIGHER", "confidence": 0.7},
                    "reasoning": "Test reasoning",
                },
                "was_accepted": persona_idx < 3,  # Accept first 3 personas per context
            }
            examples.append(ex)

    with open(examples_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return output_dir


class TestDatasetConfig:
    """Test dataset configuration parsing."""

    def test_parse_args_quick_test(self, monkeypatch):
        """Test quick-test mode argument parsing."""
        monkeypatch.setattr("sys.argv", ["generate_training_dataset.py", "--quick-test"])

        config = parse_args()

        assert config.symbols == ["BTC/USDT"]
        assert config.timeframes == ["1h"]
        assert config.window_count == 3
        assert config.output_dir == Path("outputs/dataset_test")
        assert config.resume_from is None

    def test_parse_args_custom(self, monkeypatch):
        """Test custom argument parsing."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_training_dataset.py",
                "--symbols",
                "BTC/USDT,ETH/USDT",
                "--timeframes",
                "1h,4h",
                "--windows",
                "5",
                "--output",
                "custom_output",
            ],
        )

        config = parse_args()

        assert config.symbols == ["BTC/USDT", "ETH/USDT"]
        assert config.timeframes == ["1h", "4h"]
        assert config.window_count == 5
        assert config.output_dir == Path("custom_output")

    def test_parse_args_resume(self, monkeypatch):
        """Test resume mode argument parsing."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_training_dataset.py",
                "--resume",
                "outputs/dataset/examples.jsonl",
            ],
        )

        config = parse_args()

        assert config.resume_from == Path("outputs/dataset/examples.jsonl")

    def test_all_task_types_included(self):
        """Test that all three task types are included by default."""
        config = DatasetConfig(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            window_count=1,
            window_stride_bars=100,
            lookback_bars=100,
            task_types=[
                TaskType.PREDICT_DIRECTION,
                TaskType.ASSESS_MOMENTUM,
                TaskType.IDENTIFY_SUPPORT_RESISTANCE,
            ],
            output_dir=Path("test"),
        )

        assert TaskType.PREDICT_DIRECTION in config.task_types
        assert TaskType.ASSESS_MOMENTUM in config.task_types
        assert TaskType.IDENTIFY_SUPPORT_RESISTANCE in config.task_types


@pytest.mark.asyncio
class TestPhase1DataPreparation:
    """Test Phase 1: data preparation."""

    async def test_phase1_creates_jobs_for_all_task_types(self, test_config, monkeypatch):
        """Test Phase 1 creates jobs for all configured task types."""
        # Create real OHLCV DataFrames instead of MagicMock
        ohlcv_df = create_ohlcv_dataframe(num_bars=150)

        # Mock fetch_ohlcv to return real DataFrame
        async def mock_fetch_ohlcv(*args, **kwargs):
            return ohlcv_df.tail(10).copy()

        # Mock fetch_window_data to return real DataFrame with sufficient bars
        async def mock_fetch_window(*args, **kwargs):
            return ohlcv_df.copy()

        with patch("generate_training_dataset.MarketDataService") as MockService, \
             patch("generate_training_dataset.fetch_window_data", side_effect=mock_fetch_window):

            # Configure mock service
            mock_service_instance = MockService.return_value.__aenter__.return_value
            mock_service_instance.fetch_ohlcv = mock_fetch_ohlcv

            jobs = await phase1_prepare_contexts(test_config)

            # Should have: 1 symbol × 1 timeframe × 2 windows × 2 task types = 4 jobs
            assert len(jobs) >= 4, f"Expected at least 4 jobs, got {len(jobs)}"

            # Check task types are present
            task_types_in_jobs = {job.task_type for job in jobs}
            assert TaskType.PREDICT_DIRECTION in task_types_in_jobs
            assert TaskType.ASSESS_MOMENTUM in task_types_in_jobs

    async def test_phase1_skips_insufficient_data_windows(self, test_config, monkeypatch):
        """Test Phase 1 skips windows with insufficient data."""
        # Create real OHLCV DataFrame for successful fetches
        ohlcv_df = create_ohlcv_dataframe(num_bars=150)

        # Mock to return None for some windows (simulating insufficient data)
        call_count = 0

        async def mock_fetch_window(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return None  # Skip every other window
            return ohlcv_df.copy()

        async def mock_fetch_ohlcv(*args, **kwargs):
            return ohlcv_df.tail(10).copy()

        with patch("generate_training_dataset.MarketDataService") as MockService, \
             patch("generate_training_dataset.fetch_window_data", side_effect=mock_fetch_window):

            mock_service_instance = MockService.return_value.__aenter__.return_value
            mock_service_instance.fetch_ohlcv = mock_fetch_ohlcv

            jobs = await phase1_prepare_contexts(test_config)

            # Should have fewer jobs due to skipped windows
            # Expected: 1 symbol × 1 timeframe × 1 good window (out of 2) × 2 tasks = 2 jobs
            assert len(jobs) >= 2, f"Expected at least 2 jobs, got {len(jobs)}"


class TestPhase3PostProcessing:
    """Test Phase 3: post-processing."""

    async def test_phase3_computes_stats(self, sample_examples):
        """Test Phase 3 computes correct statistics."""
        config = DatasetConfig(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            window_count=2,
            window_stride_bars=100,
            lookback_bars=100,
            task_types=[TaskType.PREDICT_DIRECTION],
            output_dir=sample_examples,
        )

        stats = await phase3_postprocess(config)

        assert stats["total_examples"] == 15
        assert stats["unique_contexts"] == 3
        assert "examples_by_regime" in stats
        assert "examples_by_persona" in stats
        assert "acceptance_rate" in stats
        assert stats["acceptance_rate"] == pytest.approx(0.6, abs=0.01)  # 9/15 accepted

    async def test_phase3_detects_incomplete_contexts(self, tmp_path):
        """Test Phase 3 detects contexts with missing personas."""
        output_dir = tmp_path / "incomplete_dataset"
        output_dir.mkdir()
        examples_file = output_dir / "examples.jsonl"

        # Create incomplete context (only 3 personas instead of 5)
        examples = [
            {
                "context_id": "incomplete_context",
                "persona": f"persona_{i}",
                "market_regime": "NEUTRAL",
                "was_accepted": True,
                "generator_signal": {"task_type": "predict_direction"},
            }
            for i in range(3)
        ]

        with open(examples_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        config = DatasetConfig(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            window_count=1,
            window_stride_bars=100,
            lookback_bars=100,
            task_types=[TaskType.PREDICT_DIRECTION],
            output_dir=output_dir,
        )

        stats = await phase3_postprocess(config)

        assert "incomplete_contexts" in stats
        assert stats["incomplete_contexts"] == 1

    async def test_phase3_creates_summary_file(self, sample_examples):
        """Test Phase 3 creates summary.json file."""
        config = DatasetConfig(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            window_count=2,
            window_stride_bars=100,
            lookback_bars=100,
            task_types=[TaskType.PREDICT_DIRECTION],
            output_dir=sample_examples,
        )

        await phase3_postprocess(config)

        summary_file = sample_examples / "summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert "generation_date" in summary
        assert "config" in summary
        assert "total_examples" in summary
        assert summary["config"]["symbols"] == ["BTC/USDT"]

    async def test_phase3_handles_missing_file(self, tmp_path):
        """Test Phase 3 handles missing examples file gracefully."""
        output_dir = tmp_path / "missing_dataset"
        output_dir.mkdir()

        config = DatasetConfig(
            symbols=["BTC/USDT"],
            timeframes=["1h"],
            window_count=1,
            window_stride_bars=100,
            lookback_bars=100,
            task_types=[TaskType.PREDICT_DIRECTION],
            output_dir=output_dir,
        )

        stats = await phase3_postprocess(config)

        assert "error" in stats
        assert stats["error"] == "Examples file not found"
