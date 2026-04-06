"""Tests for fee-aware execution context in prompts."""

import pytest
import pandas as pd
import numpy as np

from config.fee_model import FeeModelSettings
from data.prompt_builder import PromptBuilder, TaskConfig, TaskType
from data.regime_filter import MarketRegime


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 100
    timestamps = [1704067200000 + i * 3600000 for i in range(n)]

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.uniform(99, 101, n),
        "high": np.random.uniform(100, 102, n),
        "low": np.random.uniform(98, 100, n),
        "close": np.random.uniform(99, 101, n),
        "volume": np.random.uniform(1000, 2000, n),
    })


def test_execution_context_futures_mode(sample_df):
    """Verify execution context appears with Futures USDT-M parameters."""
    builder = PromptBuilder()
    fee_model = FeeModelSettings()  # Futures defaults

    task = TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )

    prompt = builder.build_prompt(
        task=task,
        df=sample_df,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model,
    )

    assert "## Execution Context" in prompt
    assert "Exchange: Binance" in prompt
    assert "Mode: Futures USDT-M" in prompt
    assert "Estimated round-trip cost:" in prompt
    assert "Minimum profitable move:" in prompt
    assert "Your prediction must account for these costs" in prompt
