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

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.random.uniform(99, 101, n),
            "high": np.random.uniform(100, 102, n),
            "low": np.random.uniform(98, 100, n),
            "close": np.random.uniform(99, 101, n),
            "volume": np.random.uniform(1000, 2000, n),
        }
    )


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


def test_execution_context_in_all_templates(sample_df):
    """Verify execution context appears in all three task templates."""
    builder = PromptBuilder()
    fee_model = FeeModelSettings()

    task_types = [
        TaskType.PREDICT_DIRECTION,
        TaskType.ASSESS_MOMENTUM,
        TaskType.IDENTIFY_SUPPORT_RESISTANCE,
    ]

    for task_type in task_types:
        task = TaskConfig(
            task_type=task_type,
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

        assert "## Execution Context" in prompt, f"Missing in {task_type.value}"
        assert "Mode: Futures USDT-M" in prompt, f"Missing in {task_type.value}"


def test_execution_context_spot_mode(sample_df):
    """Test spot mode with include_funding=False shows 'Mode: Spot'."""
    builder = PromptBuilder()
    # Spot mode: no funding costs
    fee_model = FeeModelSettings(include_funding=False)

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

    # Verify spot mode appears instead of Futures
    assert "## Execution Context" in prompt
    assert "Mode: Spot" in prompt
    assert "Mode: Futures" not in prompt
    assert "Estimated round-trip cost:" in prompt
    assert "Minimum profitable move:" in prompt


def test_execution_context_none_fee_model(sample_df):
    """Test that no execution context appears when fee_model=None."""
    builder = PromptBuilder()

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
        fee_model=None,  # No fee model = no execution context
    )

    # Verify no execution context section
    assert "## Execution Context" not in prompt
    assert "Mode: Futures USDT-M" not in prompt
    assert "Mode: Spot" not in prompt
    assert "Minimum profitable move:" not in prompt


def test_execution_context_dynamic_costs_by_timeframe(sample_df):
    """Test that execution costs vary by timeframe due to different holding periods."""
    builder = PromptBuilder()
    fee_model = FeeModelSettings()  # Include funding costs

    timeframes = ["1h", "4h", "1d"]
    costs = []

    task = TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )

    for timeframe in timeframes:
        prompt = builder.build_prompt(
            task=task,
            df=sample_df,
            symbol="BTC/USDT",
            timeframe=timeframe,
            market_regime=MarketRegime.NEUTRAL,
            fee_model=fee_model,
        )

        # Extract estimated cost from prompt
        assert "Estimated round-trip cost:" in prompt
        lines = prompt.split("\n")
        cost_line = [l for l in lines if "Estimated round-trip cost:" in l][0]
        # Parse cost: "Estimated round-trip cost: X.XXX%"
        cost_str = cost_line.split(": ")[1].rstrip("%")
        cost_value = float(cost_str)
        costs.append((timeframe, cost_value))

    # Verify costs increase with longer holding periods (more funding)
    # 1h (24 bars) < 4h (12 bars) < 1d (5 bars) in terms of funding periods
    # 1h: 24 * 1/8 = 3 periods
    # 4h: 12 * 4/8 = 6 periods
    # 1d: 5 * 24/8 = 15 periods
    assert costs[0][1] < costs[1][1] < costs[2][1], (
        f"Costs should increase with holding period: {costs}"
    )
