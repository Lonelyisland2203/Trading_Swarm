"""Tests for extended prompt builder with new task types."""

import pytest
import pandas as pd
import numpy as np

from data.prompt_builder import (
    TaskType,
    TaskConfig,
    PromptBuilder,
    MomentumAssessmentPrompt,
    SupportResistancePrompt,
    detect_swing_highs,
    detect_swing_lows,
    calculate_bb_width,
    get_bb_trend,
    sample_task,
    TASK_CONFIGS,
)
from data.regime_filter import MarketRegime


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    timestamps = [1704067200000 + i * 3_600_000 for i in range(100)]

    # Create price data with some volatility
    base_price = 50000.0
    prices = []
    for i in range(100):
        noise = np.random.normal(0, 100)
        trend = i * 10
        prices.append(base_price + trend + noise)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 50)) for p in prices],
        "low": [p - abs(np.random.normal(0, 50)) for p in prices],
        "close": [p + np.random.normal(0, 20) for p in prices],
        "volume": [100.0] * 100,
    })

    return df


class TestTaskSampling:
    """Test task sampling with new task types."""

    def test_all_three_tasks_enabled(self):
        """Test that all three task types are enabled in TASK_CONFIGS."""
        task_types = {config.task_type for config in TASK_CONFIGS}

        assert TaskType.PREDICT_DIRECTION in task_types
        assert TaskType.ASSESS_MOMENTUM in task_types
        assert TaskType.IDENTIFY_SUPPORT_RESISTANCE in task_types

    def test_sample_task_includes_new_types(self):
        """Test that new task types can be sampled."""
        # Sample many times to ensure all types can be selected
        sampled_types = set()

        for _ in range(100):
            task = sample_task(available_bars=100, seed=None)
            sampled_types.add(task.task_type)

        # Should have sampled at least 2 of the 3 task types
        assert len(sampled_types) >= 2

    def test_momentum_task_min_bars(self):
        """Test ASSESS_MOMENTUM has lower bar requirement."""
        momentum_config = next(
            c for c in TASK_CONFIGS if c.task_type == TaskType.ASSESS_MOMENTUM
        )

        assert momentum_config.min_bars_required == 30

    def test_support_resistance_task_min_bars(self):
        """Test IDENTIFY_SUPPORT_RESISTANCE requires more bars."""
        sr_config = next(
            c for c in TASK_CONFIGS
            if c.task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE
        )

        assert sr_config.min_bars_required == 100


class TestSwingDetection:
    """Test swing high/low detection."""

    def test_detect_swing_highs(self, sample_df):
        """Test swing high detection."""
        swing_highs = detect_swing_highs(sample_df, window=5, num_swings=3)

        assert isinstance(swing_highs, list)
        assert len(swing_highs) <= 3
        # All swing highs should be positive
        assert all(h > 0 for h in swing_highs)

    def test_detect_swing_lows(self, sample_df):
        """Test swing low detection."""
        swing_lows = detect_swing_lows(sample_df, window=5, num_swings=3)

        assert isinstance(swing_lows, list)
        assert len(swing_lows) <= 3
        assert all(l > 0 for l in swing_lows)

    def test_swing_detection_insufficient_data(self):
        """Test swing detection with insufficient data."""
        df = pd.DataFrame({
            "high": [100, 101, 102],
            "low": [99, 100, 101],
        })

        swing_highs = detect_swing_highs(df, window=5)
        swing_lows = detect_swing_lows(df, window=5)

        assert swing_highs == []
        assert swing_lows == []

    def test_swing_highs_are_local_maxima(self):
        """Test that detected swing highs are actual local maxima."""
        # Create data with clear swing high
        highs = [100, 105, 110, 115, 120, 115, 110, 105, 100, 95]
        df = pd.DataFrame({"high": highs, "low": [h - 5 for h in highs]})

        swing_highs = detect_swing_highs(df, window=2, num_swings=1)

        # Should detect the peak at 120
        assert len(swing_highs) == 1
        assert swing_highs[0] == 120.0

    def test_swing_lows_are_local_minima(self):
        """Test that detected swing lows are actual local minima."""
        # Create data with clear swing low
        lows = [100, 95, 90, 85, 80, 85, 90, 95, 100, 105]
        df = pd.DataFrame({"high": [l + 5 for l in lows], "low": lows})

        swing_lows = detect_swing_lows(df, window=2, num_swings=1)

        # Should detect the valley at 80
        assert len(swing_lows) == 1
        assert swing_lows[0] == 80.0


class TestBollingerBandWidth:
    """Test Bollinger Band width calculation."""

    def test_calculate_bb_width(self, sample_df):
        """Test BB width calculation."""
        bb_width = calculate_bb_width(sample_df, period=20)

        assert isinstance(bb_width, float)
        assert bb_width >= 0.0
        # Width should be reasonable (typically < 0.1 for crypto)
        assert bb_width < 1.0

    def test_bb_width_insufficient_data(self):
        """Test BB width with insufficient data."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        bb_width = calculate_bb_width(df, period=20)
        assert bb_width == 0.0

    def test_get_bb_trend_expanding(self):
        """Test BB trend detection - expanding."""
        trend = get_bb_trend(current_width=0.05, prev_width=0.03)
        assert trend == "expanding"

    def test_get_bb_trend_contracting(self):
        """Test BB trend detection - contracting."""
        trend = get_bb_trend(current_width=0.03, prev_width=0.05)
        assert trend == "contracting"

    def test_get_bb_trend_stable(self):
        """Test BB trend detection - stable."""
        trend = get_bb_trend(current_width=0.05, prev_width=0.0505)
        assert trend == "stable"


class TestMomentumPromptTemplate:
    """Test momentum assessment prompt template."""

    def test_momentum_prompt_renders(self):
        """Test momentum prompt template renders correctly."""
        template = MomentumAssessmentPrompt()

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="NEUTRAL",
            rsi=65.0,
            rsi_prev=60.0,
            rsi_delta=5.0,
            macd=100.0,
            macd_signal=95.0,
            macd_prev=90.0,
            macd_delta=10.0,
            bb_width=0.05,
            bb_trend="expanding",
            price_summary="Recent price action...",
        )

        assert "BTC/USDT" in prompt
        assert "momentum" in prompt.lower()
        assert "INCREASING" in prompt
        assert "DECREASING" in prompt
        assert "/no_think" in prompt

    def test_momentum_prompt_includes_indicators(self):
        """Test momentum prompt includes all indicators."""
        template = MomentumAssessmentPrompt()

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="NEUTRAL",
            rsi=65.0,
            rsi_prev=60.0,
            rsi_delta=5.0,
            macd=100.0,
            macd_signal=95.0,
            macd_prev=90.0,
            macd_delta=10.0,
            bb_width=0.05,
            bb_trend="expanding",
            price_summary="Recent price action...",
        )

        assert "RSI(14): 65.00" in prompt
        assert "previous: 60.00" in prompt
        assert "change: +5.00" in prompt
        assert "BB Width: 0.05" in prompt
        assert "expanding" in prompt


class TestSupportResistancePromptTemplate:
    """Test support/resistance prompt template."""

    def test_support_resistance_prompt_renders(self):
        """Test support/resistance prompt template renders correctly."""
        template = SupportResistancePrompt()

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="NEUTRAL",
            price_high=51000.0,
            price_low=49000.0,
            price_range=2000.0,
            swing_highs="$50800.00, $50600.00, $50400.00",
            swing_lows="$49200.00, $49100.00, $49000.00",
            price_summary="Recent price action...",
        )

        assert "BTC/USDT" in prompt
        assert "support" in prompt.lower()
        assert "resistance" in prompt.lower()
        assert "support_price" in prompt
        assert "resistance_price" in prompt
        assert "/no_think" in prompt

    def test_support_resistance_prompt_includes_swings(self):
        """Test support/resistance prompt includes swing points."""
        template = SupportResistancePrompt()

        prompt = template.render(
            symbol="BTC/USDT",
            timeframe="1h",
            current_price=50000.0,
            market_regime="NEUTRAL",
            price_high=51000.0,
            price_low=49000.0,
            price_range=2000.0,
            swing_highs="$50800.00, $50600.00",
            swing_lows="$49200.00, $49100.00",
            price_summary="Recent price action...",
        )

        assert "$50800.00" in prompt
        assert "$49200.00" in prompt
        assert "Swing Highs:" in prompt
        assert "Swing Lows:" in prompt


class TestPromptBuilderExtended:
    """Test PromptBuilder with new task types."""

    def test_build_momentum_prompt(self, sample_df):
        """Test building momentum assessment prompt."""
        builder = PromptBuilder()

        task = TaskConfig(
            TaskType.ASSESS_MOMENTUM,
            weight=1.0,
            difficulty=2,
            min_bars_required=30,
        )

        prompt = builder.build_prompt(
            task=task,
            df=sample_df,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
        )

        assert "BTC/USDT" in prompt
        assert "momentum" in prompt.lower()
        assert "RSI" in prompt
        assert "MACD" in prompt
        assert "BB Width" in prompt

    def test_build_support_resistance_prompt(self, sample_df):
        """Test building support/resistance prompt."""
        builder = PromptBuilder()

        task = TaskConfig(
            TaskType.IDENTIFY_SUPPORT_RESISTANCE,
            weight=1.0,
            difficulty=3,
            min_bars_required=100,
        )

        prompt = builder.build_prompt(
            task=task,
            df=sample_df,
            symbol="BTC/USDT",
            timeframe="1h",
            market_regime=MarketRegime.NEUTRAL,
        )

        assert "BTC/USDT" in prompt
        assert "support" in prompt.lower()
        assert "resistance" in prompt.lower()
        assert "Swing Highs:" in prompt
        assert "Swing Lows:" in prompt

    def test_all_templates_registered(self):
        """Test all three task types have templates registered."""
        builder = PromptBuilder()

        assert TaskType.PREDICT_DIRECTION in builder.templates
        assert TaskType.ASSESS_MOMENTUM in builder.templates
        assert TaskType.IDENTIFY_SUPPORT_RESISTANCE in builder.templates

    def test_unsupported_task_type_raises(self):
        """Test unsupported task type raises ValueError."""
        # Create df with enough data to pass min_bars check
        df = pd.DataFrame({
            "timestamp": range(250),
            "open": [50000.0] * 250,
            "high": [50100.0] * 250,
            "low": [49900.0] * 250,
            "close": [50000.0] * 250,
            "volume": [100.0] * 250,
        })

        builder = PromptBuilder()

        task = TaskConfig(
            TaskType.IDENTIFY_PATTERN,  # Not implemented
            weight=1.0,
            difficulty=5,
            min_bars_required=200,
        )

        with pytest.raises(ValueError, match="No template for task type"):
            builder.build_prompt(
                task=task,
                df=df,
                symbol="BTC/USDT",
                timeframe="1h",
                market_regime=MarketRegime.NEUTRAL,
            )
