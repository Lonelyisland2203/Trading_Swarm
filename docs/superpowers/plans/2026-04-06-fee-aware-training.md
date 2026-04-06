# Fee-Aware DPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate realistic fee model into DPO training pipeline to teach LLM fee-aware signal generation

**Architecture:** Teaching + enforcement pattern: (1) Add execution context to prompts showing fee structure, (2) Switch reward engine to use net returns instead of gross, (3) Add fee flip diagnostic to detect profitability threshold issues

**Tech Stack:** Python 3.13, FeeModelSettings (Binance Futures USDT-M), PromptBuilder, RewardEngine, DPO training pipeline

---

## File Structure

**Files to Modify:**
- `data/prompt_builder.py` - Add `_build_execution_context()` helper and `fee_model` parameter to `build_prompt()`
- `training/reward_engine.py` - Switch from `realized_return` to `net_return`
- `generate_training_dataset.py` - Add `--fee-mode` flag and `create_fee_model()` helper
- `run_dpo_training.py` - Add `--fee-mode` flag and `compute_fee_flip_diagnostic()` function

**Files to Create:**
- `tests/test_prompt_builder_fee_context.py` - Test execution context rendering
- `tests/test_reward_net_returns.py` - Test net return reward computation
- `tests/test_dpo_export_net_ranking.py` - Test preference pair ranking by net rewards
- `tests/test_fee_flip_diagnostic.py` - Test fee flip diagnostic logic

---

## Task 1: PromptBuilder - Add Execution Context Helper

**Files:**
- Modify: `data/prompt_builder.py:784-1020`
- Test: `tests/test_prompt_builder_fee_context.py` (create)

- [ ] **Step 1: Write failing test for execution context rendering**

Create `tests/test_prompt_builder_fee_context.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_fee_context.py::test_execution_context_futures_mode -xvs`
Expected: FAIL with "TypeError: build_prompt() got an unexpected keyword argument 'fee_model'"

- [ ] **Step 3: Add fee_model parameter to build_prompt() signature**

In `data/prompt_builder.py`, modify the `build_prompt()` method (around line 827):

```python
def build_prompt(
    self,
    task: TaskConfig,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    market_regime: MarketRegime,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
    fee_model: "FeeModelSettings | None" = None,  # NEW
) -> str:
```

Add import at top of file (around line 17):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.fee_model import FeeModelSettings
```

- [ ] **Step 4: Add _build_execution_context() helper method**

In `data/prompt_builder.py`, add new method to PromptBuilder class (around line 825, before `build_prompt()`):

```python
def _build_execution_context(
    self,
    timeframe: str,
    horizon_bars: int,
    fee_model: "FeeModelSettings",
) -> str:
    """
    Build execution context section for prompts.

    Args:
        timeframe: Trading timeframe (e.g., "1h", "1d")
        horizon_bars: Prediction horizon in bars
        fee_model: Fee model settings

    Returns:
        Formatted execution context section
    """
    from verifier.constants import compute_holding_periods_8h

    holding_periods = compute_holding_periods_8h(timeframe, horizon_bars)
    round_trip_cost = fee_model.round_trip_cost_pct(holding_periods)
    min_profitable = fee_model.minimum_profitable_return_pct(holding_periods)

    # Determine mode from fee model settings
    if fee_model.include_funding:
        mode = "Futures USDT-M"
    else:
        mode = "Spot"

    return f"""## Execution Context
Exchange: Binance
Mode: {mode}
Estimated round-trip cost: {round_trip_cost:.3f}%
Minimum profitable move: {min_profitable:.3f}%

Your prediction must account for these costs. Signals with expected moves smaller than the minimum profitable threshold should be rated LOW CONFIDENCE regardless of directional conviction."""
```

- [ ] **Step 5: Integrate execution context into build_prompt() logic**

In `data/prompt_builder.py`, in `build_prompt()` method, add logic after higher_tf_context section (around line 913):

```python
# Build execution context section
execution_context = ""
if fee_model is not None:
    from verifier.constants import get_horizon_bars
    try:
        horizon_bars = get_horizon_bars(timeframe)
    except KeyError:
        logger.warning("Unknown timeframe for horizon", timeframe=timeframe)
        horizon_bars = 5  # Default fallback

    execution_context = "\n" + self._build_execution_context(
        timeframe, horizon_bars, fee_model
    ) + "\n"

logger.debug(
    "Execution context built",
    timeframe=timeframe,
    has_execution_context=bool(execution_context),
)
```

- [ ] **Step 6: Update DirectionPredictionPrompt template**

In `data/prompt_builder.py`, modify DirectionPredictionPrompt.TEMPLATE (around line 417):

```python
TEMPLATE = """/no_think

You are a quantitative trading analyst. Analyze the following market data and predict the price direction for the next {horizon} bars.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}{execution_context}
## Technical Indicators
RSI(14): {rsi:.2f}
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
BB Position: {bb_position:.2f} (0.0=lower band, 0.5=middle, 1.0=upper band)

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Predict whether the price will be HIGHER or LOWER {horizon} bars from now.

Respond in JSON format:
{{"direction": "HIGHER" | "LOWER", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
```

- [ ] **Step 7: Update DirectionPredictionPrompt.render() method**

In `data/prompt_builder.py`, modify DirectionPredictionPrompt.render() (around line 442):

```python
def render(
    self,
    symbol: str,
    timeframe: str,
    current_price: float,
    market_regime: str,
    rsi: float,
    macd: float,
    macd_signal: float,
    bb_position: float,
    price_summary: str,
    horizon: int = 5,
    num_bars: int = 10,
    higher_tf_context: str | None = None,
    execution_context: str = "",  # NEW
) -> str:
    """Render direction prediction prompt."""
    # Build higher timeframe section
    higher_tf_section = ""
    if higher_tf_context:
        higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

    return self.TEMPLATE.format(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        market_regime=market_regime,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        bb_position=bb_position,
        price_summary=price_summary,
        horizon=horizon,
        num_bars=num_bars,
        higher_tf_section=higher_tf_section,
        execution_context=execution_context,  # NEW
    )
```

- [ ] **Step 8: Update prompt rendering logic in build_prompt()**

In `data/prompt_builder.py`, modify PREDICT_DIRECTION rendering (around line 932):

```python
if task.task_type == TaskType.PREDICT_DIRECTION:
    prompt = template.render(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        market_regime=market_regime.value,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        bb_position=bb_pos,
        price_summary=price_summary,
        higher_tf_context=higher_tf_context,
        execution_context=execution_context,  # NEW
    )
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_fee_context.py::test_execution_context_futures_mode -xvs`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_fee_context.py
git commit -m "feat: add execution context to DirectionPredictionPrompt

- Add fee_model parameter to build_prompt()
- Add _build_execution_context() helper method
- Update DirectionPredictionPrompt template and render()
- Add test for futures mode execution context"
```

---

## Task 2: PromptBuilder - Add Execution Context to Remaining Templates

**Files:**
- Modify: `data/prompt_builder.py:479-1006`
- Test: `tests/test_prompt_builder_fee_context.py`

- [ ] **Step 1: Write test for execution context in all templates**

Add to `tests/test_prompt_builder_fee_context.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder_fee_context.py::test_execution_context_in_all_templates -xvs`
Expected: FAIL - "Missing in assess_momentum" or "Missing in identify_support_resistance"

- [ ] **Step 3: Update MomentumAssessmentPrompt template**

In `data/prompt_builder.py`, modify MomentumAssessmentPrompt.TEMPLATE (around line 483):

```python
TEMPLATE = """/no_think

You are a momentum analyst. Analyze the following market data and assess whether momentum is increasing or decreasing.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}{execution_context}
## Technical Indicators
RSI(14): {rsi:.2f} (previous: {rsi_prev:.2f}, change: {rsi_delta:+.2f})
MACD: {macd:.4f} (Signal: {macd_signal:.4f})
MACD Previous: {macd_prev:.4f} (change: {macd_delta:+.4f})
BB Width: {bb_width:.4f} ({bb_trend})

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Is momentum INCREASING or DECREASING? Consider:
- RSI trend (rising RSI = increasing momentum)
- MACD vs signal line (diverging upward = increasing momentum)
- Price action acceleration (larger price changes = increasing momentum)
- BB width (expanding = increasing volatility/momentum)

Respond in JSON format:
{{"direction": "INCREASING" | "DECREASING", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
```

- [ ] **Step 4: Update MomentumAssessmentPrompt.render() method**

In `data/prompt_builder.py`, modify MomentumAssessmentPrompt.render() (around line 513):

```python
def render(
    self,
    symbol: str,
    timeframe: str,
    current_price: float,
    market_regime: str,
    rsi: float,
    rsi_prev: float,
    rsi_delta: float,
    macd: float,
    macd_signal: float,
    macd_prev: float,
    macd_delta: float,
    bb_width: float,
    bb_trend: str,
    price_summary: str,
    num_bars: int = 10,
    higher_tf_context: str | None = None,
    execution_context: str = "",  # NEW
) -> str:
    """Render momentum assessment prompt."""
    # Build higher timeframe section
    higher_tf_section = ""
    if higher_tf_context:
        higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

    return self.TEMPLATE.format(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        market_regime=market_regime,
        rsi=rsi,
        rsi_prev=rsi_prev,
        rsi_delta=rsi_delta,
        macd=macd,
        macd_signal=macd_signal,
        macd_prev=macd_prev,
        macd_delta=macd_delta,
        bb_width=bb_width,
        bb_trend=bb_trend,
        price_summary=price_summary,
        num_bars=num_bars,
        higher_tf_section=higher_tf_section,
        execution_context=execution_context,  # NEW
    )
```

- [ ] **Step 5: Update SupportResistancePrompt template**

In `data/prompt_builder.py`, modify SupportResistancePrompt.TEMPLATE (around line 562):

```python
TEMPLATE = """/no_think

You are a technical analyst. Identify the nearest support and resistance levels based on recent price action.

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}{execution_context}
## Price History (last 50 bars)
High: ${price_high:.4f}
Low: ${price_low:.4f}
Range: ${price_range:.4f}

## Recent Swing Points
Swing Highs: {swing_highs}
Swing Lows: {swing_lows}

## Recent Price Action (last {num_bars} bars)
{price_summary}

## Task
Identify:
1. Nearest SUPPORT level below current price
2. Nearest RESISTANCE level above current price

Consider: Previous swing highs/lows, psychological levels (round numbers), high-volume areas

Respond in JSON format:
{{
  "support_price": float,
  "support_confidence": 0.0-1.0,
  "resistance_price": float,
  "resistance_confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""
```

- [ ] **Step 6: Update SupportResistancePrompt.render() method**

In `data/prompt_builder.py`, modify SupportResistancePrompt.render() (around line 601):

```python
def render(
    self,
    symbol: str,
    timeframe: str,
    current_price: float,
    market_regime: str,
    price_high: float,
    price_low: float,
    price_range: float,
    swing_highs: str,
    swing_lows: str,
    price_summary: str,
    num_bars: int = 10,
    higher_tf_context: str | None = None,
    execution_context: str = "",  # NEW
) -> str:
    """Render support/resistance prompt."""
    # Build higher timeframe section
    higher_tf_section = ""
    if higher_tf_context:
        higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

    return self.TEMPLATE.format(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        market_regime=market_regime,
        price_high=price_high,
        price_low=price_low,
        price_range=price_range,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        price_summary=price_summary,
        num_bars=num_bars,
        higher_tf_section=higher_tf_section,
        execution_context=execution_context,  # NEW
    )
```

- [ ] **Step 7: Update ASSESS_MOMENTUM rendering in build_prompt()**

In `data/prompt_builder.py`, modify ASSESS_MOMENTUM rendering (around line 961):

```python
elif task.task_type == TaskType.ASSESS_MOMENTUM:
    # ... (existing momentum indicator calculation)

    prompt = template.render(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        market_regime=market_regime.value,
        rsi=rsi,
        rsi_prev=rsi_prev,
        rsi_delta=rsi_delta,
        macd=macd,
        macd_signal=macd_signal,
        macd_prev=macd_prev,
        macd_delta=macd_delta,
        bb_width=current_bb_width,
        bb_trend=bb_trend,
        price_summary=price_summary,
        higher_tf_context=higher_tf_context,
        execution_context=execution_context,  # NEW
    )
```

- [ ] **Step 8: Update IDENTIFY_SUPPORT_RESISTANCE rendering in build_prompt()**

In `data/prompt_builder.py`, modify IDENTIFY_SUPPORT_RESISTANCE rendering (around line 993):

```python
elif task.task_type == TaskType.IDENTIFY_SUPPORT_RESISTANCE:
    # ... (existing S/R calculation)

    prompt = template.render(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        market_regime=market_regime.value,
        price_high=price_high,
        price_low=price_low,
        price_range=price_range,
        swing_highs=swing_highs_str,
        swing_lows=swing_lows_str,
        price_summary=price_summary,
        higher_tf_context=higher_tf_context,
        execution_context=execution_context,  # NEW
    )
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder_fee_context.py::test_execution_context_in_all_templates -xvs`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add data/prompt_builder.py tests/test_prompt_builder_fee_context.py
git commit -m "feat: add execution context to all prompt templates

- Update MomentumAssessmentPrompt template and render()
- Update SupportResistancePrompt template and render()
- Update build_prompt() rendering for all task types
- Add test verifying all templates include execution context"
```

---

## Task 3: PromptBuilder - Test Execution Context Edge Cases

**Files:**
- Test: `tests/test_prompt_builder_fee_context.py`

- [ ] **Step 1: Write test for spot mode execution context**

Add to `tests/test_prompt_builder_fee_context.py`:

```python
def test_execution_context_spot_mode(sample_df):
    """Verify execution context shows Spot mode when funding disabled."""
    builder = PromptBuilder()
    fee_model = FeeModelSettings(
        maker_fee_pct=0.10,
        taker_fee_pct=0.10,
        include_funding=False,  # Spot mode
        slippage_pct=0.05,
    )

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
    assert "Mode: Spot" in prompt
    assert "Futures" not in prompt
```

- [ ] **Step 2: Write test for none mode (no execution context)**

Add to `tests/test_prompt_builder_fee_context.py`:

```python
def test_execution_context_none_mode(sample_df):
    """Verify execution context omitted when fee_model=None."""
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
        fee_model=None,  # No fee model
    )

    assert "## Execution Context" not in prompt
    assert "Estimated round-trip cost" not in prompt
```

- [ ] **Step 3: Write test for dynamic costs by timeframe**

Add to `tests/test_prompt_builder_fee_context.py`:

```python
def test_execution_context_dynamic_costs(sample_df):
    """Verify costs vary by timeframe due to different holding periods."""
    builder = PromptBuilder()
    fee_model = FeeModelSettings()  # Futures defaults

    task = TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )

    # Test 1h timeframe (24 bars = 1 funding period)
    prompt_1h = builder.build_prompt(
        task=task,
        df=sample_df,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model,
    )

    # Test 1d timeframe (5 bars = ~2.5 funding periods)
    prompt_1d = builder.build_prompt(
        task=task,
        df=sample_df,
        symbol="BTC/USDT",
        timeframe="1d",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model,
    )

    # Extract costs from prompts
    import re

    cost_1h_match = re.search(r"Estimated round-trip cost: ([\d.]+)%", prompt_1h)
    cost_1d_match = re.search(r"Estimated round-trip cost: ([\d.]+)%", prompt_1d)

    assert cost_1h_match is not None
    assert cost_1d_match is not None

    cost_1h = float(cost_1h_match.group(1))
    cost_1d = float(cost_1d_match.group(1))

    # 1d should have higher costs due to more funding periods
    assert cost_1d > cost_1h, f"1d cost ({cost_1d}%) should exceed 1h cost ({cost_1h}%)"
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_prompt_builder_fee_context.py -xvs`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_prompt_builder_fee_context.py
git commit -m "test: add execution context edge case tests

- Test spot mode (no funding)
- Test none mode (section omitted)
- Test dynamic costs vary by timeframe"
```

---

## Task 4: RewardEngine - Switch to Net Returns

**Files:**
- Modify: `training/reward_engine.py:94-232`
- Test: `tests/test_reward_net_returns.py` (create)

- [ ] **Step 1: Write failing test for net return usage**

Create `tests/test_reward_net_returns.py`:

```python
"""Tests for net return usage in reward computation."""

import pytest
from datetime import datetime, UTC

from config.fee_model import FeeModelSettings
from swarm.training_capture import TrainingExample
from training.reward_engine import compute_reward
from verifier.outcome import VerifiedOutcome


def test_reward_uses_net_return():
    """Verify reward engine uses net_return, not realized_return."""
    # Example with positive gross but negative net
    outcome = VerifiedOutcome(
        example_id="test_001",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30024.0,
        actual_direction="HIGHER",
        realized_return=0.0008,  # +0.08% gross (log return)
        net_return=-0.00013,     # -0.013% net (after 0.093% fees)
        max_adverse_excursion=-0.0001,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    example = TrainingExample(
        example_id="test_001",
        context_id="BTC_USDT_1h_1704067200000_predict_direction",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="technical_analyst",
        task_prompt="...",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.8,
            "reasoning": "Test reasoning",
        },
        critique="Test critique",
        was_accepted=True,
        generation_duration_sec=1.0,
    )

    reward = compute_reward(outcome, example)

    # Net return is negative, so return_reward should be negative
    assert reward.return_reward < 0, f"Expected negative return_reward for net_return={outcome.net_return}"
    assert reward.net_return == outcome.net_return
    assert reward.realized_return == outcome.realized_return


def test_profitable_after_fees():
    """Example with +0.30% gross → +0.207% net (profitable)."""
    outcome = VerifiedOutcome(
        example_id="test_002",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30090.0,
        actual_direction="HIGHER",
        realized_return=0.003,      # +0.30% gross
        net_return=0.00207,         # +0.207% net (after fees)
        max_adverse_excursion=-0.0002,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    example = TrainingExample(
        example_id="test_002",
        context_id="BTC_USDT_1h_1704067200000_predict_direction",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="momentum_trader",
        task_prompt="...",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.9,
            "reasoning": "Strong momentum",
        },
        critique="Good signal",
        was_accepted=True,
        generation_duration_sec=1.2,
    )

    reward = compute_reward(outcome, example)

    # Net return is positive, return_reward should be positive
    assert reward.return_reward > 0, f"Expected positive return_reward for net_return={outcome.net_return}"
    # Final reward should be positive (good signal)
    assert reward.final_reward > 0


def test_unprofitable_after_fees():
    """Example with +0.08% gross → -0.013% net (unprofitable)."""
    outcome = VerifiedOutcome(
        example_id="test_003",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30024.0,
        actual_direction="HIGHER",
        realized_return=0.0008,     # +0.08% gross
        net_return=-0.00013,        # -0.013% net (after fees)
        max_adverse_excursion=-0.00015,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    example = TrainingExample(
        example_id="test_003",
        context_id="BTC_USDT_1h_1704067200000_predict_direction",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="swing_trader",
        task_prompt="...",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.6,
            "reasoning": "Marginal signal",
        },
        critique="Low conviction",
        was_accepted=True,
        generation_duration_sec=0.9,
    )

    reward = compute_reward(outcome, example)

    # Net return is negative despite correct direction
    assert reward.return_reward < 0, f"Expected negative return_reward for net_return={outcome.net_return}"
    # This signal should be penalized despite correct direction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward_net_returns.py::test_reward_uses_net_return -xvs`
Expected: FAIL - "Expected negative return_reward but got positive" (currently using realized_return)

- [ ] **Step 3: Modify compute_reward() to use net_return**

In `training/reward_engine.py`, modify `compute_reward()` function (around line 143):

Change from:
```python
# Always compute return component
return_reward = compute_return_reward(
    verified_outcome.realized_return,
    scaling.return_scale,
)
```

To:
```python
# Always compute return component (use NET return after fees)
return_reward = compute_return_reward(
    verified_outcome.net_return,
    scaling.return_scale,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward_net_returns.py -xvs`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add training/reward_engine.py tests/test_reward_net_returns.py
git commit -m "feat: switch reward engine to use net returns

- Change compute_reward() to use net_return instead of realized_return
- Add tests verifying net return usage
- Signals below fee threshold now get negative rewards"
```

---

## Task 5: DPO Export - Test Preference Pair Net Ranking

**Files:**
- Test: `tests/test_dpo_export_net_ranking.py` (create)

- [ ] **Step 1: Write test for preference pair net ranking**

Create `tests/test_dpo_export_net_ranking.py`:

```python
"""Tests for preference pair ranking by net rewards."""

import pytest
from datetime import datetime, UTC

from swarm.training_capture import TrainingExample
from training.dpo_export import construct_preference_pairs
from training.reward_engine import compute_reward
from verifier.outcome import VerifiedOutcome


def test_preference_pairs_rank_by_net_reward():
    """Verify chosen example has higher NET reward, not gross."""
    # Example A: +0.30% gross → +0.207% net (profitable)
    example_a = TrainingExample(
        example_id="test_A",
        context_id="BTC_USDT_1h_1704067200000_predict_direction",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="momentum_trader",
        task_prompt="Analyze BTC/USDT and predict direction...",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.9,
            "reasoning": "Strong bullish momentum with RSI 68",
        },
        critique="Good signal",
        was_accepted=True,
        generation_duration_sec=1.2,
    )

    outcome_a = VerifiedOutcome(
        example_id="test_A",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30090.0,
        actual_direction="HIGHER",
        realized_return=0.003,      # +0.30% gross
        net_return=0.00207,         # +0.207% net (profitable after fees)
        max_adverse_excursion=-0.0002,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    reward_a = compute_reward(outcome_a, example_a)

    # Example B: +0.08% gross → -0.013% net (unprofitable)
    example_b = TrainingExample(
        example_id="test_B",
        context_id="BTC_USDT_1h_1704067200000_predict_direction",  # Same context
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="swing_trader",
        task_prompt="Analyze BTC/USDT and predict direction...",
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.6,
            "reasoning": "Marginal bullish signal, low conviction",
        },
        critique="Weak signal",
        was_accepted=True,
        generation_duration_sec=0.9,
    )

    outcome_b = VerifiedOutcome(
        example_id="test_B",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30024.0,
        actual_direction="HIGHER",
        realized_return=0.0008,     # +0.08% gross
        net_return=-0.00013,        # -0.013% net (unprofitable after fees)
        max_adverse_excursion=-0.00015,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    reward_b = compute_reward(outcome_b, example_b)

    # Verify rewards
    assert reward_a.final_reward > 0, "Example A should have positive reward"
    assert reward_b.final_reward < 0, "Example B should have negative reward"

    # Construct preference pairs
    pairs = construct_preference_pairs(
        [(example_a, outcome_a, reward_a), (example_b, outcome_b, reward_b)],
        min_delta=0.1,
        min_personas_per_context=1,  # Allow pairs with <3 personas for testing
    )

    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    pair = pairs[0]

    # Example A should be chosen (higher net reward)
    assert pair.chosen_example_id == example_a.example_id, \
        f"Expected {example_a.example_id} as chosen, got {pair.chosen_example_id}"
    assert pair.rejected_example_id == example_b.example_id, \
        f"Expected {example_b.example_id} as rejected, got {pair.rejected_example_id}"
    assert pair.reward_delta > 0, f"Reward delta should be positive, got {pair.reward_delta}"

    # Verify reasoning appears in pair
    assert "Strong bullish momentum" in pair.chosen_reasoning
    assert "Marginal bullish signal" in pair.rejected_reasoning
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_dpo_export_net_ranking.py::test_preference_pairs_rank_by_net_reward -xvs`
Expected: PASS (preference pairs already rank by final_reward, which now uses net_return)

- [ ] **Step 3: Commit**

```bash
git add tests/test_dpo_export_net_ranking.py
git commit -m "test: verify preference pairs rank by net rewards

- Test that chosen example has higher NET reward
- Verify profitable (+0.30% gross) chosen over unprofitable (+0.08% gross)
- Confirms DPO training optimizes for net profitability"
```

---

## Task 6: Run DPO Training - Add Fee Flip Diagnostic

**Files:**
- Modify: `run_dpo_training.py:130-292`
- Test: `tests/test_fee_flip_diagnostic.py` (create)

- [ ] **Step 1: Write failing test for fee flip detection**

Create `tests/test_fee_flip_diagnostic.py`:

```python
"""Tests for fee flip diagnostic logic."""

import pytest
from datetime import datetime, UTC
from io import StringIO
import sys

from config.fee_model import FeeModelSettings
from swarm.training_capture import TrainingExample
from verifier.outcome import VerifiedOutcome


def test_fee_flip_detection():
    """Verify flip detection identifies examples that cross profitability threshold."""
    from run_dpo_training import compute_fee_flip_diagnostic

    fee_model = FeeModelSettings()  # Futures defaults

    # Example 1: +0.30% gross → profitable after fees (no flip)
    example1 = TrainingExample(
        example_id="ex1",
        context_id="ctx1",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="momentum_trader",
        task_prompt="...",
        generator_signal={"direction": "HIGHER", "confidence": 0.9, "reasoning": "..."},
        critique="...",
        was_accepted=True,
        generation_duration_sec=1.0,
    )

    outcome1 = VerifiedOutcome(
        example_id="ex1",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30090.0,
        actual_direction="HIGHER",
        realized_return=0.003,      # +0.30% gross
        net_return=0.00207,         # +0.207% net (profitable)
        max_adverse_excursion=None,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    # Example 2: +0.08% gross → unprofitable after fees (FLIP)
    example2 = TrainingExample(
        example_id="ex2",
        context_id="ctx2",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="swing_trader",
        task_prompt="...",
        generator_signal={"direction": "HIGHER", "confidence": 0.6, "reasoning": "..."},
        critique="...",
        was_accepted=True,
        generation_duration_sec=0.9,
    )

    outcome2 = VerifiedOutcome(
        example_id="ex2",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30024.0,
        actual_direction="HIGHER",
        realized_return=0.0008,     # +0.08% gross
        net_return=-0.00013,        # -0.013% net (unprofitable)
        max_adverse_excursion=None,
        verification_timestamp=datetime.now(UTC).isoformat(),
    )

    examples_and_outcomes = [
        (example1, outcome1),
        (example2, outcome2),
    ]

    # Capture printed output
    captured_output = StringIO()
    sys.stdout = captured_output

    compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Verify diagnostic output
    assert "=== FEE FLIP DIAGNOSTIC ===" in output
    assert "1h" in output
    assert "50.0%" in output or "50%" in output  # 1 out of 2 flipped = 50%
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fee_flip_diagnostic.py::test_fee_flip_detection -xvs`
Expected: FAIL with "ImportError: cannot import name 'compute_fee_flip_diagnostic'"

- [ ] **Step 3: Add FEE_FLIP_WARNING_THRESHOLD constant**

In `run_dpo_training.py`, add after imports (around line 48):

```python
# --------------------------------------------------------------------------- #
# Fee flip diagnostic
# --------------------------------------------------------------------------- #

FEE_FLIP_WARNING_THRESHOLD = 0.15  # 15% flip rate triggers warning
```

- [ ] **Step 4: Implement compute_fee_flip_diagnostic() function**

In `run_dpo_training.py`, add function after the constant (around line 133):

```python
def compute_fee_flip_diagnostic(
    examples_and_outcomes: list[tuple[TrainingExample, VerifiedOutcome]],
    fee_model: FeeModelSettings,
) -> None:
    """
    Print diagnostic showing examples that flip from positive to negative
    under realistic fees, grouped by timeframe.

    Args:
        examples_and_outcomes: List of (example, outcome) tuples
        fee_model: Fee model to use for realistic cost calculation
    """
    if not examples_and_outcomes:
        return

    # Group by timeframe
    by_timeframe: dict[str, list[tuple[TrainingExample, VerifiedOutcome]]] = {}
    for example, outcome in examples_and_outcomes:
        tf = example.timeframe
        if tf not in by_timeframe:
            by_timeframe[tf] = []
        by_timeframe[tf].append((example, outcome))

    # Compute flips per timeframe
    flip_stats: dict[str, dict] = {}
    for tf, pairs in by_timeframe.items():
        total = len(pairs)
        flipped = 0
        old_net_sum = 0.0
        new_net_sum = 0.0

        horizon_bars = get_horizon_bars(tf)
        holding_periods = compute_holding_periods_8h(tf, horizon_bars)

        for example, outcome in pairs:
            # Convert log return to percentage
            gross_pct = (math.exp(outcome.realized_return) - 1) * 100

            # Old: flat 0.1% cost (deprecated model)
            old_net_pct = gross_pct - 0.1

            # New: realistic fees
            new_net_pct = fee_model.net_return(gross_pct, holding_periods)

            old_net_sum += old_net_pct
            new_net_sum += new_net_pct

            # Check for flip: was positive under old model, negative under new
            if old_net_pct > 0 and new_net_pct < 0:
                flipped += 1

        flip_stats[tf] = {
            "total": total,
            "flipped": flipped,
            "flip_rate": flipped / total if total > 0 else 0,
            "avg_old_net": old_net_sum / total if total > 0 else 0,
            "avg_new_net": new_net_sum / total if total > 0 else 0,
        }

    # Print table
    print()
    print("=== FEE FLIP DIAGNOSTIC ===")
    print(
        "Examples that were profitable under flat 0.1% fees but are unprofitable "
        "with realistic Binance Futures fees:"
    )
    print()
    print(
        "Timeframe | Total Examples | Flipped to Negative | Flip Rate | "
        "Avg Old Net | Avg New Net"
    )
    print(
        "----------|----------------|---------------------|-----------|"
        "-------------|-------------"
    )

    total_examples = 0
    total_flipped = 0

    for tf in sorted(flip_stats.keys()):
        stats = flip_stats[tf]
        print(
            f"{tf:>9} | {stats['total']:>14} | {stats['flipped']:>19} | "
            f"{stats['flip_rate']:>8.1%} | {stats['avg_old_net']:>+10.2f}% | "
            f"{stats['avg_new_net']:>+10.2f}%"
        )
        total_examples += stats["total"]
        total_flipped += stats["flipped"]

    print(
        "----------|----------------|---------------------|-----------|"
        "-------------|-------------"
    )
    overall_flip_rate = total_flipped / total_examples if total_examples > 0 else 0
    print(
        f"{'TOTAL':>9} | {total_examples:>14} | {total_flipped:>19} | "
        f"{overall_flip_rate:>8.1%} |             |"
    )
    print()

    # 1d funding breakdown
    if "1d" in flip_stats:
        horizon_1d = get_horizon_bars("1d")
        periods_1d = compute_holding_periods_8h("1d", horizon_1d)
        funding_1d = fee_model.funding_rate_pct * periods_1d
        print(
            f"1d funding cost alone: {funding_1d:.2f}% "
            f"({periods_1d:.0f} periods × {fee_model.funding_rate_pct:.2f}%)"
        )
        print()

    # Warnings
    for tf, stats in flip_stats.items():
        if stats["flip_rate"] > FEE_FLIP_WARNING_THRESHOLD:
            print(
                f"WARNING: {tf} timeframe has {stats['flip_rate']:.1%} flip rate - "
                "signals may not clear fee hurdle."
            )
            print(
                "Consider focusing training on longer timeframes or increasing "
                "signal selectivity."
            )
            print()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_fee_flip_diagnostic.py::test_fee_flip_detection -xvs`
Expected: PASS

- [ ] **Step 6: Integrate diagnostic into phase3_reward()**

In `run_dpo_training.py`, modify `phase3_reward()` function (around line 287):

Add at the end of the function, before the return statement:

```python
# Phase 3 diagnostic
examples_and_outcomes = [(ex, outcome) for ex, outcome, _ in result]
compute_fee_flip_diagnostic(examples_and_outcomes, fee_model=fee_model)

return result
```

- [ ] **Step 7: Run test to verify integration**

Run: `pytest tests/test_fee_flip_diagnostic.py -xvs`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add run_dpo_training.py tests/test_fee_flip_diagnostic.py
git commit -m "feat: add fee flip diagnostic to DPO pipeline

- Add compute_fee_flip_diagnostic() function
- Print table showing flip rate by timeframe
- Warn when >15% of examples flip to negative
- Integrate into phase3_reward()
- Add test for flip detection logic"
```

---

## Task 7: Run DPO Training - Add CLI Fee Mode Flag

**Files:**
- Modify: `run_dpo_training.py:361-497`

- [ ] **Step 1: Add --fee-mode argument to parser**

In `run_dpo_training.py`, in `parse_args()` function (around line 409):

```python
parser.add_argument(
    "--force",
    action="store_true",
    help="Skip 24h promotion cooldown check",
)
parser.add_argument(
    "--fee-mode",
    type=str,
    default="futures_usdt",
    choices=["futures_usdt", "spot", "none"],
    help="Fee model mode: futures_usdt (default: 0.02%%/0.05%% maker/taker + funding), "
         "spot (0.10%% maker/taker, no funding), none (legacy 0.1%% flat)",
)
return parser.parse_args()
```

- [ ] **Step 2: Add create_fee_model() helper function**

In `run_dpo_training.py`, add after imports and before phase1_load() (around line 51):

```python
def create_fee_model(mode: str) -> FeeModelSettings | None:
    """
    Create fee model based on mode string.

    Args:
        mode: One of "futures_usdt", "spot", "none"

    Returns:
        FeeModelSettings instance or None for legacy mode

    Raises:
        ValueError: If mode is invalid
    """
    if mode == "futures_usdt":
        return FeeModelSettings()  # Defaults: Binance Futures USDT-M
    elif mode == "spot":
        return FeeModelSettings(
            maker_fee_pct=0.10,
            taker_fee_pct=0.10,
            entry_order_type="taker",
            exit_order_type="taker",
            bnb_discount_enabled=False,
            include_funding=False,
            slippage_pct=0.05,
        )
    elif mode == "none":
        return None  # Legacy behavior: no fee model
    else:
        raise ValueError(f"Invalid fee mode: {mode}")
```

- [ ] **Step 3: Update main() to create and use fee_model**

In `run_dpo_training.py`, modify `main()` function (around line 438):

Add after logging configuration section:

```python
logger.info("=" * 60)
logger.info("STARTING DPO TRAINING PIPELINE")
logger.info("=" * 60)

# Create fee model based on CLI flag
fee_model = create_fee_model(args.fee_mode)

logger.info(
    "Configuration",
    dataset=str(args.dataset),
    output_dir=str(output_dir),
    min_delta=args.min_delta,
    dry_run=args.dry_run,
    save_pairs=args.save_pairs,
    force=args.force,
    fee_mode=args.fee_mode,
)
```

- [ ] **Step 4: Pass fee_model to phase2_verify()**

In `run_dpo_training.py`, modify phase2_verify() call (around line 443):

```python
# Phase 2: Verify
matched = phase2_verify(examples, fee_model=fee_model)
if not matched:
    logger.error("No examples verified — cannot build preference pairs")
    sys.exit(1)
```

- [ ] **Step 5: Pass fee_model to phase3_reward()**

In `run_dpo_training.py`, modify phase3_reward() call (around line 449):

```python
# Phase 3: Reward
examples_with_rewards = phase3_reward(matched, fee_model=fee_model)
```

- [ ] **Step 6: Test CLI flag parsing**

Run: `python run_dpo_training.py --help`
Expected: See `--fee-mode` option with three choices

- [ ] **Step 7: Commit**

```bash
git add run_dpo_training.py
git commit -m "feat: add --fee-mode CLI flag to DPO training

- Add --fee-mode argument (futures_usdt | spot | none)
- Add create_fee_model() helper function
- Pass fee_model through verification and reward phases
- Default to futures_usdt mode"
```

---

## Task 8: Generate Training Dataset - Add Fee Mode Support

**Files:**
- Modify: `generate_training_dataset.py:442-571`

- [ ] **Step 1: Add --fee-mode argument to parser**

In `generate_training_dataset.py`, in `parse_args()` function (around line 511):

```python
parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Quick test mode: 1 symbol, 1 timeframe, 3 windows",
)
parser.add_argument(
    "--fee-mode",
    type=str,
    default="futures_usdt",
    choices=["futures_usdt", "spot", "none"],
    help="Fee model for execution context in prompts: futures_usdt (default), spot, or none (omit context)",
)

args = parser.parse_args()
```

- [ ] **Step 2: Add fee_mode to DatasetConfig dataclass**

In `generate_training_dataset.py`, modify DatasetConfig (around line 64):

```python
@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    symbols: list[str]
    timeframes: list[str]
    window_count: int
    window_stride_bars: int
    lookback_bars: int
    task_types: list[TaskType]
    output_dir: Path
    resume_from: Optional[Path] = None
    fee_mode: str = "futures_usdt"  # NEW
```

- [ ] **Step 3: Add create_fee_model() helper function**

In `generate_training_dataset.py`, add after DatasetConfig (around line 75):

```python
def create_fee_model(mode: str) -> "FeeModelSettings | None":
    """
    Create fee model based on mode string.

    Args:
        mode: One of "futures_usdt", "spot", "none"

    Returns:
        FeeModelSettings instance or None for no execution context

    Raises:
        ValueError: If mode is invalid
    """
    from config.fee_model import FeeModelSettings

    if mode == "futures_usdt":
        return FeeModelSettings()  # Defaults: Binance Futures USDT-M
    elif mode == "spot":
        return FeeModelSettings(
            maker_fee_pct=0.10,
            taker_fee_pct=0.10,
            entry_order_type="taker",
            exit_order_type="taker",
            bnb_discount_enabled=False,
            include_funding=False,
            slippage_pct=0.05,
        )
    elif mode == "none":
        return None  # No execution context
    else:
        raise ValueError(f"Invalid fee mode: {mode}")
```

- [ ] **Step 4: Pass fee_mode to DatasetConfig in parse_args()**

In `generate_training_dataset.py`, modify `parse_args()` return statement (around line 537):

```python
return DatasetConfig(
    symbols=symbols,
    timeframes=timeframes,
    window_count=window_count,
    window_stride_bars=args.stride,
    lookback_bars=args.lookback,
    task_types=task_types,
    output_dir=output_dir,
    resume_from=resume_from,
    fee_mode=args.fee_mode,  # NEW
)
```

- [ ] **Step 5: Create fee_model in phase1_prepare_contexts()**

In `generate_training_dataset.py`, modify `phase1_prepare_contexts()` (around line 102):

```python
jobs = []
regime_classifier = RegimeClassifier()
prompt_builder = PromptBuilder()
fee_model = create_fee_model(config.fee_mode)  # NEW

async with MarketDataService() as service:
```

- [ ] **Step 6: Pass fee_model to build_prompt()**

In `generate_training_dataset.py`, modify prompt building (around line 189):

```python
task_prompt = prompt_builder.build_prompt(
    task=task_config,
    df=df,
    symbol=window.symbol,
    timeframe=window.timeframe,
    market_regime=market_regime,
    fee_model=fee_model,  # NEW
)
```

- [ ] **Step 7: Test CLI flag parsing**

Run: `python generate_training_dataset.py --help`
Expected: See `--fee-mode` option with three choices

- [ ] **Step 8: Commit**

```bash
git add generate_training_dataset.py
git commit -m "feat: add --fee-mode flag to dataset generation

- Add --fee-mode CLI argument (futures_usdt | spot | none)
- Add create_fee_model() helper function
- Pass fee_model to PromptBuilder.build_prompt()
- Add fee_mode to DatasetConfig
- Default to futures_usdt mode"
```

---

## Task 9: Integration Test - End-to-End Fee-Aware Workflow

**Files:**
- Test: `tests/test_fee_aware_integration.py` (create)

- [ ] **Step 1: Write integration test**

Create `tests/test_fee_aware_integration.py`:

```python
"""End-to-end integration test for fee-aware training workflow."""

import pytest
from pathlib import Path
import tempfile

from config.fee_model import FeeModelSettings
from data.prompt_builder import PromptBuilder, TaskConfig, TaskType
from data.regime_filter import MarketRegime
from swarm.training_capture import TrainingExample
from training.reward_engine import compute_reward
from training.dpo_export import construct_preference_pairs
from verifier.outcome import VerifiedOutcome
from tests.fixtures.timeframe_fixtures import create_test_df_bullish


def test_fee_aware_end_to_end():
    """
    Test complete fee-aware workflow:
    1. Prompt includes execution context
    2. Reward uses net returns
    3. Preference pairs rank by net profitability
    """
    # Step 1: Generate prompt with execution context
    builder = PromptBuilder()
    fee_model = FeeModelSettings()  # Futures defaults

    df = create_test_df_bullish(bars=100)

    task = TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )

    prompt = builder.build_prompt(
        task=task,
        df=df,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model,
    )

    # Verify prompt has execution context
    assert "## Execution Context" in prompt
    assert "Mode: Futures USDT-M" in prompt
    assert "Estimated round-trip cost:" in prompt

    # Step 2: Create two signals - one profitable, one not
    import datetime

    # Signal A: +0.30% gross → profitable after fees
    example_a = TrainingExample(
        example_id="integ_a",
        context_id="BTC_USDT_1h_test_ctx",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="momentum_trader",
        task_prompt=prompt,
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.9,
            "reasoning": "Strong momentum signal",
        },
        critique="Good",
        was_accepted=True,
        generation_duration_sec=1.0,
    )

    outcome_a = VerifiedOutcome(
        example_id="integ_a",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30090.0,
        actual_direction="HIGHER",
        realized_return=0.003,
        net_return=0.00207,
        max_adverse_excursion=-0.0002,
        verification_timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    # Signal B: +0.08% gross → unprofitable after fees
    example_b = TrainingExample(
        example_id="integ_b",
        context_id="BTC_USDT_1h_test_ctx",  # Same context for pairing
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        market_regime="NEUTRAL",
        persona="swing_trader",
        task_prompt=prompt,
        generator_signal={
            "direction": "HIGHER",
            "confidence": 0.6,
            "reasoning": "Marginal signal",
        },
        critique="Weak",
        was_accepted=True,
        generation_duration_sec=0.9,
    )

    outcome_b = VerifiedOutcome(
        example_id="integ_b",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1704067200000,
        holding_period_bars=24,
        entry_price=30000.0,
        exit_price=30024.0,
        actual_direction="HIGHER",
        realized_return=0.0008,
        net_return=-0.00013,
        max_adverse_excursion=-0.00015,
        verification_timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    # Step 3: Compute rewards (should use net returns)
    reward_a = compute_reward(outcome_a, example_a)
    reward_b = compute_reward(outcome_b, example_b)

    assert reward_a.final_reward > 0, "Profitable signal should have positive reward"
    assert reward_b.final_reward < 0, "Unprofitable signal should have negative reward"

    # Step 4: Construct preference pairs
    pairs = construct_preference_pairs(
        [(example_a, outcome_a, reward_a), (example_b, outcome_b, reward_b)],
        min_delta=0.1,
        min_personas_per_context=1,
    )

    assert len(pairs) == 1
    pair = pairs[0]

    # Step 5: Verify preference ranking by net profitability
    assert pair.chosen_example_id == "integ_a"
    assert pair.rejected_example_id == "integ_b"
    assert pair.reward_delta > 0

    # Step 6: Verify prompt appeared in both examples
    assert pair.prompt == prompt
    assert "## Execution Context" in pair.prompt


def test_fee_mode_none_backward_compatibility():
    """Verify fee_mode=none preserves legacy behavior (no execution context)."""
    builder = PromptBuilder()

    df = create_test_df_bullish(bars=100)

    task = TaskConfig(
        task_type=TaskType.PREDICT_DIRECTION,
        weight=1.0,
        difficulty=2,
        min_bars_required=50,
    )

    # fee_model=None should omit execution context
    prompt = builder.build_prompt(
        task=task,
        df=df,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=None,
    )

    assert "## Execution Context" not in prompt
    assert "Estimated round-trip cost" not in prompt
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_fee_aware_integration.py -xvs`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_fee_aware_integration.py
git commit -m "test: add end-to-end fee-aware integration tests

- Test complete workflow: prompt → reward → pairs
- Verify profitable signals chosen over unprofitable
- Test backward compatibility with fee_model=None"
```

---

## Task 10: Documentation and Final Validation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with session summary**

In `CLAUDE.md`, update "Current State" section:

```markdown
**Completed:**
- Sessions 1-13: ... (existing sessions)
- Session 14: Fee-Aware DPO Training (execution context in prompts, net return rewards, fee flip diagnostic)

**Next:** TBD
```

In "Architecture Decisions" section, add:

```markdown
### Fee-Aware Training (Session 14)
- **Teaching + Enforcement Pattern:** Prompts show fee structure, rewards use net returns
- **Execution Context Section:** Dynamic costs per timeframe (1h: 0.093%, 1d: ~0.14%)
- **Net Return Rewards:** RewardEngine switched from realized_return to net_return
- **Fee Flip Diagnostic:** Reports % of examples that cross profitability threshold
- **CLI Integration:** `--fee-mode` flag (futures_usdt | spot | none) in both dataset generation and DPO training
- **Backward Compatible:** `--fee-mode none` preserves legacy behavior for ablation studies
```

In "File Index" section, add:

```markdown
### Tests
- ... (existing test files)
- `tests/test_prompt_builder_fee_context.py` - 5 tests for execution context rendering
- `tests/test_reward_net_returns.py` - 3 tests for net return reward computation
- `tests/test_dpo_export_net_ranking.py` - 1 test for preference pair net ranking
- `tests/test_fee_flip_diagnostic.py` - 1 test for fee flip detection
- `tests/test_fee_aware_integration.py` - 2 end-to-end integration tests
```

Update test count:

```markdown
**Total Tests:** 701 → 713 passing
```

- [ ] **Step 2: Run full test suite**

Run: `pytest -xvs --tb=short`
Expected: All tests pass (701 existing + 12 new = 713 total)

- [ ] **Step 3: Commit CLAUDE.md update**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Session 14 completion

- Mark Session 14 (Fee-Aware DPO Training) as complete
- Document teaching + enforcement architecture
- Add new test files to index
- Update test count: 701 → 713 passing"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ Execution context in prompts (Tasks 1-3)
- ✅ Net return rewards (Task 4)
- ✅ Preference pair net ranking (Task 5)
- ✅ Fee flip diagnostic (Task 6)
- ✅ CLI fee mode flag - run_dpo_training.py (Task 7)
- ✅ CLI fee mode flag - generate_training_dataset.py (Task 8)
- ✅ Integration tests (Task 9)
- ✅ Documentation (Task 10)

**2. Placeholder scan:**
- No TBD, TODO, or "implement later"
- All code blocks complete
- All test cases have full implementation
- No references to undefined functions

**3. Type consistency:**
- `fee_model: FeeModelSettings | None` used consistently
- `execution_context: str = ""` used consistently
- All template render() methods have matching signatures

**4. Test coverage:**
- PromptBuilder: 5 tests (futures/spot/none modes, dynamic costs, all templates)
- RewardEngine: 3 tests (net return usage, profitable/unprofitable cases)
- DPO Export: 1 test (net ranking verification)
- Fee Flip Diagnostic: 1 test (flip detection)
- Integration: 2 tests (end-to-end, backward compatibility)
- **Total new tests:** 12

**5. Backward compatibility:**
- `fee_model=None` parameter default preserves existing behavior
- `--fee-mode none` CLI flag for ablation studies
- All existing tests continue to pass

---

## Success Criteria

1. ✅ **Execution Context appears in prompts** - Tasks 1-3 implement for all 3 templates
2. ✅ **Net rewards computed correctly** - Task 4 switches to net_return
3. ✅ **Preference pairs rank by net** - Task 5 validates net ranking
4. ✅ **Fee flip diagnostic works** - Task 6 implements and tests diagnostic
5. ✅ **Tests pass** - 12 new tests across 4 test files
6. ✅ **Backward compatible** - fee_model=None and --fee-mode none tested
