# Multi-Timeframe Context Design

**Date:** 2026-04-06
**Status:** Approved
**Implementation:** Pending

## Problem Statement

Currently, each trading signal is generated for a single timeframe in isolation. This leaves alpha on the table because a bullish 1h signal that aligns with bullish 4h and 1d trends is far more reliable than one that contradicts higher timeframes. Multi-timeframe confluence is a fundamental concept in technical analysis that can significantly improve signal quality.

## Goals

1. Add higher timeframe trend context to signal generation prompts
2. Enable the LLM generator to consider multi-timeframe confluence when making predictions
3. Keep implementation simple and prompt-length controlled (max 2 higher timeframes)
4. Maintain backward compatibility with existing single-timeframe workflows
5. No changes to VRAM management or inference queue architecture

## Non-Goals

- Changing the inference queue or adding separate model calls for higher timeframes
- Modifying training or evaluation pipelines (they receive enhanced prompts transparently)
- Implementing automatic timeframe selection strategies (caller decides which TFs to provide)
- Adding multi-timeframe analysis to verifier or reward layers

## Architecture Overview

### Core Components

**1. Timeframe Hierarchy Constant**

Define canonical timeframe ordering in `data/prompt_builder.py`:

```python
TIMEFRAME_HIERARCHY = ["1m", "5m", "15m", "1h", "4h", "1d"]
```

**2. New Helper Functions** (all in `data/prompt_builder.py`)

- `get_higher_timeframes(current_tf: str, available_tfs: list[str]) -> list[str]`
  - Returns up to 2 nearest higher timeframes from available set
  - Respects hierarchy ordering

- `summarize_timeframe(df: pd.DataFrame, timeframe: str) -> dict`
  - Generates trend summary from OHLCV DataFrame
  - Returns structured dict with trend classification and human-readable text

- `compute_confluence(summaries: list[dict]) -> str`
  - Analyzes trend alignment across timeframes
  - Returns "aligned", "conflicting", or "mixed"

**3. Modified Components**

- `PromptBuilder.build_prompt()` - Add `higher_tf_data: dict[str, pd.DataFrame] | None = None` parameter
- Each prompt template - Add optional higher timeframe section
- `run_swarm_workflow()` in `swarm/orchestrator.py` - Add `higher_tf_data` parameter and pass through

### Data Flow

```
Caller (e.g., generate_training_dataset.py)
  ↓
run_swarm_workflow(
    symbol="BTC/USDT",
    timeframe="1h",
    higher_tf_data={"4h": df_4h, "1d": df_1d}
)
  ↓
PromptBuilder.build_prompt(higher_tf_data=...)
  ↓
get_higher_timeframes("1h", ["4h", "1d"]) → ["4h", "1d"]
  ↓
summarize_timeframe(df_4h, "4h") → {trend: "bullish", text: "..."}
summarize_timeframe(df_1d, "1d") → {trend: "bullish", text: "..."}
  ↓
compute_confluence([...]) → "aligned"
  ↓
Inject "## Higher Timeframe Context" section into prompt
  ↓
Return enhanced prompt to workflow
```

### Timeframe Selection Logic

Given current timeframe and available higher timeframes:

1. Filter `available_tfs` to only those higher in hierarchy than `current_tf`
2. Sort by hierarchy position (nearest first)
3. Take first 2 timeframes
4. If 0 available → omit section entirely
5. If 1 available → use 1
6. If 2+ available → use 2 nearest

**Examples:**

- Current: "1m", Available: ["5m", "15m", "1h", "4h"] → Select: ["5m", "15m"]
- Current: "1h", Available: ["4h", "1d"] → Select: ["4h", "1d"]
- Current: "1h", Available: ["1d"] → Select: ["1d"]
- Current: "1d", Available: [] → Omit section

## Data Structures and Indicators

### Trend Summary Structure

`summarize_timeframe()` extracts these indicators from `compute_all_indicators()`:

**1. Ichimoku Cloud Position**
- Extract `ichimoku_cloud_bottom` and `ichimoku_cloud_top`
- Compare current price:
  - Above cloud (top) → "above" (bullish signal)
  - Below cloud (bottom) → "below" (bearish signal)
  - Inside cloud → "inside" (neutral signal)

**2. KAMA Trend**
- Extract `kama` series
- Compute slope: `kama.iloc[-1] - kama.iloc[-5]` (5-bar lookback)
- Threshold: 0.1% of current price
  - Slope > threshold → "rising" (bullish)
  - Slope < -threshold → "falling" (bearish)
  - Otherwise → "flat" (neutral)

**3. Donchian Channel Position**
- Extract `donchian_upper` and `donchian_lower`
- Compute percentile position: `(price - lower) / (upper - lower)`
  - Position > 0.8 → "upper" (near resistance)
  - Position < 0.2 → "lower" (near support)
  - Otherwise → "middle" (neutral)

**4. RSI Zone**
- Extract scalar `rsi` value
  - RSI > 70 → "overbought"
  - RSI < 30 → "oversold"
  - Otherwise → "neutral"

### Trend Classification

Combine cloud position and KAMA slope to determine overall trend:

- Cloud "above" AND KAMA "rising" → "bullish"
- Cloud "below" AND KAMA "falling" → "bearish"
- Cloud "above" AND KAMA "falling" → "neutral" (conflicting signals)
- Cloud "below" AND KAMA "rising" → "neutral" (conflicting signals)
- Cloud "inside" → "neutral" (regardless of KAMA)

### Return Format

`summarize_timeframe()` returns a dict:

```python
{
    "timeframe": "4h",
    "trend": "bullish",  # bullish | bearish | neutral
    "cloud_position": "above",  # above | below | inside
    "kama_slope": "rising",  # rising | falling | flat
    "donchian_position": "upper",  # upper | lower | middle
    "rsi_zone": "neutral",  # overbought | oversold | neutral
    "rsi_value": 58,  # actual RSI value
    "text": "4h: Bullish (above cloud, KAMA rising), near Donchian upper, RSI 58 (neutral)"
}
```

### Confluence Logic

Given list of higher timeframe summaries:

1. Extract trend from each: `[summary["trend"] for summary in summaries]`
2. Classify:
   - All trends same → "aligned" (e.g., all "bullish" or all "bearish")
   - All trends opposite current TF → "conflicting"
   - Mixed trends → "mixed"

**Confluence Text Format:**

- Aligned bullish: "Confluence: Aligned with higher timeframes (bullish)"
- Aligned bearish: "Confluence: Aligned with higher timeframes (bearish)"
- Aligned neutral: "Confluence: Higher timeframes neutral"
- Mixed: "Confluence: Mixed signals across timeframes"
- Conflicting: "Confluence: Conflicting with higher timeframe trend"

## Prompt Template Integration

### Template Modification

Each prompt template (DirectionPredictionPrompt, MomentumAssessmentPrompt, SupportResistancePrompt) will:

1. Add `higher_tf_context: str | None = None` parameter to `render()` method
2. Inject optional section after "## Market Data" and before "## Technical Indicators"
3. Only include section when `higher_tf_context` is provided

**Example Template Addition:**

```python
TEMPLATE = """/no_think

You are a quantitative trading analyst...

## Market Data
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}
Market Regime: {market_regime}
{higher_tf_section}
## Technical Indicators
RSI(14): {rsi:.2f}
...
"""

def render(self, ..., higher_tf_context: str | None = None):
    higher_tf_section = ""
    if higher_tf_context:
        higher_tf_section = f"\n## Higher Timeframe Context\n{higher_tf_context}\n"

    return self.TEMPLATE.format(
        ...,
        higher_tf_section=higher_tf_section,
    )
```

**Example Rendered Output:**

```
## Higher Timeframe Context
4h: Bullish (above cloud, KAMA rising), near Donchian upper, RSI 58 (neutral)
1d: Bullish (above cloud, KAMA rising), middle of Donchian, RSI 52 (neutral)
Confluence: Aligned with higher timeframes (bullish)
```

### PromptBuilder Changes

`build_prompt()` signature change:

```python
def build_prompt(
    self,
    task: TaskConfig,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    market_regime: MarketRegime,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
) -> str:
```

**Implementation steps:**

1. Check if `higher_tf_data` is provided and not empty
2. Call `get_higher_timeframes(timeframe, list(higher_tf_data.keys()))`
3. For each selected timeframe, call `summarize_timeframe(df, tf)`
4. Call `compute_confluence(summaries)`
5. Format the complete higher timeframe section:
   ```python
   lines = [summary["text"] for summary in summaries]
   lines.append(confluence_text)
   higher_tf_context = "\n".join(lines)
   ```
6. Pass `higher_tf_context=...` to `template.render()`

### Backward Compatibility

- All new parameters have `None` defaults
- Existing code continues to work without changes
- Single-timeframe workflows omit the section (clean prompts)
- No breaking changes to existing APIs

## Orchestrator Integration

### API Changes

`run_swarm_workflow()` signature change:

```python
async def run_swarm_workflow(
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    market_regime: MarketRegime,
    task_prompt: str,
    task_type: TaskType = TaskType.PREDICT_DIRECTION,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[SwarmState, TrainingExample]:
```

**Implementation:**

The orchestrator receives `higher_tf_data` and passes it directly to PromptBuilder:

```python
prompt = builder.build_prompt(
    task=task,
    df=ohlcv_df,
    symbol=symbol,
    timeframe=timeframe,
    market_regime=market_regime,
    higher_tf_data=higher_tf_data,  # Pass through
)
```

No other orchestrator changes needed - the enhanced prompt flows through the existing workflow transparently.

## Testing Strategy

### Unit Tests

**New test file:** `tests/test_prompt_builder_mtf.py`

**1. Test `get_higher_timeframes()`**

```python
def test_get_higher_timeframes_returns_2_nearest():
    result = get_higher_timeframes("1m", ["5m", "15m", "1h", "4h"])
    assert result == ["5m", "15m"]

def test_get_higher_timeframes_returns_1_when_only_1_available():
    result = get_higher_timeframes("1h", ["4h"])
    assert result == ["4h"]

def test_get_higher_timeframes_returns_empty_when_none_available():
    result = get_higher_timeframes("1d", ["1h", "4h"])
    assert result == []

def test_get_higher_timeframes_skips_unknown_timeframes():
    result = get_higher_timeframes("1h", ["3h", "4h", "1d"])
    assert result == ["4h", "1d"]  # "3h" not in hierarchy
```

**2. Test `summarize_timeframe()`**

```python
def test_summarize_timeframe_bullish_setup():
    # Create DataFrame with known bullish pattern
    df = create_test_df_bullish()  # Fixture
    result = summarize_timeframe(df, "4h")

    assert result["trend"] == "bullish"
    assert result["cloud_position"] == "above"
    assert result["kama_slope"] == "rising"
    assert "Bullish" in result["text"]

def test_summarize_timeframe_bearish_setup():
    df = create_test_df_bearish()
    result = summarize_timeframe(df, "1h")

    assert result["trend"] == "bearish"
    assert result["cloud_position"] == "below"
    assert result["kama_slope"] == "falling"

def test_summarize_timeframe_neutral_setup():
    df = create_test_df_neutral()
    result = summarize_timeframe(df, "1d")

    assert result["trend"] == "neutral"
    assert result["cloud_position"] == "inside"

def test_summarize_timeframe_rsi_zones():
    df_overbought = create_test_df(rsi=75)
    assert summarize_timeframe(df_overbought, "4h")["rsi_zone"] == "overbought"

    df_oversold = create_test_df(rsi=25)
    assert summarize_timeframe(df_oversold, "4h")["rsi_zone"] == "oversold"
```

**3. Test `compute_confluence()`**

```python
def test_compute_confluence_aligned_bullish():
    summaries = [
        {"trend": "bullish", "timeframe": "4h"},
        {"trend": "bullish", "timeframe": "1d"},
    ]
    result = compute_confluence(summaries)
    assert "aligned" in result.lower()
    assert "bullish" in result.lower()

def test_compute_confluence_mixed():
    summaries = [
        {"trend": "bullish", "timeframe": "4h"},
        {"trend": "bearish", "timeframe": "1d"},
    ]
    result = compute_confluence(summaries)
    assert "mixed" in result.lower()
```

**4. Test `PromptBuilder.build_prompt()` integration**

```python
def test_build_prompt_includes_higher_tf_section():
    builder = PromptBuilder()
    task = TaskConfig(TaskType.PREDICT_DIRECTION, ...)
    df = create_test_df()
    higher_tf_data = {
        "4h": create_test_df_bullish(),
        "1d": create_test_df_bullish(),
    }

    prompt = builder.build_prompt(
        task=task,
        df=df,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        higher_tf_data=higher_tf_data,
    )

    assert "## Higher Timeframe Context" in prompt
    assert "4h:" in prompt
    assert "1d:" in prompt
    assert "Confluence:" in prompt

def test_build_prompt_omits_section_when_no_data():
    builder = PromptBuilder()
    prompt = builder.build_prompt(..., higher_tf_data=None)

    assert "## Higher Timeframe Context" not in prompt
```

### Integration Tests

**Extend `tests/test_orchestrator.py`:**

```python
async def test_run_swarm_workflow_accepts_higher_tf_data():
    higher_tf_data = {
        "4h": create_test_df_bullish(),
        "1d": create_test_df_bullish(),
    }

    state, example = await run_swarm_workflow(
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_prompt="...",
        higher_tf_data=higher_tf_data,
    )

    # Verify higher TF context appears in captured prompt
    assert "Higher Timeframe Context" in example.prompt

async def test_run_swarm_workflow_backward_compatible():
    # Test without higher_tf_data parameter
    state, example = await run_swarm_workflow(
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv_df=df,
        market_regime=MarketRegime.NEUTRAL,
        task_prompt="...",
    )

    assert state["workflow_status"] == "success"
    assert "Higher Timeframe Context" not in example.prompt
```

### Test Fixtures

Create reusable fixtures in `tests/fixtures/`:

```python
def create_test_df_bullish(bars: int = 100) -> pd.DataFrame:
    """Create DataFrame with bullish indicator pattern."""
    # Price trending up, above Ichimoku cloud, KAMA rising, etc.
    ...

def create_test_df_bearish(bars: int = 100) -> pd.DataFrame:
    """Create DataFrame with bearish indicator pattern."""
    ...

def create_test_df_neutral(bars: int = 100) -> pd.DataFrame:
    """Create DataFrame with neutral indicator pattern."""
    ...
```

## Error Handling and Edge Cases

### Edge Case Handling

**1. Insufficient Data for Indicators**

If higher timeframe DataFrame lacks sufficient bars (e.g., <50 bars for Ichimoku):

- Behavior: Skip that timeframe entirely
- Logging: `logger.warning("Insufficient data", timeframe="4h", bars=30, required=50)`
- Impact: Continue with remaining higher timeframes

**2. Invalid Timeframe in Hierarchy**

If caller provides timeframe not in `TIMEFRAME_HIERARCHY` (e.g., "3h", "2d"):

- Behavior: Skip unknown timeframes
- Logging: `logger.warning("Skipping timeframe", timeframe="3h", reason="not in hierarchy")`
- Impact: Use only recognized timeframes

**3. No Higher Timeframes Available**

If current TF is "1d" (top of hierarchy) or no valid higher TFs after filtering:

- Behavior: Omit entire "Higher Timeframe Context" section
- Logging: `logger.debug("No higher timeframes available", current_tf="1d")`
- Impact: Prompt generation continues normally (single-timeframe mode)

**4. NaN/Missing Indicator Values**

If `compute_all_indicators()` returns NaN for critical indicators:

- Behavior: Use neutral fallback values
  - Cloud position: "inside" (neutral)
  - KAMA slope: "flat" (neutral)
  - Donchian position: "middle" (neutral)
  - RSI: 50 (neutral)
- Logging: `logger.warning("Missing indicator", timeframe="4h", indicator="kama")`
- Impact: Summary still generated with neutral classification

**5. Empty DataFrame**

If `higher_tf_data` contains empty DataFrames:

- Behavior: Skip that timeframe (treat as insufficient data)
- Logging: `logger.warning("Empty DataFrame", timeframe="4h")`
- Impact: Continue with other available timeframes

### Error Handling Principles

1. **Graceful Degradation:** Never fail prompt generation due to higher TF issues
2. **Fail-Safe Defaults:** Always prefer neutral classifications over errors
3. **Transparent Logging:** Log warnings for debugging but don't spam
4. **Data Validation:** Validate at entry point (`build_prompt`) before processing
5. **Type Safety:** Use type hints and runtime checks for `higher_tf_data` structure

### Validation Logic

At the start of `build_prompt()`:

```python
if higher_tf_data is not None:
    # Validate structure
    if not isinstance(higher_tf_data, dict):
        logger.warning("Invalid higher_tf_data type", type=type(higher_tf_data))
        higher_tf_data = None
    else:
        # Filter out invalid entries
        valid_data = {}
        for tf, df in higher_tf_data.items():
            if not isinstance(df, pd.DataFrame):
                logger.warning("Invalid DataFrame", timeframe=tf, type=type(df))
                continue
            if len(df) == 0:
                logger.warning("Empty DataFrame", timeframe=tf)
                continue
            if len(df) < 50:  # Minimum for Ichimoku
                logger.warning("Insufficient data", timeframe=tf, bars=len(df))
                continue
            valid_data[tf] = df

        higher_tf_data = valid_data if valid_data else None
```

## Implementation Checklist

### Phase 1: Core Functions (data/prompt_builder.py)

- [ ] Add `TIMEFRAME_HIERARCHY` constant
- [ ] Implement `get_higher_timeframes()`
- [ ] Implement `summarize_timeframe()`
- [ ] Implement `compute_confluence()`
- [ ] Add validation logic for `higher_tf_data`

### Phase 2: Template Updates (data/prompt_builder.py)

- [ ] Update `DirectionPredictionPrompt.render()` - add `higher_tf_context` parameter
- [ ] Update `MomentumAssessmentPrompt.render()` - add `higher_tf_context` parameter
- [ ] Update `SupportResistancePrompt.render()` - add `higher_tf_context` parameter
- [ ] Update template strings to include optional higher TF section

### Phase 3: PromptBuilder Integration (data/prompt_builder.py)

- [ ] Update `PromptBuilder.build_prompt()` signature
- [ ] Add higher TF processing logic
- [ ] Pass `higher_tf_context` to template renders
- [ ] Add comprehensive logging

### Phase 4: Orchestrator Integration (swarm/orchestrator.py)

- [ ] Update `run_swarm_workflow()` signature
- [ ] Pass `higher_tf_data` through to PromptBuilder
- [ ] Add integration logging

### Phase 5: Unit Tests (tests/test_prompt_builder_mtf.py)

- [ ] Test `get_higher_timeframes()` - all edge cases
- [ ] Test `summarize_timeframe()` - bullish/bearish/neutral patterns
- [ ] Test `summarize_timeframe()` - RSI zones
- [ ] Test `summarize_timeframe()` - error handling
- [ ] Test `compute_confluence()` - aligned/mixed/conflicting
- [ ] Test `PromptBuilder.build_prompt()` - section inclusion/omission
- [ ] Test `PromptBuilder.build_prompt()` - all 3 task types

### Phase 6: Integration Tests

- [ ] Extend `tests/test_orchestrator.py` - higher_tf_data acceptance
- [ ] Test backward compatibility (no higher_tf_data)
- [ ] Test end-to-end prompt flow

### Phase 7: Test Fixtures

- [ ] Create `create_test_df_bullish()` fixture
- [ ] Create `create_test_df_bearish()` fixture
- [ ] Create `create_test_df_neutral()` fixture

## Success Criteria

1. **Functionality:**
   - Prompts include higher TF context when data provided
   - Prompts omit section when no data provided
   - Up to 2 nearest higher timeframes selected correctly
   - Trend summaries accurate for known indicator patterns

2. **Backward Compatibility:**
   - All existing tests pass without modification
   - Single-timeframe workflows unchanged
   - No breaking API changes

3. **Test Coverage:**
   - All new functions have >90% coverage
   - Edge cases tested (empty data, NaN values, invalid TFs)
   - Integration tests verify end-to-end flow

4. **Code Quality:**
   - Type hints on all new functions
   - Comprehensive logging for debugging
   - Clear docstrings with examples

## Future Enhancements (Out of Scope)

These are explicitly NOT part of this design but could be considered later:

1. **Adaptive timeframe selection based on signal strength**
2. **Multi-timeframe backtesting in verifier layer**
3. **Timeframe-weighted reward signals**
4. **Auto-discovery of optimal timeframe combinations per symbol**
5. **Inter-timeframe divergence detection (e.g., RSI divergence across TFs)**

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Prompt length bloat | Token costs increase | Cap at 2 TFs, concise summaries |
| Indicator computation overhead | Slower prompt generation | Use `compute_all_indicators()` (already optimized) |
| Test complexity | Harder to maintain | Reusable fixtures with clear patterns |
| Backward compatibility break | Existing code fails | Optional parameters with None defaults |
| NaN handling bugs | Prompt generation fails | Graceful degradation, neutral fallbacks |

## Estimated Impact

**Code Changes:**
- Modified files: 2 (prompt_builder.py, orchestrator.py)
- New test file: 1 (test_prompt_builder_mtf.py)
- Lines added: ~200-250
- Lines modified: ~50

**Testing:**
- New unit tests: ~15-20
- New integration tests: ~3-5
- Test fixtures: 3

**Performance:**
- Prompt generation: +10-20ms (2x `compute_all_indicators()` calls)
- Prompt length: +150-200 tokens per prompt
- Memory: Negligible (no additional DataFrame storage)

## References

- Existing code: `data/prompt_builder.py:461-639`
- Existing code: `data/indicators.py:1-100` (compute_all_indicators)
- Existing code: `swarm/orchestrator.py:137-150` (run_swarm_workflow)
- Project memory: `CLAUDE.md` - Indicator System (Session 11b)
