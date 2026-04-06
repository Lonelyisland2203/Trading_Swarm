# Fee-Aware DPO Training Design

**Date:** 2026-04-06
**Status:** Design
**Goal:** Integrate realistic fee model into DPO training pipeline to teach model fee-aware signal generation

---

## Problem Statement

The current DPO training pipeline optimizes signals based on **gross returns** (before trading costs), but production trading requires **net returns** (after fees, funding, slippage). This misalignment causes the model to:

1. Generate signals with positive gross returns but negative net returns
2. Ignore the fee hurdle that makes small-margin trades unprofitable
3. Fail to learn that longer timeframes have higher funding costs

**Example of the problem:**
- Signal predicts +0.08% gross return
- Binance Futures round-trip cost: 0.093% (maker entry, taker exit, 1 funding period)
- Net return: +0.08% - 0.093% = **-0.013%** (losing trade)
- Current system: Rewards this signal positively (gross > 0)
- Desired system: Penalizes this signal (net < 0)

---

## Solution Overview

Implement **teaching + enforcement** pattern:

1. **Teaching (Prompts):** Add "Execution Context" section showing fee structure
2. **Enforcement (Rewards):** Use net returns for reward computation and preference ranking
3. **Diagnostic (Pipeline):** Report fee flip rate to detect if model accuracy is below fee threshold

---

## Architecture

### Stage 1: Prompt Generation (Teaching)

**File:** `data/prompt_builder.py`

**Change:** Add "## Execution Context" section to all three prompt templates (DirectionPredictionPrompt, MomentumAssessmentPrompt, SupportResistancePrompt).

**New section format:**
```
## Execution Context
Exchange: Binance
Mode: Futures USDT-M
Estimated round-trip cost: 0.093%
Minimum profitable move: 0.093%

Your prediction must account for these costs. Signals with expected moves smaller
than the minimum profitable threshold should be rated LOW CONFIDENCE regardless
of directional conviction.
```

**Implementation details:**
- `PromptBuilder.build_prompt()` receives optional `fee_model: FeeModelSettings` parameter
- If fee_model provided, compute:
  - `round_trip_cost_pct = fee_model.round_trip_cost_pct(holding_periods_8h)`
  - `min_profitable_pct = fee_model.minimum_profitable_return_pct(holding_periods_8h)`
  - `holding_periods_8h = compute_holding_periods_8h(timeframe, horizon_bars)`
- Mode determined from fee_model type:
  - FeeModelSettings with default params → "Futures USDT-M"
  - FeeModelSettings with custom spot params → "Spot"
  - None → section omitted (backward compatible)
- Format values to 3 decimal places (e.g., "0.093%")

**Backward compatibility:**
- If `fee_model=None`, section is omitted entirely
- Existing code continues to work without changes

---

### Stage 2: Reward Computation (Enforcement)

**File:** `training/reward_engine.py`

**Change:** Switch `compute_reward()` to use `net_return` instead of `realized_return`.

**Current implementation:**
```python
return_reward = compute_return_reward(
    verified_outcome.realized_return,  # GROSS return
    scaling.return_scale,
)
```

**New implementation:**
```python
return_reward = compute_return_reward(
    verified_outcome.net_return,  # NET return (after fees)
    scaling.return_scale,
)
```

**Impact:**
- Examples with positive gross but negative net will get negative return_reward
- Preference pairs will rank by net profitability, not gross profitability
- No API changes needed (net_return already computed by verifier)

**Already implemented (Session 11a):**
- `VerifiedOutcome.net_return` field exists
- `verifier/engine.py` computes net_return using fee_model
- `verifier/outcome.py` has `apply_fee_model()` helper

---

### Stage 3: Preference Pair Construction

**File:** `training/dpo_export.py`

**Change:** Ensure ranking uses NET rewards (already correct by design).

**Current behavior (already correct):**
```python
sorted_group = sorted(
    group,
    key=lambda x: x[2].final_reward,  # ComputedReward.final_reward
    reverse=True,
)
```

Since `ComputedReward.final_reward` is derived from `net_return` (after Stage 2 change), preference pairs will automatically rank by net profitability.

**Validation test:**
Construct two examples with:
- Example A: +0.30% gross → +0.207% net (profitable after fees)
- Example B: +0.08% gross → -0.013% net (unprofitable after fees)

Verify:
- Example A gets positive reward
- Example B gets negative reward
- Preference pair selects A as "chosen", B as "rejected"

---

### Stage 4: Fee Flip Diagnostic

**File:** `run_dpo_training.py`

**New function:** `compute_fee_flip_diagnostic(examples_and_outcomes, fee_model)`

**Purpose:** Report how many training examples flip from positive to negative reward once fees are applied, grouped by timeframe.

**Output format:**
```
=== FEE FLIP DIAGNOSTIC ===
Examples that were profitable under flat 0.1% fees but are unprofitable
with realistic Binance Futures fees:

Timeframe | Total Examples | Flipped to Negative | Flip Rate | Avg Old Net | Avg New Net
----------|----------------|---------------------|-----------|-------------|------------
       1m |            450 |                  12 |      2.7% |     +0.12% |     +0.09%
       1h |            450 |                  45 |     10.0% |     +0.15% |     +0.08%
       1d |            450 |                 180 |     40.0% |     +0.11% |     -0.03%
----------|----------------|---------------------|-----------|-------------|------------
    TOTAL |           1350 |                 237 |     17.6% |            |

1d funding cost alone: 0.05% (5 periods × 0.01%)

WARNING: 1d timeframe has 40.0% flip rate - signals may not clear fee hurdle.
Consider focusing training on longer timeframes or increasing signal selectivity.
```

**Thresholds:**
- Flip rate > 15%: Print warning for that timeframe
- Suggests either:
  1. Model accuracy is below fee threshold → need better signals
  2. Timeframe is too short for this fee structure → focus on longer TFs

**Integration point:** Called in `phase3_reward()` after reward computation.

---

### Stage 5: CLI Integration

**File:** `run_dpo_training.py`

**New CLI flag:**
```python
parser.add_argument(
    "--fee-mode",
    type=str,
    default="futures_usdt",
    choices=["futures_usdt", "spot", "none"],
    help="Fee model mode: futures_usdt (0.02%/0.05% maker/taker + funding), "
         "spot (0.10% maker/taker, no funding), none (legacy 0.1% flat)"
)
```

**Fee model instantiation:**
```python
if args.fee_mode == "futures_usdt":
    fee_model = FeeModelSettings()  # Default: Binance Futures
elif args.fee_mode == "spot":
    fee_model = FeeModelSettings(
        maker_fee_pct=0.10,
        taker_fee_pct=0.10,
        entry_order_type="taker",
        exit_order_type="taker",
        bnb_discount_enabled=False,
        include_funding=False,
        slippage_pct=0.05,
    )
elif args.fee_mode == "none":
    fee_model = None  # Legacy behavior
```

**Propagation:**
- Pass `fee_model` to `phase2_verify()`
- Pass `fee_model` to `phase3_reward()` for diagnostic
- Pass `fee_model` to `phase1_load()` → ... → `PromptBuilder.build_prompt()`

**Backward compatibility:**
- `--fee-mode none` preserves exact legacy behavior
- Enables ablation studies (compare fee-aware vs fee-naive training)

---

## Data Flow

```
generate_training_dataset.py
  ↓
phase1_prepare_contexts()
  ├─ fee_model = FeeModelSettings(mode=args.fee_mode)  # NEW
  ├─ prompt_builder.build_prompt(..., fee_model=fee_model)  # NEW
  └─ Prompts now include "Execution Context" section
  ↓
run_dpo_training.py --fee-mode futures_usdt
  ↓
phase2_verify(fee_model=fee_model)
  └─ VerifiedOutcome.net_return computed using fee_model
  ↓
phase3_reward(fee_model=fee_model)
  ├─ compute_reward() uses net_return (not realized_return)  # CHANGED
  ├─ ComputedReward.final_reward based on net profitability
  └─ compute_fee_flip_diagnostic() reports flip rate  # NEW
  ↓
phase4_pairs()
  └─ Preference pairs ranked by net rewards (automatic)
  ↓
phase5_train()
  └─ DPO training optimizes for net profitability
```

---

## Component Design

### 1. PromptBuilder Extension

**Method signature change:**
```python
def build_prompt(
    self,
    task: TaskConfig,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    market_regime: MarketRegime,
    higher_tf_data: dict[str, pd.DataFrame] | None = None,
    fee_model: FeeModelSettings | None = None,  # NEW
) -> str:
```

**New helper method:**
```python
def _build_execution_context(
    self,
    timeframe: str,
    horizon_bars: int,
    fee_model: FeeModelSettings,
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

**Template integration:**
All three templates add `{execution_context}` placeholder:
```python
TEMPLATE = """/no_think

You are a quantitative trading analyst...

## Market Data
...
{execution_context}
## Technical Indicators
...
"""
```

**Rendering:**
```python
execution_context = ""
if fee_model is not None:
    horizon_bars = get_horizon_bars(timeframe)  # From task_type
    execution_context = "\n" + self._build_execution_context(
        timeframe, horizon_bars, fee_model
    ) + "\n"

prompt = template.render(
    ...,
    execution_context=execution_context,
)
```

---

### 2. generate_training_dataset.py Extension

**New CLI flag:**
```python
parser.add_argument(
    "--fee-mode",
    type=str,
    default="futures_usdt",
    choices=["futures_usdt", "spot", "none"],
    help="Fee model for execution context in prompts"
)
```

**Fee model creation:**
```python
def create_fee_model(mode: str) -> FeeModelSettings | None:
    """Create fee model based on mode string."""
    if mode == "futures_usdt":
        return FeeModelSettings()  # Defaults
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
        return None
    else:
        raise ValueError(f"Unknown fee mode: {mode}")
```

**Integration in phase1_prepare_contexts():**
```python
async def phase1_prepare_contexts(config: DatasetConfig) -> list[InferenceJob]:
    ...
    fee_model = create_fee_model(config.fee_mode)  # NEW
    prompt_builder = PromptBuilder()

    for window, df in ...:
        ...
        task_prompt = prompt_builder.build_prompt(
            task=task_config,
            df=df,
            symbol=window.symbol,
            timeframe=window.timeframe,
            market_regime=market_regime,
            fee_model=fee_model,  # NEW
        )
```

**DatasetConfig extension:**
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

---

### 3. Fee Flip Diagnostic Implementation

**Location:** `run_dpo_training.py`

**Function signature:**
```python
def compute_fee_flip_diagnostic(
    examples_and_outcomes: list[tuple[TrainingExample, VerifiedOutcome]],
    fee_model: FeeModelSettings,
) -> None:
    """
    Print diagnostic showing examples that flip from positive to negative
    under realistic fees, grouped by timeframe.
    """
```

**Algorithm:**
1. Group examples by timeframe
2. For each example:
   - Convert log return to percentage: `gross_pct = (exp(realized_return) - 1) * 100`
   - Compute old net (legacy): `old_net_pct = gross_pct - 0.1`
   - Compute new net (realistic): `new_net_pct = fee_model.net_return(gross_pct, holding_periods)`
   - Check if flipped: `old_net_pct > 0 and new_net_pct < 0`
3. Compute flip rate per timeframe
4. Print table with warnings

**Constants:**
```python
FEE_FLIP_WARNING_THRESHOLD = 0.15  # 15% flip rate triggers warning
```

---

## Testing Strategy

### Test 1: Execution Context Rendering

**File:** `tests/test_prompt_builder_fee_context.py`

**Test cases:**
1. `test_execution_context_futures_mode()` - Verify Futures USDT-M section
2. `test_execution_context_spot_mode()` - Verify Spot section
3. `test_execution_context_none_mode()` - Verify section omitted
4. `test_execution_context_dynamic_costs()` - Verify costs vary by timeframe
5. `test_execution_context_in_all_templates()` - Verify all 3 templates include section

**Example:**
```python
def test_execution_context_futures_mode():
    builder = PromptBuilder()
    fee_model = FeeModelSettings()  # Futures defaults

    df = create_test_df(bars=100)
    task = TaskConfig(TaskType.PREDICT_DIRECTION, ...)

    prompt = builder.build_prompt(
        task=task,
        df=df,
        symbol="BTC/USDT",
        timeframe="1h",
        market_regime=MarketRegime.NEUTRAL,
        fee_model=fee_model,
    )

    assert "## Execution Context" in prompt
    assert "Mode: Futures USDT-M" in prompt
    assert "Estimated round-trip cost: 0.093%" in prompt
    assert "Minimum profitable move: 0.093%" in prompt
```

---

### Test 2: Net Reward Computation

**File:** `tests/test_reward_net_returns.py`

**Test cases:**
1. `test_reward_uses_net_return()` - Verify net_return used, not realized_return
2. `test_profitable_after_fees()` - Example with +0.30% gross → positive reward
3. `test_unprofitable_after_fees()` - Example with +0.08% gross → negative reward
4. `test_fee_model_none_uses_gross()` - Legacy mode uses realized_return

**Example:**
```python
def test_unprofitable_after_fees():
    # Example with +0.08% gross, unprofitable after 0.093% fees
    outcome = VerifiedOutcome(
        realized_return=0.0008,  # +0.08% gross (log return ≈ percentage)
        net_return=-0.00013,     # -0.013% net (after fees)
        actual_direction="HIGHER",
        ...
    )

    example = TrainingExample(
        generator_signal={"direction": "HIGHER", "confidence": 0.8, ...},
        ...
    )

    reward = compute_reward(outcome, example)

    # Return component should be negative (net < 0)
    assert reward.return_reward < 0
    # Final reward likely negative (depends on directional/MAE)
    # This signal should be penalized despite correct direction
```

---

### Test 3: Preference Pair Ranking

**File:** `tests/test_dpo_export_net_ranking.py`

**Test case:**
```python
def test_preference_pairs_rank_by_net_reward():
    """Verify chosen example has higher NET reward, not gross."""
    # Example A: +0.30% gross → +0.207% net (profitable)
    example_a = TrainingExample(...)
    outcome_a = VerifiedOutcome(net_return=0.00207, ...)
    reward_a = compute_reward(outcome_a, example_a)  # Positive

    # Example B: +0.08% gross → -0.013% net (unprofitable)
    example_b = TrainingExample(...)
    outcome_b = VerifiedOutcome(net_return=-0.00013, ...)
    reward_b = compute_reward(outcome_b, example_b)  # Negative

    # Same context_id for valid pairing
    example_a.context_id = "BTC_USDT_1h_12345_predict_direction"
    example_b.context_id = "BTC_USDT_1h_12345_predict_direction"

    pairs = construct_preference_pairs(
        [(example_a, outcome_a, reward_a), (example_b, outcome_b, reward_b)],
        min_delta=0.1,
    )

    assert len(pairs) == 1
    pair = pairs[0]

    # Example A should be chosen (higher net reward)
    assert pair.chosen_example_id == example_a.example_id
    assert pair.rejected_example_id == example_b.example_id
    assert pair.reward_delta > 0
```

---

### Test 4: Fee Flip Diagnostic

**File:** `tests/test_fee_flip_diagnostic.py`

**Test cases:**
1. `test_fee_flip_detection()` - Verify flip detection logic
2. `test_fee_flip_grouping_by_timeframe()` - Verify timeframe grouping
3. `test_fee_flip_warning_threshold()` - Verify warning triggers at 15%

---

## Error Handling

### Invalid fee_mode
```python
if args.fee_mode not in ["futures_usdt", "spot", "none"]:
    logger.error("Invalid fee mode", fee_mode=args.fee_mode)
    sys.exit(1)
```

### Missing fee model in verification
```python
if fee_model is None and args.fee_mode != "none":
    logger.warning("Fee model is None but fee_mode is not 'none'")
    # Fall back to no fee adjustment
```

### Horizon bars not found
```python
try:
    horizon_bars = get_horizon_bars(timeframe)
except KeyError:
    logger.warning("Unknown timeframe for horizon", timeframe=timeframe)
    horizon_bars = 5  # Default fallback
```

---

## Backward Compatibility

All changes are **backward compatible** by design:

1. **PromptBuilder:** `fee_model=None` (default) omits Execution Context section
2. **RewardEngine:** Change is internal (net vs gross), no API change
3. **CLI:** `--fee-mode none` preserves exact legacy behavior
4. **Dataset generation:** Can regenerate old datasets with `--fee-mode none`

**Migration path:**
1. Regenerate training datasets with `--fee-mode futures_usdt`
2. Run DPO training with `--fee-mode futures_usdt`
3. Compare performance vs legacy with `--fee-mode none` (ablation)

---

## Success Criteria

1. **Execution Context appears in prompts** - All generated prompts contain fee structure
2. **Net rewards computed correctly** - Reward engine uses net_return, not realized_return
3. **Preference pairs rank by net** - Chosen examples have higher net reward
4. **Fee flip diagnostic works** - Reports flip rate grouped by timeframe
5. **Tests pass** - All 4 test suites pass with 100% coverage
6. **Backward compatible** - `--fee-mode none` produces identical results to current system

---

## Future Enhancements

1. **Dynamic fee model per symbol** - Different exchanges have different fees
2. **Time-varying funding rates** - Use historical funding data instead of fixed 0.01%
3. **Position sizing integration** - Teach model to suggest smaller positions for low-confidence signals
4. **Fee-adjusted Sharpe ratio** - Evaluate strategy performance using net returns
5. **Adaptive confidence thresholds** - Learn optimal confidence cutoff based on fee structure

---

## References

- Session 11a: Realistic Fee Model Implementation
- Session 10: End-to-End DPO Workflow
- Session 9: Dataset Generation Pipeline
- `config/fee_model.py` - FeeModelSettings implementation
- `verifier/outcome.py` - apply_fee_model() implementation
