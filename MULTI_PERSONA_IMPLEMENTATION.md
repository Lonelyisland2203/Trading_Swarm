# Multi-Persona Signal Generation for DPO Training

**Date**: 2026-04-03
**Status**: ✅ Complete
**Tests**: All 316 tests passing

## Overview

Modified the orchestrator to enable **cross-persona signal generation** - generating multiple signals from different trading personas for the **same market context**. This is a critical prerequisite for DPO (Direct Preference Optimization) training, which requires preference pairs constructed from signals generated with the same prompt.

## Problem Statement

### The DPO Requirement

DPO training requires preference pairs where:
1. Both signals are generated from the **SAME prompt** (same market context)
2. Signals are ranked by their realized performance (rewards)
3. The model learns: "given this context, prefer signal A over signal B"

### Previous Architecture Gap

The original `run_swarm_workflow()` generated **ONE signal per context**:
- Randomly samples a single persona
- Generates one signal for that persona
- No way to construct valid same-prompt preference pairs

**Root cause analysis** (by root-cause-engineer agent):
> "Within-batch ranking creates spurious pairs from unrelated contexts. BTC in bull market vs ETH in bear market is not a valid preference pair - different contexts require different strategies."

## Solution: Cross-Persona Generation

Generate signals from **ALL 5 personas** for the same market context:

1. **CONTRARIAN** - Profits from market overreactions
2. **MOMENTUM** - Rides established trends
3. **MEAN_REVERSION** - Expects return to average
4. **BREAKOUT** - Capitalizes on range breakouts
5. **CONSERVATIVE** - Risk-averse capital preservation

All personas receive:
- Identical `task_prompt` (market analysis question)
- Identical `market_context` (indicators, regime, price data)
- Same timestamp and symbol

This enables valid preference pair construction by ranking signals by computed rewards.

## Implementation

### File 1: `swarm/generator.py`

**Modified**: Added `persona_override` parameter to `generate_signal()`

```python
async def generate_signal(
    client: OllamaClient,
    model: str,
    prompt: str,
    regime: MarketRegime,
    temperature: float = 0.7,
    seed: int | None = None,
    persona_override: TradingPersona | None = None,  # NEW PARAMETER
) -> GeneratorSignal | None:
    """
    Generate trading signal with persona-enhanced prompt.

    Args:
        persona_override: Optional specific persona (for cross-persona generation)
    """
    # Sample persona or use override for cross-persona generation
    persona = persona_override if persona_override is not None else sample_persona(regime, seed=seed)
```

**Impact**:
- Backwards compatible (default behavior unchanged)
- When `persona_override` is provided, uses that persona instead of sampling

### File 2: `swarm/orchestrator.py`

**Added**: New function `run_multi_persona_workflow()`

```python
async def run_multi_persona_workflow(
    symbol: str,
    timeframe: str,
    ohlcv_df: pd.DataFrame,
    market_regime: MarketRegime,
    task_prompt: str,
) -> tuple[dict, list[TrainingExample]]:
    """
    Generate signals from ALL personas for the same context (DPO training).

    Returns:
        Tuple of (summary_state, training_examples):
        - summary_state: Workflow summary with counts
        - training_examples: List of TrainingExample (one per successful generation)
    """
```

**Workflow**:

1. **Build market context once** (shared by all personas)
   ```python
   market_context = _build_market_context(ohlcv_df, market_regime)
   timestamp_ms = int(ohlcv_df["timestamp"].iloc[-1])
   ```

2. **Loop through all 5 personas**
   ```python
   for persona in TradingPersona:
       signal = await generate_signal(
           client, model, task_prompt, market_regime,
           persona_override=persona  # Force this persona
       )
       await client.unload_current()  # VRAM safety
   ```

3. **Critique each signal independently**
   ```python
   critique = await evaluate_signal(
       client, settings.ollama.critic_model,
       asdict(signal), market_context, task_prompt
   )
   await client.unload_current()  # VRAM safety
   ```

4. **Apply acceptance decision per signal**
   ```python
   accepted, reason = should_accept_signal(asdict(critique), market_regime)
   training_example.was_accepted = accepted
   ```

5. **Return all training examples** (both accepted and rejected)
   ```python
   return summary, training_examples  # List of 0-5 examples
   ```

**Graceful Degradation**:
- Generator failure for a persona → skip, continue with others
- Critic failure → accept signal with flag
- All generators failed → return empty list

**VRAM Safety**:
- Explicit `unload_current()` after each generator/critic call
- RTX 5070 Ti (16 GB) constraint: never load both models simultaneously
- `keep_alive=0` setting ensures immediate model unload

### Summary State

```python
{
    "workflow_status": "success" | "partial_success" | "all_failed",
    "personas_attempted": 5,
    "signals_generated": 0-5,  # Number of successful generations
    "signals_accepted": 0-5,   # Number passing acceptance threshold
    "errors": ["persona_name: error_message", ...]
}
```

### Training Examples

Each `TrainingExample` contains:
- `generator_signal`: Signal dict with persona field
- `critique`: Critic evaluation (if successful)
- `was_accepted`: Boolean acceptance decision
- `acceptance_reason` or `rejection_reason`: Why this decision

**All examples are returned** (both accepted and rejected) because DPO needs both for preference pairs.

## Backwards Compatibility

**No breaking changes**:
- Existing `run_swarm_workflow()` untouched
- New `run_multi_persona_workflow()` is additive
- Existing tests all pass (316/316)

**Migration path**:
- Session 8 (DPO Fine-Tuning) will use `run_multi_persona_workflow()`
- Legacy inference can continue using `run_swarm_workflow()`

## Usage Example

```python
# Multi-persona generation for DPO training
summary, training_examples = await run_multi_persona_workflow(
    symbol="BTC/USDT",
    timeframe="1h",
    ohlcv_df=df,
    market_regime=MarketRegime.NEUTRAL,
    task_prompt=prompt,
)

print(f"Generated {summary['signals_generated']} signals")
print(f"Accepted {summary['signals_accepted']} signals")

# Verify outcomes and compute rewards
verified_outcomes = [verify_example(ex, df) for ex in training_examples]
rewards = [compute_reward(outcome, ex) for outcome, ex in zip(verified_outcomes, training_examples)]

# Create preference pairs by ranking
sorted_pairs = sorted(
    zip(training_examples, rewards),
    key=lambda x: x[1].final_reward,
    reverse=True
)

chosen_example = sorted_pairs[0][0]   # Best signal
rejected_example = sorted_pairs[-1][0]  # Worst signal

# Both have same task_prompt → valid DPO preference pair
```

## Next Steps: Session 8 (DPO Fine-Tuning)

With cross-persona generation now available:

1. **Preference Pair Construction**
   - Rank signals by `ComputedReward.final_reward`
   - Construct (chosen, rejected) pairs from same context
   - Filter pairs with minimum reward delta (e.g., Δ > 0.2)

2. **DPO Dataset Format**
   - Convert to HuggingFace format:
     ```python
     {
         "prompt": task_prompt,
         "chosen": chosen_signal.reasoning,
         "rejected": rejected_signal.reasoning,
     }
     ```

3. **Training Pipeline**
   - Use unsloth for 4-bit QLoRA fine-tuning
   - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
   - Separate training environment (Process B)

4. **Validation**
   - Walk-forward validation
   - Out-of-sample IC evaluation
   - Calibration analysis

## Testing

**Status**: All 316 tests passing

**Modified tests**: None (backwards compatible)

**New tests needed** (Session 8):
- Test multi-persona workflow with mocked Ollama
- Test preference pair construction
- Test graceful degradation (partial failures)
- Test VRAM safety (model unloading)

## Architecture Decisions

### ✅ Decision 1: Separate function vs modify existing
- **Chosen**: Create new `run_multi_persona_workflow()`
- **Rationale**: Preserves backwards compatibility, clearer intent

### ✅ Decision 2: Return all examples vs only accepted
- **Chosen**: Return all examples (both accepted and rejected)
- **Rationale**: DPO needs both for preference pairs

### ✅ Decision 3: VRAM management strategy
- **Chosen**: Explicit `unload_current()` after each model call
- **Rationale**: RTX 5070 Ti (16 GB) constraint, defensive programming

### ✅ Decision 4: Error handling strategy
- **Chosen**: Skip failed personas, continue with others
- **Rationale**: Graceful degradation - partial data better than no data

## Performance Considerations

**Time complexity**:
- Single-persona: 1 generator call + 1 critic call
- Multi-persona: 5 generator calls + 5 critic calls
- **~5x slower** but necessary for DPO training

**VRAM usage**:
- Sequential model loading (never simultaneous)
- Peak: 8 GB (Qwen3-8B) or 14 GB (DeepSeek-R1-14B)
- Well within 16 GB constraint

**Optimization opportunities** (future):
- Batch generation with single model load (requires Ollama batch API)
- Parallel critique (critic is smaller model)
- Cached persona prompts

## Files Modified

1. **swarm/generator.py** (+1 parameter)
   - Added `persona_override` to `generate_signal()`
   - Backwards compatible

2. **swarm/orchestrator.py** (+211 lines)
   - Imported `TradingPersona`
   - Added `run_multi_persona_workflow()`

**Total additions**: 212 lines of production code

## Verification

```bash
# All tests pass
source venv/bin/activate
python -m pytest tests/ -v
# ===== 316 passed, 9 warnings in 15.89s =====
```

**Ready for Session 8**: ✅
