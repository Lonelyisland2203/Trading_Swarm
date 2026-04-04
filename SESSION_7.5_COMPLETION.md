# Session 7.5: Multi-Persona Generation + DPO Export Utilities

**Date**: 2026-04-03
**Status**: ✅ Complete
**Tests**: 330 tests passing (316 original + 14 new)

## Overview

Implemented cross-persona signal generation and DPO preference pair construction utilities. This completes the prerequisite infrastructure for Session 8 (DPO Fine-Tuning).

## Deliverables

### 1. Multi-Persona Signal Generation

**Files Modified**:
- `swarm/generator.py` (+1 line)
  - Added `persona_override` parameter to `generate_signal()`

- `swarm/orchestrator.py` (+214 lines)
  - Added `run_multi_persona_workflow()` function
  - Generates signals from all 5 personas for same context

- `swarm/training_capture.py` (+2 fields)
  - Added `context_id` field (for DPO pairing)
  - Added `critic_error` field (for error tracking)

**Key Features**:
- Generates signals from ALL 5 personas (CONTRARIAN, MOMENTUM, MEAN_REVERSION, BREAKOUT, CONSERVATIVE)
- All personas receive identical `task_prompt` and `market_context`
- VRAM-safe with explicit model unloading between personas
- Graceful degradation (partial failures don't break workflow)
- Returns summary with counts and all training examples

### 2. DPO Export Utilities

**Files Created**:
- `training/dpo_export.py` (341 lines)
  - `construct_preference_pairs()` - Rank signals by rewards, create (chosen, rejected) pairs
  - `validate_preference_pair()` - Ensure same context, sufficient reward delta
  - `export_to_huggingface_format()` - Convert to HF DPO dataset format
  - `export_to_jsonl()` - Export pairs to JSONL with optional metadata

- `tests/test_training/test_dpo_export.py` (399 lines, 14 tests)
  - Test reward delta computation
  - Test preference pair validation (context ID, reward ordering, delta threshold)
  - Test pair construction from multi-persona signals
  - Test HuggingFace format export

**Preference Pair Construction Logic**:
```python
1. Group examples by context_id
2. Filter contexts with < min_personas_per_context
3. Rank signals within each context by reward (descending)
4. Create pairs: best vs worst, 2nd-best vs 2nd-worst, etc.
5. Validate each pair (same context, delta >= min_delta)
6. Export to HuggingFace format
```

**HuggingFace DPO Format**:
```json
{
    "prompt": "<market analysis task>",
    "chosen": "<reasoning for better signal>",
    "rejected": "<reasoning for worse signal>"
}
```

## Architecture Validation (root-cause-engineer)

### ✅ Issues Fixed

1. **Missing `persona` field in multi-persona workflow**
   - Fixed: Now sets `training_example.persona = persona.value`

2. **Missing `critic_error` field on TrainingExample dataclass**
   - Fixed: Added `critic_error: str | None = None` field

3. **Missing `context_id` for DPO pairing**
   - Fixed: Added `context_id` field and set shared UUID per workflow run

4. **Missing `full_generator_prompt`**
   - Fixed: Set `full_generator_prompt = task_prompt`

### ✅ VRAM Safety Verified

- Sequential model loading (never simultaneous)
- Explicit `unload_current()` after each generator/critic call
- Semaphore enforcement (max 1 concurrent model)
- `keep_alive=0` hard validation
- Defensive unload on exception

### ✅ Same-Prompt Requirement Verified

- All personas receive identical `task_prompt`
- Market context built once and shared
- Timestamp captured once
- Persona difference only in system prompt, not task prompt

### ✅ Preference Pair Validity Verified

- Validation enforces same `context_id`
- Validation enforces minimum reward delta
- Validation enforces correct reward ordering
- Examples without `context_id` are skipped

## Test Coverage

**Total**: 330 tests (316 original + 14 new)

**New DPO Export Tests**:
- `test_compute_reward_delta` (3 tests)
- `test_validate_preference_pair` (5 tests)
- `test_construct_preference_pairs` (4 tests)
- `test_export_to_huggingface_format` (2 tests)

**Test Quality**:
- Edge cases: empty context_id, mismatched IDs, insufficient delta
- Multiple contexts handled separately
- Floating point precision issues addressed
- Persona diversity validation

## Usage Example

```python
# Multi-persona generation
summary, training_examples = await run_multi_persona_workflow(
    symbol="BTC/USDT",
    timeframe="1h",
    ohlcv_df=df,
    market_regime=MarketRegime.NEUTRAL,
    task_prompt=prompt,
)

# Verify outcomes and compute rewards
verified_outcomes = [verify_example(ex, df) for ex in training_examples]
examples_with_rewards = [
    (ex, outcome, compute_reward(outcome, ex))
    for ex, outcome in zip(training_examples, verified_outcomes)
]

# Construct preference pairs
pairs = construct_preference_pairs(
    examples_with_rewards,
    min_delta=0.2,
    min_personas_per_context=3,
)

# Export to HuggingFace format
hf_dataset = export_to_huggingface_format(pairs)

# Save to JSONL
export_to_jsonl(pairs, "outputs/dpo_pairs.jsonl")
```

## Backwards Compatibility

**No breaking changes**:
- Existing `run_swarm_workflow()` untouched
- New functions are additive
- All 316 original tests still pass

**Migration path**:
- Session 8 will use `run_multi_persona_workflow()`
- Legacy inference can continue using single-persona workflow

## Performance Characteristics

**Time Complexity**:
- Single-persona: 1 generator + 1 critic call
- Multi-persona: 5 generator + 5 critic calls
- **~5x slower** but necessary for DPO training

**VRAM Usage**:
- Sequential loading (never simultaneous)
- Peak: 14 GB (DeepSeek-R1-14B critic)
- Well within 16 GB RTX 5070 Ti constraint

**Pair Construction**:
- O(N log N) for sorting (N = examples per context)
- Typical: 5 personas → 2 pairs per context

## Next Session: Session 8 (DPO Fine-Tuning)

**Prerequisites**: ✅ Complete

With multi-persona generation and DPO export utilities now available:

1. **Training Pipeline**
   - LoRA configuration (target modules, rank, alpha)
   - Unsloth 4-bit QLoRA setup
   - DPO trainer with gradient accumulation
   - Separate training environment (Process B)

2. **Dataset Preparation**
   - Collect multi-persona signals
   - Verify outcomes with backtesting
   - Compute rewards
   - Construct preference pairs
   - Export to HuggingFace dataset

3. **Training Infrastructure**
   - Walk-forward validation splits
   - Checkpoint management
   - VRAM monitoring
   - Training metrics logging

4. **Evaluation**
   - Out-of-sample IC validation
   - Calibration analysis
   - Model comparison (pre/post DPO)

## Files Modified Summary

| File | Lines Changed | Type |
|------|---------------|------|
| `swarm/generator.py` | +1 | Parameter addition |
| `swarm/orchestrator.py` | +214 | New function |
| `swarm/training_capture.py` | +2 | Field additions |
| `training/dpo_export.py` | +341 | New file |
| `tests/test_training/test_dpo_export.py` | +399 | New tests |
| **Total** | **+957** | **5 files** |

## Documentation

- `MULTI_PERSONA_IMPLEMENTATION.md` - Implementation details
- `SESSION_7.5_COMPLETION.md` - This file
- `CLAUDE.md` - Updated with session completion
- Root-cause-engineer diagnostic report - Validation findings

## Verification

```bash
# All tests pass
source venv/bin/activate
python -m pytest tests/ -v
# ===== 330 passed, 9 warnings in 14.40s =====
```

**Ready for Session 8**: ✅
