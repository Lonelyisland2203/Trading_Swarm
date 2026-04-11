# GRPO Trainer Design Specification

**Date:** 2026-04-11
**Session:** 17D
**Status:** Approved

## Overview

This document specifies the design for `training/grpo_trainer.py`, the core GRPO (Group Relative Policy Optimization) training loop following the DeepSeek-R1 algorithm.

## Architecture Decision

**Approach:** Monolithic Trainer (single `GRPOTrainer` class)

GRPO requires generating G=4 completions per input, computing rewards, then computing group-relative advantages before gradient updates. This doesn't fit the HuggingFace Trainer pattern. A custom training loop gives full control over:
- Sequential generation (VRAM constraint)
- Reference model weight swapping (KL penalty)
- Group-relative advantage computation

## Module Structure

```
training/
├── grpo_trainer.py      # Main trainer
├── grpo_data.py         # GRPOTrainingExample + walk-forward functions
└── tests/training/
    ├── test_grpo_trainer.py
    └── test_grpo_data.py
```

## Core Data Types

### GRPOTrainingExample (grpo_data.py)

```python
@dataclass(frozen=True)
class GRPOTrainingExample:
    """Single training example for GRPO."""
    market_snapshot: str      # Input prompt
    actual_direction: str     # Ground truth: "LONG", "SHORT", "FLAT"
    gross_return_pct: float   # Ground truth return for reward computation
    timestamp_ms: int         # For temporal ordering
```

### GRPOStepResult (grpo_trainer.py)

```python
@dataclass
class GRPOStepResult:
    """Result of a single training step."""
    step: int
    mean_reward: float
    mean_advantage: float
    kl_divergence: float
    loss: float
    vram_mb: int
```

### GRPOTrainingResult (grpo_trainer.py)

```python
@dataclass
class GRPOTrainingResult:
    """Result of full training run."""
    success: bool
    adapter_path: Path | None
    steps_completed: int
    final_metrics: dict
    error: str | None
```

## Training Loop Algorithm

For each training example (market_snapshot):

1. **Generate G=4 completions SEQUENTIALLY** (VRAM constraint)
   - Each completion uses current policy (base + LoRA)
   - temperature=0.7, top_p=0.9, max_tokens=512
   - Clear KV cache between generations

2. **Score each completion:**
   - Parse predicted_direction from completion text
   - Call `compute_grpo_reward()` for each
   - Returns 4 reward values

3. **Compute group-relative advantages:**
   - `advantage_i = (reward_i - mean) / std`
   - Uses `compute_group_advantages()` from grpo_reward.py

4. **Compute GRPO loss for each completion:**
   - `ratio = π(completion|prompt) / π_ref(completion|prompt)`
   - `clipped_ratio = clip(ratio, 1-ε, 1+ε)` where ε=0.2
   - `policy_loss = -min(ratio * advantage, clipped_ratio * advantage)`
   - `kl_penalty = β * KL(π || π_ref)` where β=0.04
   - `loss = policy_loss + kl_penalty`

5. **Accumulate gradients** (gradient_accumulation_steps=16)

6. **Update LoRA weights** every 16 steps

## Reference Model Handling

The SFT adapter is the reference policy (π_ref). To avoid loading two models:

- Load base model once with 4-bit quantization
- Load SFT adapter weights into a separate state dict (CPU)
- For KL computation: temporarily swap LoRA weights, compute log probs, swap back
- This keeps VRAM at ~10GB instead of ~20GB

## Direction Parsing

```python
def parse_direction(completion: str) -> str:
    """Extract direction from DECISION section."""
    # Look for DECISION section, then find LONG/SHORT/FLAT keyword
    # Returns "FLAT" if unparseable (conservative default)
```

## VRAM Management

### Budget (16GB RTX 5070 Ti)

| Component | Estimated VRAM |
|-----------|---------------|
| Base model (4-bit) | ~5 GB |
| LoRA weights (r=32) | ~0.2 GB |
| KV cache (1 sequence) | ~1 GB |
| Gradients + optimizer | ~3 GB |
| Activations (grad checkpointing) | ~1 GB |
| **Total** | **~10 GB** |

### Sequential Generation

- Generate completion 1, store tokens + log probs
- Clear KV cache
- Generate completion 2, store tokens + log probs
- Repeat for G=4
- Peak VRAM stays under 14GB threshold

### VRAM Monitoring

```python
def _log_vram(self, step: int) -> int:
    """Log VRAM every 100 steps, warn if >14GB."""
    vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)
    if vram_mb > 14 * 1024:
        logger.warning(f"VRAM exceeded 14GB: {vram_mb}MB")
    return vram_mb
```

## Checkpointing

### Directory Structure

```
adapters/
├── grpo_checkpoints/
│   ├── checkpoint-500/
│   │   ├── adapter_model.safetensors
│   │   └── metadata.json
│   ├── checkpoint-1000/
│   └── ...
└── grpo_latest/
    ├── adapter_model.safetensors
    └── metadata.json
```

### Metadata Contents

```json
{
  "step": 500,
  "mean_reward": 0.23,
  "mean_advantage": 0.0,
  "kl_divergence": 0.012,
  "loss": 0.45,
  "config_hash": "abc123...",
  "timestamp_ms": 1234567890
}
```

- Save every 500 steps via `model.save_pretrained()`
- Config hash for reproducibility verification
- Final adapter saved to `grpo_latest/`

## Logging

### Log File

```
training/logs/grpo_YYYYMMDD_HHMMSS.jsonl
```

### Log Entry Format

```json
{"step": 100, "mean_reward": 0.23, "mean_advantage": 0.0, "kl": 0.012, "loss": 0.45, "vram_mb": 10240, "timestamp": 1234567890}
```

## Pre-flight Checks

Order (fail-fast, cheap to expensive):

1. **Data check:** examples list non-empty
2. **Temporal check:** examples sorted by timestamp_ms
3. **VRAM check:** `check_vram_availability(min_free_gb=9.0)`
4. **Lock check:** `check_can_train()` returns True
5. **OLLAMA_KEEP_ALIVE=0** enforced
6. **STOP file check:** `execution/state/STOP` does not exist

## Process Isolation

```python
def train(self, examples: list[GRPOTrainingExample]) -> GRPOTrainingResult:
    # Pre-flight (before acquiring lock)
    ok, reason = self._run_preflight()
    if not ok:
        return GRPOTrainingResult(success=False, error=reason, ...)

    # Acquire lock for entire training duration
    with acquire_training_lock():
        # Load model, run training loop
        ...
```

## Error Handling

| Error Type | Handling |
|------------|----------|
| OOM during generation | Catch, log VRAM, save checkpoint, return partial result |
| Corrupt completion (unparseable) | Assign FLAT direction, log warning, continue |
| NaN in loss | Skip step, log warning, continue if <5% of steps |
| STOP file appears mid-training | Check every 100 steps, graceful shutdown with checkpoint |
| Lock lost | Fatal error, save emergency checkpoint |

### Graceful Shutdown

```python
def _check_stop_file(self) -> bool:
    return Path("execution/state/STOP").exists()

# In training loop:
if step % 100 == 0 and self._check_stop_file():
    logger.warning("STOP file detected, saving checkpoint and exiting")
    self._save_checkpoint(step, "early_stop")
    return GRPOTrainingResult(success=False, error="STOP file", ...)
```

### GPU Cleanup

```python
def _cleanup(self):
    del self.model
    del self.optimizer
    torch.cuda.empty_cache()
    gc.collect()
```

## Testing Strategy

### Mock Strategy

All HuggingFace calls mocked. No GPU required for tests.

```python
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode.return_value = (
        "## THESIS\nBullish\n## EVIDENCE\nRSI oversold\n"
        "## RISK\nVolatility\n## DECISION\nLONG"
    )
    return tokenizer
```

### Test Cases (test_grpo_trainer.py)

| Test | Verifies |
|------|----------|
| `test_advantage_computation_known_inputs` | advantages sum ≈ 0, normalized correctly |
| `test_advantage_single_completion` | G=1 → advantage = 0.0 |
| `test_clipping_ratio` | ratio outside [0.8, 1.2] gets clipped |
| `test_kl_penalty_applied` | loss includes β × KL term |
| `test_reference_model_is_sft_adapter` | Reference loaded from `adapters/sft_base/` |
| `test_checkpoint_saves_every_500_steps` | Verify save calls at correct intervals |
| `test_vram_warning_above_14gb` | Warning logged when threshold exceeded |
| `test_stop_file_triggers_graceful_shutdown` | Early exit with checkpoint |
| `test_preflight_fails_without_lock` | Failure when lock unavailable |
| `test_direction_parsing_variants` | "LONG", "long", "HIGHER" all → "LONG" |
| `test_corrupt_completion_defaults_flat` | Unparseable text → FLAT |

### Test Cases (test_grpo_data.py)

| Test | Verifies |
|------|----------|
| `test_temporal_ordering_enforced` | Examples sorted by timestamp_ms |
| `test_no_future_data_in_training` | All train timestamps < test timestamps |
| `test_replay_buffer_sampling` | 15% sampled from history |

### Coverage Target

- `grpo_trainer.py`: 90%+
- `grpo_data.py`: 95%+

## Dependencies

Uses existing modules:
- `training/grpo_config.py` — All hyperparameters
- `training/grpo_reward.py` — `compute_grpo_reward()`, `compute_group_advantages()`
- `training/process_lock.py` — `acquire_training_lock()`
- `training/vram_check.py` — `check_vram_availability()`

External:
- `transformers` — Model loading, generation
- `peft` — LoRA adapter management
- `torch` — Gradient computation
- `bitsandbytes` — 4-bit quantization

## Walk-Forward Integration

New functions in `grpo_data.py` (parallel to existing `walk_forward.py` for DPO):

```python
def create_grpo_walk_forward_split(
    examples: list[GRPOTrainingExample],
    train_window: int = 500,
    test_window: int = 100,
    replay_ratio: float = 0.15,
) -> GRPOWalkForwardSplit:
    """Create temporal train/test split for GRPO examples."""
    ...
```

## Configuration Reference

From `grpo_config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| group_size | 4 | G: completions per prompt |
| kl_penalty_beta | 0.04 | β: KL divergence coefficient |
| clip_epsilon | 0.2 | ε: PPO-style ratio clipping |
| learning_rate | 5e-6 | AdamW learning rate |
| gradient_accumulation_steps | 16 | Effective batch size |
| max_steps | 5000 | Training duration |
| checkpoint_interval_steps | 500 | Checkpoint frequency |
| max_vram_gb | 14.0 | VRAM warning threshold |
| vram_log_interval_steps | 100 | VRAM logging frequency |
