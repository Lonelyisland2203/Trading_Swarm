---
name: grpo-experiment
description: Scaffold and run a single GRPO experiment during autoresearch
---

# GRPO Experiment

Run a single controlled experiment with automatic commit/revert based on IC improvement.

## Protected Files — NEVER MODIFY

- `fee_model.py` / `FeeModelSettings`
- Evaluation metrics (IC/Brier/MACE definitions)
- `market_data.py`, `indicators.py` (data pipeline)
- VRAM constraints, safety controls
- `process_lock.py`, execution safety

## Allowed Modifications

Choose **ONE** per experiment:

| Parameter | File | Range |
|-----------|------|-------|
| G (group size) | `grpo_config.py` | 2, 4, 8 |
| β (KL penalty) | `grpo_config.py` | 0.01–0.1 |
| ε (clipping) | `grpo_config.py` | 0.1–0.3 |
| Learning rate | `grpo_config.py` | 1e-5–5e-4 |
| LoRA rank | `grpo_config.py` | 16, 32, 64 |
| LoRA alpha | `grpo_config.py` | 32, 64, 128 |
| Reward weights | `reward_config.py` | structure vs decision |
| Asymmetry coefficients | `reward_config.py` | false_bullish: -1.2 to -2.0 |
| Indicator subset | `signal_config.py` | drop 1-2 indicators |
| Prompt template | `prompts/` | wording variations |

## Workflow

### 1. Read Current State
```bash
cat training/grpo_config.py
cat results.tsv | tail -5
```
Identify current best IC and config.

### 2. Propose Modification
State clearly:
```
MODIFICATION: [parameter] from [old] to [new]
HYPOTHESIS: [why this might improve IC]
```

### 3. Commit Change
```bash
git add training/grpo_config.py  # or relevant file
git commit -m "experiment: [parameter] [old]->[new]"
```

### 4. Pre-flight
Run `/preflight`. Stop if any check fails.

### 5. Training Run
```bash
python -m training.grpo_trainer \
  --max_steps 1000 \
  --subset_size 2000 \
  --eval_steps 200
```
Expected time: ~15 min on RTX 5070 Ti.

### 6. Evaluate
```bash
python -m evaluation.metrics \
  --checkpoint latest \
  --test_window held_out
```
Capture: IC, Brier, MACE

### 7. Log Results
Append to `results.tsv`:
```
timestamp	config_hash	IC	Brier	MACE	description
2024-01-15T12:00:00	a1b2c3d4	0.052	0.243	0.018	β 0.04->0.06
```

### 8. Decision
```python
if new_IC > best_IC:
    # KEEP
    print(f"✓ KEPT: IC improved {best_IC:.4f} -> {new_IC:.4f}")
else:
    # REVERT
    git reset --hard HEAD~1
    print(f"✗ REVERTED: IC {new_IC:.4f} <= best {best_IC:.4f}")
```

### 9. Summary
```
## Experiment Summary

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| IC | 0.048 | 0.052 | +0.004 |
| Brier | 0.251 | 0.243 | -0.008 |
| MACE | 0.020 | 0.018 | -0.002 |

**Decision**: KEPT / REVERTED
**Running Best IC**: 0.052
**Experiments Run**: 14
**Success Rate**: 4/14 (28.6%)
```

## Constraints

- One modification per experiment (isolate variables)
- Always run /preflight before training
- Never skip evaluation step
- Always log to results.tsv before deciding
- Revert immediately if IC doesn't improve
- Stop if 5 consecutive experiments fail to improve
