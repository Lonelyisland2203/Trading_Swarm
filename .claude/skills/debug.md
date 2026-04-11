---
name: debug
description: Structured debugging workflow for test failures and runtime errors
---

# Debug Workflow

Follow these steps in order. Do not skip steps.

## 1. REPRODUCE

Run the failing test/command in isolation:
```bash
# For test failure
pytest tests/path/test_module.py::test_name -v --tb=long

# For runtime error
python -m module.path 2>&1 | tee /tmp/debug_output.txt
```

Capture:
- Full traceback
- Last successful state
- Input that triggered failure

## 2. PRE-FLIGHT CHECK

**Check this first. 80% of runtime errors come from pre-flight violations.**

| Violation | Symptom |
|-----------|---------|
| DATA skipped | `KeyError`, missing OHLCV, empty DataFrame |
| TEMPORAL skipped | Future data in training, impossible accuracy |
| VRAM skipped | OOM, CUDA error, model load timeout |
| LOCK skipped | Race condition, corrupted state, deadlock |
| LOAD skipped | Connection refused, Ollama not responding |

Quick check:
```bash
# Was pre-flight run?
grep -n "preflight\|pre_flight" $(git diff --name-only HEAD~1)

# VRAM state
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Lock state
ls -la /tmp/trade_swarm_*.lock 2>/dev/null

# Ollama state
curl -s http://localhost:11434/api/tags | jq '.models | length'
```

## 3. ISOLATE

Categorize the failure:

| Category | Indicators | Check |
|----------|------------|-------|
| VRAM | OOM, CUDA error, model timeout | `nvidia-smi`, process list |
| Temporal | Impossible metrics, future leak | Check `end_ts`/`as_of` in call chain |
| Lock | Deadlock, race, corruption | Lock file timestamps, process tree |
| Fee Model | Wrong PnL, reward mismatch | Verify `net_return()` usage |
| Data | Empty df, missing keys | Cache state, API response |

```bash
# Trace the specific failure point
grep -n "ERROR\|Exception\|Traceback" /tmp/debug_output.txt
```

## 4. HYPOTHESISE

State **one** specific, falsifiable hypothesis:

```
HYPOTHESIS: [Component] fails because [specific cause] when [condition].

Example: "GRPOTrainer fails because VRAM is not released between
generator and critic calls when batch size exceeds 4."
```

Do not proceed without a clear hypothesis.

## 5. FIX

Implement the **minimal** fix:
- Change one thing at a time
- Verify fix addresses root cause, not symptom
- Check fix doesn't violate hard constraints

```bash
# Verify fix
pytest tests/path/test_module.py::test_name -v
```

## 6. REGRESSION TEST

Write a test that would have caught this:

```python
def test_regression_issue_NNN() -> None:
    """Regression test: [brief description of bug].

    Root cause: [what went wrong]
    Fix: [what was changed]
    """
    # Arrange: Set up the exact failure condition
    # Act: Trigger the code path that failed
    # Assert: Verify the fix holds
```

Add to appropriate test file. Run full suite:
```bash
pytest
```

## Output Format

```
## Debug Report

**Failure**: [test name or error description]
**Traceback**: [key lines]

### Pre-flight Check
- DATA: ✓/✗
- TEMPORAL: ✓/✗
- VRAM: ✓/✗
- LOCK: ✓/✗
- LOAD: ✓/✗

### Category
[VRAM | Temporal | Lock | Fee Model | Data | Other]

### Hypothesis
[One specific hypothesis]

### Fix
[File:line — change description]

### Regression Test
[Test file:test name]

**Status**: RESOLVED / NEEDS ESCALATION
```
