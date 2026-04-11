---
name: review
description: Review staged changes against project hard constraints before commit
---

# Pre-Commit Constraint Review

Run `git diff --staged` and review against all 9 checks below. Report pass/fail per check with specific `file:line` references for failures.

## Checks

### 1. VRAM Isolation
- Any code path that could load both models simultaneously?
- Any missing `OLLAMA_KEEP_ALIVE=0`?
- Any missing semaphore release between model calls?

### 2. Temporal Safety
- Any `get_ohlcv` or data fetch without `end_ts`/`as_of` anchoring?
- Any training data that could leak future information?

### 3. Fee Model Integrity
- Any reward calculation not using `net_return()`?
- Any modification to `fee_model.py` or `FeeModelSettings`?

### 4. Process Isolation
- Any concurrent process A+B without `process_lock`?
- Any missing `fcntl` lock acquisition?

### 5. Pre-flight Order
- Any training or execution code that skips Data→Temporal→VRAM→Lock→Load?

### 6. Asymmetric Penalties
- Any symmetric reward clipping?
- Any equal treatment of false bullish vs false bearish?

### 7. Anti-patterns
- Unsloth imports?
- DPO usage where GRPO should be used?
- Linear fee approximations?
- Cached `temp>0` responses?

### 8. Test Coverage
- Any new public function without corresponding test?

### 9. Type Hints
- Any new public function missing type annotations?

## Output Format

```
## Constraint Review

| Check | Status | Notes |
|-------|--------|-------|
| VRAM Isolation | ✓ PASS / ✗ FAIL | file:line if fail |
| Temporal Safety | ✓ PASS / ✗ FAIL | file:line if fail |
| Fee Model Integrity | ✓ PASS / ✗ FAIL | file:line if fail |
| Process Isolation | ✓ PASS / ✗ FAIL | file:line if fail |
| Pre-flight Order | ✓ PASS / ✗ FAIL | file:line if fail |
| Asymmetric Penalties | ✓ PASS / ✗ FAIL | file:line if fail |
| Anti-patterns | ✓ PASS / ✗ FAIL | file:line if fail |
| Test Coverage | ✓ PASS / ✗ FAIL | missing tests |
| Type Hints | ✓ PASS / ✗ FAIL | file:line if fail |

**Result**: READY TO COMMIT / BLOCKED (N issues)
```
