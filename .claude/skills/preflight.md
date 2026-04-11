---
name: preflight
description: Pre-flight checks before any training or execution run
---

# Pre-flight Checklist

Execute checks in exact order: **Data → Temporal → VRAM → Lock → Load**

If any check fails, report which step failed and **do not proceed**.

## Checks

### 1. DATA
```bash
# Check diskcache exists and has recent entries
ls -la data/cache/ 2>/dev/null || echo "FAIL: No cache directory"
# Verify required symbols have data
```
- Is required market data available?
- Is cache not stale (check timestamps)?

### 2. TEMPORAL
- Is `end_ts` or `as_of` set correctly in the run config?
- Verify no future data leakage in training/eval windows
- Check that all data fetches anchor to point-in-time

### 3. VRAM
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
```
- VRAM usage must be <2GB (idle state)
- If higher, identify loaded processes:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
- Warn and block if models already loaded

### 4. LOCK
```bash
# Check for stale lock files
ls -la /tmp/trade_swarm_*.lock 2>/dev/null
# Verify process_lock is available
```
- Is `process_lock` available?
- No stale lock files from crashed processes?

### 5. LOAD
```bash
# Ollama health check
curl -s http://localhost:11434/api/tags | head -c 100
```
- Is Ollama running?
- Can it respond to health check?

## Environment Checks

```bash
# OLLAMA_KEEP_ALIVE must be 0
echo $OLLAMA_KEEP_ALIVE  # Must output "0"

# STOP file must not exist
ls execution/state/STOP 2>/dev/null && echo "FAIL: STOP file exists"
```

## Output Format

```
## Pre-flight Check

| Step | Status | Details |
|------|--------|---------|
| DATA | ✓ PASS / ✗ FAIL | cache status |
| TEMPORAL | ✓ PASS / ✗ FAIL | end_ts value |
| VRAM | ✓ PASS / ✗ FAIL | usage: X/16GB |
| LOCK | ✓ PASS / ✗ FAIL | lock status |
| LOAD | ✓ PASS / ✗ FAIL | ollama status |
| ENV:KEEP_ALIVE | ✓ PASS / ✗ FAIL | value |
| ENV:STOP_FILE | ✓ PASS / ✗ FAIL | exists/not |

**Result**: READY / BLOCKED at step N
```
