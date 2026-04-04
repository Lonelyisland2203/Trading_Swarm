# Session 3 Implementation - Swarm Layer Part 1

**Date:** 2026-04-03
**Agent:** root-cause-engineer (architecture review) + claude-md-custodian (project memory)
**Status:** ✅ COMPLETE - 95 Tests Passing

---

## Executive Summary

Implemented the first half of the Swarm Layer with **VRAM-safe Ollama client** and **persona-based signal generation**. All architectural decisions validated against the 16 GB VRAM constraint.

**Test Coverage:** 95 tests passing (+37 new tests)
**Risk Level:** LOW - Safe to proceed to Session 4
**VRAM Safety:** VERIFIED CORRECT - Semaphore prevents concurrent model loads

---

## Architecture Review

Comprehensive architecture validation performed by root-cause-engineer agent covering 6 critical decisions:

### Decision 1: VRAM-Safe Model Management ✅

**Problem:** RTX 5070 Ti (16 GB) cannot load both models simultaneously
**Solution:** Semaphore + explicit unload pattern

```python
class OllamaClient:
    def __init__(self):
        self._model_lock = asyncio.Semaphore(1)  # Single-model exclusion
        self._current_model: str | None = None

    async def generate(self, model, prompt, options):
        async with self._model_lock:
            if self._current_model and self._current_model != model:
                await self._force_unload(self._current_model)  # Defensive unload
            result = await self._call_ollama(model, prompt, options)
            self._current_model = model
            return result
```

**Critical Pattern:** Always unload before switching models (defensive against stale state).

**VRAM Budget:**
- Qwen3-8B (4-bit): ~5 GB
- DeepSeek-R1-14B (4-bit): ~9 GB
- Combined: ~14 GB (UNSAFE)
- Single model: Max 10 GB with 6 GB buffer for CUDA kernels

### Decision 2: Temperature-Gated Caching ✅

**Problem:** Caching stochastic generations returns stale results
**Solution:** Only cache when temperature=0 and top_p=1.0

```python
def make_llm_cache_key(model, prompt, options):
    if options.get("temperature", 0.7) > 0.0:
        return None  # Signal: do not cache
    if options.get("top_p", 1.0) < 1.0:
        return None

    # Deterministic - safe to cache
    return f"llm:{sha256(json.dumps({model, prompt, seed}, sort_keys=True))}"
```

**Cache Tiers:**
- **Disk cache:** diskcache, 7-day TTL (reuse existing infrastructure)
- **Memory cache:** Not implemented (complexity vs benefit)

### Decision 3: Conservative Token Estimation ✅

**Problem:** Overestimating tokens causes truncation; underestimating causes context overflow
**Solution:** Conservative 3 chars/token estimate + truncate oldest price bars

```python
CHARS_PER_TOKEN_CONSERVATIVE = 3  # Underestimate for safety margin
MAX_PROMPT_TOKENS = 3000  # Leave 2k buffer for response (8k context)

def truncate_prompt(prompt, max_tokens=3000):
    # Preserve priority order:
    # 1. System instruction + /no_think
    # 2. Technical indicators
    # 3. Current price/regime
    # 4. Recent price action (truncate oldest bars first)
```

**Why NOT tiktoken:**
- 50 MB dependency for GPT-specific tokenizer
- Model mismatch (Qwen3/DeepSeek use different tokenizers)
- Ollama provides actual token count - calibrate post-hoc

### Decision 4: Fail-Fast Error Handling ✅

**Problem:** Auto-pulling models introduces unpredictable 5+ minute latency
**Solution:** Fail fast with actionable error message

```python
NON_RETRYABLE_ERRORS = (
    ModelNotFoundError,   # User must run: ollama pull <model>
    VRAMExhaustedError,   # CRITICAL: Never retry OOM
    json.JSONDecodeError, # Malformed response - retry won't help
)

RETRYABLE_ERRORS = (
    OllamaNetworkError,
    aiohttp.ClientError,
    asyncio.TimeoutError,
)

if response.status == 404:
    raise ModelNotFoundError(f"Model '{model}' not found. Run: ollama pull {model}")
```

**Retry Configuration:**
- Max retries: 3
- Base delay: 2.0s (longer than market data - model loading takes time)
- Exponential backoff: min(2.0 * 2^attempt, 30.0)

### Decision 5: Regime-Informed Persona Sampling ✅

**Problem:** Pure random or hardcoded persona selection ignores market context
**Solution:** Weighted sampling with regime modifiers

```python
BASE_PERSONA_WEIGHTS = {
    TradingPersona.CONTRARIAN: 0.20,
    TradingPersona.MOMENTUM: 0.25,
    TradingPersona.MEAN_REVERSION: 0.20,
    TradingPersona.BREAKOUT: 0.15,
    TradingPersona.CONSERVATIVE: 0.20,
}

REGIME_MODIFIERS = {
    MarketRegime.RISK_OFF: {
        TradingPersona.CONSERVATIVE: 1.5,  # Boost conservative
        TradingPersona.BREAKOUT: 0.5,      # Reduce breakout
    },
    MarketRegime.RISK_ON: {
        TradingPersona.MOMENTUM: 1.3,
        TradingPersona.MEAN_REVERSION: 0.7,
    },
}

def sample_persona(regime, seed=None):
    rng = random.Random(seed) if seed else _persona_rng  # Isolated RNG
    weights = BASE_PERSONA_WEIGHTS.copy()

    for persona, modifier in REGIME_MODIFIERS.get(regime, {}).items():
        weights[persona] *= modifier

    normalized = {p: w / sum(weights.values()) for p, w in weights.items()}
    return rng.choices(list(normalized.keys()), weights=list(normalized.values()), k=1)[0]
```

**Why This Approach:**
- Diversity: Not deterministic like round-robin
- Context-aware: Soft guidance from regime state
- Learnable: DPO training discovers which personas work when

### Decision 6: Multi-Stage JSON Extraction ✅

**Problem:** LLMs produce markdown fences, thinking tags, truncated JSON
**Solution:** 4-stage fallback + single clarification retry

```python
def extract_signal(raw_response, persona):
    # Attempt 1: Direct JSON parse
    try:
        return _validate_and_build_signal(json.loads(raw_response), persona, raw_response)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown fences (```json ... ```)
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL)
    if fence_match:
        try:
            return _validate_and_build_signal(json.loads(fence_match.group(1)), persona, raw_response)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Extract from thinking tags (<think>...</think>)
    think_match = re.search(r'</think>\s*(.*?)$', raw_response, re.DOTALL)
    if think_match:
        return extract_signal(think_match.group(1), persona)  # Recursive

    # Attempt 4: Regex extraction (last resort)
    direction_match = re.search(r'"direction"\s*:\s*"(HIGHER|LOWER)"', raw_response, re.IGNORECASE)
    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_response)
    if direction_match and confidence_match:
        return GeneratorSignal(
            direction=direction_match.group(1).upper(),
            confidence=float(confidence_match.group(1)),
            reasoning="[Extracted via regex fallback]",
            persona=persona,
            raw_response=raw_response,
        )

    raise ResponseValidationError("All extraction attempts failed")
```

**Single Retry Strategy:**
- On parse failure, append clarification prompt
- Only 1 retry (if fails twice, mark invalid)
- Returns `None` on total failure (skip this sample in training)

---

## Files Implemented

### swarm/exceptions.py

Custom exception hierarchy for proper error handling:

```python
class OllamaError(Exception): pass

class ModelNotFoundError(OllamaError): pass        # Non-retryable
class VRAMExhaustedError(OllamaError): pass        # CRITICAL: Never retry
class OllamaNetworkError(OllamaError): pass        # Retryable
class ResponseValidationError(OllamaError): pass
class TokenBudgetExceededError(OllamaError): pass
```

### swarm/ollama_client.py (361 lines)

Key components:

- **make_llm_cache_key()** - Temperature-gated cache key generation
- **OllamaClient class** - VRAM-aware async client
  - `__init__()` - Validates OLLAMA_KEEP_ALIVE=0
  - `__aenter__/__aexit__()` - Context manager for cleanup
  - `generate()` - Main generation method with semaphore protection
  - `unload_current()` - Explicit model unload (called between phases)
  - `_generate_with_retry()` - Exponential backoff retry logic
  - `_call_generate_api()` - HTTP POST to Ollama /api/generate
  - `_force_unload()` - Empty prompt with keep_alive=0
- **estimate_tokens()** - Conservative token counting
- **truncate_prompt()** - Structure-aware prompt truncation

### swarm/generator.py (276 lines)

Key components:

- **TradingPersona enum** - Five persona types
- **BASE_PERSONA_WEIGHTS** - Default sampling weights (sum to 1.0)
- **REGIME_MODIFIERS** - Regime-specific weight multipliers
- **PERSONA_PROMPTS** - System prompts for each persona
- **GeneratorSignal dataclass** - Validated signal output
- **sample_persona()** - Regime-informed weighted sampling
- **extract_signal()** - Multi-stage JSON extraction
- **_validate_and_build_signal()** - Schema validation
- **generate_signal()** - Main async generation function

---

## Test Coverage

### tests/test_ollama_client.py (17 tests)

**Cache Key Generation (4 tests):**
- Deterministic generations produce cache keys
- Stochastic generations (temp>0) return None
- Same inputs → same cache key (reproducibility)
- Different inputs → different cache keys

**Token Estimation (4 tests):**
- Conservative estimation (3 chars/token)
- Short prompts not truncated
- Long prompts truncate oldest price bars
- Irreducible prompts raise TokenBudgetExceededError

**Ollama Client (9 tests):**
- Context manager lifecycle (session creation/cleanup)
- Deterministic results cached
- Stochastic results not cached
- ModelNotFoundError on 404
- VRAMExhaustedError on OOM
- Network errors retried
- Retry exhaustion raises last exception
- Explicit model unload

### tests/test_generator.py (20 tests)

**Persona Sampling (5 tests):**
- Same seed → same persona (reproducibility)
- Different seeds → different personas (variation)
- Regime modifiers affect distribution
- Base weights sum to 1.0 (validation)
- All 5 personas reachable

**Response Extraction (10 tests):**
- Direct JSON extraction
- Markdown fence extraction (```json ... ```)
- Thinking tag extraction (<think>...</think>)
- Regex fallback extraction
- Invalid direction raises ValueError
- Confidence clamped to [0, 1]
- Case-insensitive direction matching
- Unparseable response raises ResponseValidationError
- Raw response preserved in signal
- Missing reasoning handled gracefully

**Persona Prompts (5 tests):**
- All personas have prompts defined
- All prompts are unique
- Contrarian prompt mentions overreaction/RSI
- Momentum prompt mentions trend/MACD
- Conservative prompt mentions capital preservation/risk

---

## Test Results

```
============================= test session starts ==============================
collected 95 items

tests/test_config.py ................                                   [ 18%]
tests/test_data_layer.py .....................                          [ 40%]
tests/test_indicators.py ...................                            [ 60%]
tests/test_generator.py ....................                            [ 82%]
tests/test_ollama_client.py .................                           [100%]

============================= 95 passed in 13.31s ==============================
```

**Breakdown:**
- Session 1 (Config): 18 tests
- Session 2 (Data Layer): 21 tests
- Session 2 (Indicators): 19 tests
- Session 3 (Generator): 20 tests
- Session 3 (Ollama Client): 17 tests

**Total: 95 tests, 100% passing**

---

## Critical Patterns Established

### 1. Isolated Random Number Generation

Matches data layer pattern from `prompt_builder.py`:

```python
# Module-level isolated RNG
_persona_rng = random.Random()

def sample_persona(regime, seed=None):
    rng = random.Random(seed) if seed else _persona_rng
    # ... sampling logic
```

**Why:** Prevents global state pollution, ensures reproducibility with seeds.

### 2. Async Context Managers

Matches data layer pattern from `cache_wrapper.py`, `market_data.py`:

```python
class OllamaClient:
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.unload_current()  # Guaranteed cleanup
        if self._session:
            await self._session.close()
```

**Why:** Guarantees resource cleanup even on exceptions.

### 3. Conservative Estimation with Post-Hoc Calibration

```python
estimated_tokens = len(prompt) / CHARS_PER_TOKEN_CONSERVATIVE

# Log actual vs estimated for calibration
if "prompt_eval_count" in result:
    actual_tokens = result["prompt_eval_count"]
    ratio = actual_tokens / estimated_tokens
    logger.debug("Token estimation", estimated=estimated_tokens, actual=actual_tokens, ratio=ratio)
```

**Why:** Underestimate for safety, improve accuracy over time with real data.

### 4. Temperature-Gated Caching

Only cache deterministic generations:

```python
if options.get("temperature", 0.7) > 0.0:
    return None  # Do not cache stochastic generations
```

**Why:** Prevents stale cached results for non-deterministic prompts.

---

## VRAM Safety Verification

### Invariants Enforced

1. **OLLAMA_KEEP_ALIVE=0** - Validated at client initialization
2. **Single model at a time** - Semaphore prevents concurrent loads
3. **Explicit unload** - Must call `unload_current()` between phases
4. **Defensive unload** - Always unload when switching models

### Usage Pattern (Session 4 Orchestrator)

```python
async def run_signal_generation(prompt):
    async with OllamaClient() as client:
        # Generator phase
        signal = await generate_signal(client, GENERATOR_MODEL, prompt, regime)
        await client.unload_current()  # EXPLICIT unload before Critic

        # Critic phase
        critique = await client.generate(CRITIC_MODEL, critique_prompt)
        await client.unload_current()  # Cleanup
```

### VRAM Budget Confirmation

| Phase | Model | VRAM | Safe? |
|-------|-------|------|-------|
| Generator only | Qwen3-8B (4-bit) | ~5 GB | ✅ |
| Critic only | DeepSeek-R1-14B (4-bit) | ~9 GB | ✅ |
| Both simultaneously | Combined | ~14 GB | ❌ UNSAFE |
| Sequential with unload | One at a time | Max 10 GB | ✅ |

**Buffer:** 6 GB for CUDA kernels, attention cache, activations

---

## Known Issues

None - all tests passing.

---

## Next Steps

**Session 4: Swarm Layer Part 2**

Implement critic and orchestrator:

1. **swarm/critic.py** - DeepSeek-R1 critique with reasoning evaluation
   - Critique prompt builder
   - Reasoning quality scoring
   - Contradiction detection
   - Confidence calibration

2. **swarm/orchestrator.py** - LangGraph workflow
   - Generator → Critic pipeline
   - State management
   - Error handling and retries
   - Signal aggregation

**Prerequisites:**
- Ollama running: `ollama serve`
- Models available: `ollama pull qwen3:8b && ollama pull deepseek-r1:14b`

---

**Implemented by:** Claude Sonnet 4.5
**Architecture validated by:** root-cause-engineer agent
**Project memory updated by:** claude-md-custodian agent
**Status:** COMPLETE & VERIFIED
