"""
Async Ollama client with VRAM management, caching, and retry logic.

Critical Design:
- Semaphore ensures only one model loaded at a time (16 GB VRAM constraint)
- Explicit unload between model switches (OLLAMA_KEEP_ALIVE=0 enforcement)
- Context manager guarantees cleanup on exceptions
- Caches only deterministic generations (temperature=0)
"""

import asyncio
import hashlib
import json
from pathlib import Path

import aiohttp
from diskcache import Cache
from loguru import logger

from config.settings import settings
from .exceptions import (
    ModelNotFoundError,
    OllamaError,
    OllamaNetworkError,
    VRAMExhaustedError,
)

# Token estimation constants
CHARS_PER_TOKEN_CONSERVATIVE = 3  # Conservative estimate for safety margin
MAX_PROMPT_TOKENS = 3000  # Leave headroom for response (8k context - 3k prompt - 2k buffer)

# Retry configuration
RETRYABLE_ERRORS = (
    OllamaNetworkError,
    aiohttp.ClientError,
    asyncio.TimeoutError,
)

NON_RETRYABLE_ERRORS = (
    ModelNotFoundError,
    VRAMExhaustedError,
    json.JSONDecodeError,
)


def make_llm_cache_key(model: str, prompt: str, options: dict) -> str | None:
    """
    Create cache key for LLM generation.

    Returns None if generation is non-deterministic (temperature > 0 or top_p < 1.0).
    This prevents caching stochastic generations that would return stale results.

    Args:
        model: Model identifier
        prompt: Input prompt
        options: Generation options (temperature, top_p, seed, etc.)

    Returns:
        SHA256 hash key for deterministic generations, None otherwise
    """
    # Only cache deterministic generations
    if options.get("temperature", 0.7) > 0.0:
        return None
    if options.get("top_p", 1.0) < 1.0:
        return None

    # Deterministic - safe to cache
    key_data = {
        "model": model,
        "prompt": prompt,
        "seed": options.get("seed", 0),
    }
    key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    return f"llm:{key_hash}"


class OllamaClient:
    """
    Async Ollama client with VRAM-aware model management.

    Critical Invariants:
    - Only ONE model loaded at a time (enforced via semaphore)
    - Explicit unload before switching models
    - keep_alive=0 verified at initialization

    Usage:
        async with OllamaClient() as client:
            response = await client.generate("qwen3:8b", prompt)
            await client.unload_current()  # Explicit unload
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
        cache_dir: Path | None = None,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (default from settings)
            timeout: Request timeout in seconds
            cache_dir: Cache directory (default: .cache/llm/)
        """
        self.base_url = base_url or settings.ollama.base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # VRAM safety: verify keep_alive=0
        if settings.ollama.keep_alive != 0:
            raise ValueError(f"VRAM safety: keep_alive must be 0, got {settings.ollama.keep_alive}")

        # Model exclusion lock
        self._model_lock = asyncio.Semaphore(1)
        self._current_model: str | None = None

        # Disk cache for deterministic generations
        cache_path = cache_dir or Path(".cache/llm/")
        cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_path))

        # HTTP session (created in __aenter__)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OllamaClient":
        """Create HTTP session."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup: unload model and close session."""
        await self.unload_current()
        if self._session:
            await self._session.close()
            self._session = None

    async def generate(
        self,
        model: str,
        prompt: str,
        options: dict | None = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Generate response from Ollama model with VRAM-safe model switching.

        Args:
            model: Model identifier (e.g., "qwen3:8b")
            prompt: Input prompt
            options: Generation options (temperature, top_p, etc.)
            max_retries: Maximum retry attempts

        Returns:
            Response dict with keys: response, model, created_at, done, etc.

        Raises:
            ModelNotFoundError: Model not available (non-retryable)
            VRAMExhaustedError: VRAM exhausted (non-retryable)
            OllamaError: Other Ollama errors
        """
        options = options or {}

        # Check cache (only for deterministic generations)
        cache_key = make_llm_cache_key(model, prompt, options)
        if cache_key:
            cached = await asyncio.to_thread(self._cache.get, cache_key)
            if cached is not None:
                logger.debug("Cache hit", model=model, prompt_len=len(prompt))
                return cached

        # VRAM-safe model switching
        async with self._model_lock:
            # Always unload if switching models (defensive against stale state)
            if self._current_model and self._current_model != model:
                logger.info("Switching models", from_model=self._current_model, to_model=model)
                await self._force_unload(self._current_model)

            # Generate with retry logic
            response = await self._generate_with_retry(model, prompt, options, max_retries)

            # Update current model state
            self._current_model = model

            # Cache deterministic results
            if cache_key:
                await asyncio.to_thread(
                    self._cache.set,
                    cache_key,
                    response,
                    expire=7 * 24 * 3600,  # 7 days
                )

            return response

    async def unload_current(self) -> None:
        """
        Unload currently loaded model to free VRAM.

        Must be called before switching to a different model or ending generation phase.
        """
        async with self._model_lock:
            if self._current_model:
                logger.info("Unloading model", model=self._current_model)
                await self._force_unload(self._current_model)
                self._current_model = None

    async def _generate_with_retry(
        self,
        model: str,
        prompt: str,
        options: dict,
        max_retries: int,
    ) -> dict:
        """
        Generate with exponential backoff retry.

        Retries only on transient errors (network, timeout).
        Fails fast on permanent errors (model not found, VRAM exhausted).
        """
        last_exception = None
        base_delay = 2.0  # Longer than market data (model loading takes time)

        for attempt in range(max_retries):
            try:
                return await self._call_generate_api(model, prompt, options)

            except RETRYABLE_ERRORS as e:
                last_exception = e
                delay = min(base_delay * (2**attempt), 30.0)
                logger.warning(
                    "Retryable error, backing off",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

            except NON_RETRYABLE_ERRORS:
                # Do not retry - fail fast
                raise

        # Exhausted retries
        raise last_exception

    async def _call_generate_api(
        self,
        model: str,
        prompt: str,
        options: dict,
    ) -> dict:
        """
        Call Ollama /api/generate endpoint.

        Raises:
            ModelNotFoundError: Model not found
            VRAMExhaustedError: VRAM exhausted
            OllamaNetworkError: Network error
        """
        if not self._session:
            raise OllamaError("Client not initialized - use async context manager")

        # `think` is a top-level Ollama parameter (not inside options) for qwen3
        top_level_keys = {"think"}
        top_level_params = {k: v for k, v in options.items() if k in top_level_keys}
        model_options = {k: v for k, v in options.items() if k not in top_level_keys}

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                **model_options,
                "keep_alive": 0,  # Force unload after generation (VRAM safety)
            },
            **top_level_params,
        }

        url = f"{self.base_url}/api/generate"

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 404:
                    raise ModelNotFoundError(f"Model '{model}' not found. Run: ollama pull {model}")

                if resp.status == 500:
                    error_text = await resp.text()
                    if "out of memory" in error_text.lower() or "vram" in error_text.lower():
                        raise VRAMExhaustedError(
                            f"VRAM exhausted loading model '{model}'. Unload current model first."
                        )
                    raise OllamaError(f"Ollama internal error: {error_text}")

                resp.raise_for_status()
                result = await resp.json()

                # Log token usage for calibration
                if "eval_count" in result:
                    estimated_tokens = len(prompt) / CHARS_PER_TOKEN_CONSERVATIVE
                    actual_tokens = result.get("prompt_eval_count", 0)
                    logger.debug(
                        "Token estimation",
                        estimated=int(estimated_tokens),
                        actual=actual_tokens,
                        ratio=actual_tokens / estimated_tokens if estimated_tokens > 0 else 0,
                    )

                return result

        except aiohttp.ClientError as e:
            raise OllamaNetworkError(f"Network error: {e}")

    async def _force_unload(self, model: str) -> None:
        """
        Force model unload via keep_alive=0 generation.

        Sends empty prompt with keep_alive=0 to trigger immediate unload.
        """
        if not self._session:
            return

        payload = {
            "model": model,
            "prompt": "",  # Empty prompt
            "stream": False,
            "options": {"keep_alive": 0},
        }

        url = f"{self.base_url}/api/generate"

        try:
            async with self._session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    logger.debug("Model unloaded", model=model)
                else:
                    logger.warning("Unload may have failed", model=model, status=resp.status)
        except Exception as e:
            logger.warning("Exception during unload", model=model, error=str(e))


def estimate_tokens(text: str) -> int:
    """
    Conservative token estimation for prompt truncation.

    Uses 3 chars/token ratio (conservative for non-ASCII, numbers, symbols).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN_CONSERVATIVE


def truncate_prompt(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """
    Truncate prompt to fit within token budget.

    Preserves critical sections:
    1. System instruction + /no_think directive
    2. Technical indicators section
    3. Current price/regime
    4. Recent price action (truncates oldest bars first)

    Args:
        prompt: Input prompt
        max_tokens: Maximum token budget

    Returns:
        Truncated prompt

    Raises:
        TokenBudgetExceededError: Prompt too large even after truncation
    """
    from .exceptions import TokenBudgetExceededError

    estimated_tokens = estimate_tokens(prompt)

    if estimated_tokens <= max_tokens:
        return prompt

    # Find price action section
    price_section_marker = "## Recent Price Action"

    if price_section_marker not in prompt:
        # No price section to truncate - prompt is irreducibly large
        raise TokenBudgetExceededError(
            f"Prompt exceeds {max_tokens} tokens even without price action section"
        )

    # Split into sections
    parts = prompt.split(price_section_marker)
    header = parts[0]
    price_section = parts[1] if len(parts) > 1 else ""

    # Truncate oldest bars from price section
    price_lines = price_section.strip().split("\n")

    while estimate_tokens(header + price_section_marker + "\n".join(price_lines)) > max_tokens:
        if len(price_lines) <= 1:
            # Can't truncate further
            raise TokenBudgetExceededError(
                f"Prompt exceeds {max_tokens} tokens even with minimal price action"
            )
        price_lines = price_lines[1:]  # Remove oldest bar

    truncated = header + price_section_marker + "\n" + "\n".join(price_lines)

    logger.info(
        "Prompt truncated",
        original_tokens=estimated_tokens,
        final_tokens=estimate_tokens(truncated),
        bars_removed=len(price_section.strip().split("\n")) - len(price_lines),
    )

    return truncated
