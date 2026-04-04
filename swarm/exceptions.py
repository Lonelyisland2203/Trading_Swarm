"""
Custom exception hierarchy for Ollama client and swarm operations.

Distinguishes between retryable and non-retryable errors for proper error handling.
"""


class OllamaError(Exception):
    """Base class for Ollama-related errors."""
    pass


class ModelNotFoundError(OllamaError):
    """
    Model not available in Ollama.

    This is a non-retryable error - requires manual intervention to pull model.
    """
    pass


class VRAMExhaustedError(OllamaError):
    """
    VRAM exhausted during model loading or generation.

    CRITICAL: Never retry this error. Must unload current model first.
    """
    pass


class OllamaNetworkError(OllamaError):
    """
    Transient network error communicating with Ollama.

    Safe to retry with exponential backoff.
    """
    pass


class ResponseValidationError(OllamaError):
    """
    LLM response failed validation after all extraction attempts.

    May indicate:
    - Truncated response (hit token limit mid-JSON)
    - Model did not follow instructions
    - Schema drift
    """
    pass


class TokenBudgetExceededError(OllamaError):
    """
    Prompt exceeds maximum token budget even after truncation.

    Indicates prompt is too large to fit in context window.
    """
    pass
