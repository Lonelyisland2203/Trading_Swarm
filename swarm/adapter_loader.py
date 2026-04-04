"""
LoRA adapter loading for fine-tuned models.

Manages loading and using fine-tuned LoRA adapters with the Ollama client.
Provides graceful fallback to base model if adapter loading fails.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


class AdapterNotFoundError(Exception):
    """Raised when no adapter is found."""

    pass


class AdapterLoadError(Exception):
    """Raised when adapter loading fails."""

    pass


def get_adapter_directory() -> Path:
    """
    Get the directory where adapters are stored.

    Returns:
        Path to adapter directory

    Example:
        >>> adapter_dir = get_adapter_directory()
        >>> adapter_dir.exists()
        True
    """
    adapter_dir = Path("models/adapters")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    return adapter_dir


def find_latest_adapter(persona: str | None = None) -> Optional[Path]:
    """
    Find the most recent promoted adapter.

    Looks for adapter checkpoints in models/adapters/ directory.
    Filters by persona if specified, otherwise returns latest across all personas.

    Args:
        persona: Optional persona filter (e.g., "MOMENTUM")

    Returns:
        Path to latest adapter directory, or None if no adapters found

    Example:
        >>> latest = find_latest_adapter(persona="MOMENTUM")
        >>> if latest:
        ...     print(f"Found adapter: {latest.name}")
    """
    adapter_dir = get_adapter_directory()

    # Pattern: adapter-{persona}-{timestamp}.promoted
    # e.g., adapter-MOMENTUM-1640995200000.promoted
    if persona:
        pattern = f"adapter-{persona}-*.promoted"
    else:
        pattern = "adapter-*.promoted"

    # Find all promoted adapters
    promoted_adapters = sorted(adapter_dir.glob(pattern), reverse=True)

    if not promoted_adapters:
        logger.debug(f"No promoted adapters found for persona={persona}")
        return None

    latest = promoted_adapters[0]
    logger.debug(f"Latest adapter: {latest.name}")
    return latest


def load_adapter_metadata(adapter_path: Path) -> dict:
    """
    Load adapter metadata from checkpoint directory.

    Metadata includes:
    - Training timestamp
    - Persona
    - Test set IC
    - Baseline IC
    - IC improvement
    - LoRA configuration
    - Model base

    Args:
        adapter_path: Path to adapter checkpoint directory

    Returns:
        Dictionary with adapter metadata

    Raises:
        AdapterLoadError: If metadata file not found or invalid

    Example:
        >>> metadata = load_adapter_metadata(Path("models/adapters/adapter-MOMENTUM-123.promoted"))
        >>> print(f"IC: {metadata['test_ic']}")
    """
    metadata_file = adapter_path / "metadata.json"

    if not metadata_file.exists():
        raise AdapterLoadError(f"Metadata file not found: {metadata_file}")

    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        return metadata
    except json.JSONDecodeError as e:
        raise AdapterLoadError(f"Invalid metadata file: {e}")
    except Exception as e:
        raise AdapterLoadError(f"Error loading metadata: {e}")


def get_adapter_model_tag(adapter_path: Path) -> str:
    """
    Get the Ollama model tag for loading this adapter.

    Ollama supports loading LoRA adapters via custom model tags.
    This function constructs the appropriate tag.

    Args:
        adapter_path: Path to adapter checkpoint directory

    Returns:
        Ollama model tag string

    Raises:
        AdapterLoadError: If adapter files not found

    Example:
        >>> tag = get_adapter_model_tag(Path("models/adapters/adapter-MOMENTUM-123.promoted"))
        >>> # Returns: "qwen3:8b-lora-adapter-MOMENTUM-123"
    """
    adapter_weights = adapter_path / "adapter_model.safetensors"

    if not adapter_weights.exists():
        raise AdapterLoadError(f"Adapter weights not found: {adapter_weights}")

    # Construct model tag
    # Format: {base_model}-lora-{adapter_name}
    base_model = settings.ollama.generator_model
    adapter_name = adapter_path.name.replace(".promoted", "")

    # Note: Ollama LoRA loading is a planned feature, not yet available
    # This is a placeholder for future implementation
    model_tag = f"{base_model}-lora-{adapter_name}"

    logger.debug(f"Adapter model tag: {model_tag}")
    return model_tag


def should_use_adapter(persona: str | None = None) -> tuple[bool, str]:
    """
    Determine if adapter should be used for signal generation.

    Checks:
    1. Adapters enabled in settings
    2. Latest adapter exists for persona
    3. Adapter metadata is valid

    Args:
        persona: Optional persona filter

    Returns:
        Tuple of (should_use: bool, reason: str)

    Example:
        >>> should_use, reason = should_use_adapter(persona="MOMENTUM")
        >>> if should_use:
        ...     print(f"Using adapter: {reason}")
        ... else:
        ...     print(f"Using base model: {reason}")
    """
    # Check if adapters are enabled (future setting)
    # For now, always use base model since Ollama LoRA support is pending
    use_adapters = False  # TODO: Add to settings when Ollama supports LoRA

    if not use_adapters:
        return False, "Adapters disabled in settings"

    # Find latest adapter
    latest_adapter = find_latest_adapter(persona=persona)

    if latest_adapter is None:
        return False, f"No promoted adapter found for persona={persona}"

    # Validate adapter metadata
    try:
        metadata = load_adapter_metadata(latest_adapter)

        # Check if adapter is recent enough (within last 30 days)
        # This prevents using stale adapters
        import time

        adapter_age_days = (time.time() * 1000 - metadata["timestamp_ms"]) / (
            1000 * 60 * 60 * 24
        )

        if adapter_age_days > 30:
            return False, f"Adapter too old: {adapter_age_days:.1f} days"

        return True, f"Using adapter: {latest_adapter.name}"

    except AdapterLoadError as e:
        return False, f"Adapter validation failed: {e}"


def mark_adapter_promoted(adapter_path: Path) -> None:
    """
    Mark an adapter as promoted by renaming directory.

    Promoted adapters are those that have passed evaluation criteria
    and are ready for production use.

    Args:
        adapter_path: Path to adapter checkpoint directory (without .promoted)

    Raises:
        ValueError: If adapter already promoted

    Example:
        >>> adapter = Path("models/adapters/adapter-MOMENTUM-1640995200000")
        >>> mark_adapter_promoted(adapter)
        >>> # Now: models/adapters/adapter-MOMENTUM-1640995200000.promoted
    """
    if adapter_path.name.endswith(".promoted"):
        raise ValueError(f"Adapter already promoted: {adapter_path}")

    promoted_path = adapter_path.parent / f"{adapter_path.name}.promoted"

    if promoted_path.exists():
        logger.warning(f"Promoted adapter already exists: {promoted_path}")
        return

    adapter_path.rename(promoted_path)
    logger.info(f"Adapter promoted: {promoted_path.name}")


def get_fallback_model() -> str:
    """
    Get the base model tag for fallback.

    Returns:
        Base model tag from settings

    Example:
        >>> fallback = get_fallback_model()
        >>> # Returns: "qwen3:8b"
    """
    return settings.ollama.generator_model
