"""
Training example capture for DPO fine-tuning.

Records complete context including market data, prompts, LLM outputs, critique,
and eventual ground truth for training data generation.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path

from loguru import logger

# Schema version - bump on breaking changes
TRAINING_EXAMPLE_VERSION = "1.0.0"


@dataclass(slots=True)
class TrainingExample:
    """
    Complete training example for DPO fine-tuning.

    Captures all context needed to:
    - Reproduce the signal generation
    - Evaluate prompt quality
    - Compute rewards from realized outcomes
    - Train generator model with DPO

    Version: 1.0.0 (bump on schema changes)
    """

    # Metadata
    version: str = field(default=TRAINING_EXAMPLE_VERSION)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    example_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""  # Shared across examples from same market context (for DPO pairing)

    # Market context (point-in-time)
    symbol: str = ""
    timeframe: str = ""
    timestamp_ms: int = 0  # For reproducibility
    market_regime: str = ""
    indicators: dict = field(default_factory=dict)  # RSI, MACD, BB, etc.

    # Prompts (for prompt engineering analysis)
    task_prompt: str = ""
    full_generator_prompt: str = ""  # Includes persona
    critique_prompt: str = ""

    # Generator outputs
    persona: str = ""
    generator_signal: dict = field(default_factory=dict)  # Structured
    generator_raw_response: str = ""  # Raw LLM output

    # Critic outputs
    critique: dict = field(default_factory=dict)  # Structured
    critic_raw_response: str = ""  # Raw LLM output
    critic_error: str | None = None  # Error message if critic failed

    # Workflow outcome
    was_accepted: bool = False
    acceptance_reason: str = ""
    rejection_reason: str | None = None

    # Ground truth (populated by Verifier in Session 5)
    actual_direction: str | None = None
    realized_return: float | None = None
    max_adverse_excursion: float | None = None  # For risk analysis

    # Reward (computed in Training Layer)
    reward: float | None = None
    reward_components: dict = field(default_factory=dict)  # Per-metric breakdown

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def save(self, output_dir: Path) -> Path:
        """
        Save training example to JSON file.

        Args:
            output_dir: Directory to save examples

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.example_id}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.debug("Training example saved", path=str(filepath))
        return filepath


def is_compatible_version(example_version: str) -> bool:
    """
    Check if training example version is compatible with current code.

    Uses semantic versioning: Major.Minor.Patch
    Compatible if major version matches.

    Args:
        example_version: Version string from training example

    Returns:
        True if compatible, False otherwise
    """
    try:
        current_major = int(TRAINING_EXAMPLE_VERSION.split(".")[0])
        example_major = int(example_version.split(".")[0])
        return current_major == example_major
    except (ValueError, IndexError):
        return False


def load_training_examples(path: Path) -> list[TrainingExample]:
    """
    Load training examples from directory with version filtering.

    Skips examples with incompatible versions.

    Args:
        path: Directory containing training example JSON files

    Returns:
        List of loaded training examples
    """
    if not path.exists():
        logger.warning("Training examples directory does not exist", path=str(path))
        return []

    examples = []
    skipped = 0

    for file in path.glob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)

            version = data.get("version", "0.0.0")
            if not is_compatible_version(version):
                logger.warning("Skipping incompatible example", file=file.name, version=version)
                skipped += 1
                continue

            # Convert ISO timestamp back to datetime if needed
            if isinstance(data.get("created_at"), str):
                # Keep as string for now - will parse if needed
                pass

            examples.append(TrainingExample(**data))

        except Exception as e:
            logger.error("Failed to load training example", file=file.name, error=str(e))
            skipped += 1

    logger.info(
        "Training examples loaded",
        loaded=len(examples),
        skipped=skipped,
        path=str(path),
    )

    return examples


def load_examples_from_jsonl(path: Path) -> list["TrainingExample"]:
    """
    Load training examples from a JSONL file (InferenceQueue output format).

    Each line is a JSON-serialised TrainingExample dict.
    Skips malformed lines and version-incompatible examples.

    Args:
        path: Path to JSONL file

    Returns:
        List of loaded training examples
    """
    if not path.exists():
        logger.warning("JSONL file does not exist", path=str(path))
        return []

    examples = []
    skipped = 0

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line", line=line_num, error=str(e))
                skipped += 1
                continue

            version = data.get("version", "0.0.0")
            if not is_compatible_version(version):
                logger.warning("Skipping incompatible version", line=line_num, version=version)
                skipped += 1
                continue

            try:
                examples.append(TrainingExample(**data))
            except TypeError as e:
                logger.warning("Skipping undeserializable example", line=line_num, error=str(e))
                skipped += 1

    logger.info(
        "Examples loaded from JSONL",
        loaded=len(examples),
        skipped=skipped,
        path=str(path),
    )
    return examples


def filter_by_acceptance(
    examples: list[TrainingExample],
    accepted_only: bool = False,
    rejected_only: bool = False,
) -> list[TrainingExample]:
    """
    Filter training examples by acceptance status.

    Args:
        examples: List of training examples
        accepted_only: Only return accepted signals
        rejected_only: Only return rejected signals

    Returns:
        Filtered list
    """
    if accepted_only:
        return [ex for ex in examples if ex.was_accepted]
    elif rejected_only:
        return [ex for ex in examples if not ex.was_accepted]
    else:
        return examples


def filter_by_regime(
    examples: list[TrainingExample],
    regime: str,
) -> list[TrainingExample]:
    """
    Filter training examples by market regime.

    Args:
        examples: List of training examples
        regime: Market regime string (e.g., "RISK_OFF")

    Returns:
        Filtered list
    """
    return [ex for ex in examples if ex.market_regime == regime]


def filter_by_persona(
    examples: list[TrainingExample],
    persona: str,
) -> list[TrainingExample]:
    """
    Filter training examples by generator persona.

    Args:
        examples: List of training examples
        persona: Persona string (e.g., "momentum")

    Returns:
        Filtered list
    """
    return [ex for ex in examples if ex.persona == persona]


def filter_complete(
    examples: list[TrainingExample],
) -> list[TrainingExample]:
    """
    Filter to examples with ground truth (ready for reward computation).

    Args:
        examples: List of training examples

    Returns:
        Examples with actual_direction and realized_return populated
    """
    return [
        ex for ex in examples
        if ex.actual_direction is not None and ex.realized_return is not None
    ]
