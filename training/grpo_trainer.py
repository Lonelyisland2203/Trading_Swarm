"""
GRPO (Group Relative Policy Optimization) training loop.

Implements the DeepSeek-R1 GRPO algorithm with:
- Sequential G=4 completion generation (VRAM constraint)
- Reference model weight swapping for KL penalty
- Asymmetric reward computation via grpo_reward.py
- Checkpointing every 500 steps

CRITICAL: This module runs in Process B (training), which is mutually exclusive
with Process A (inference). Never run both simultaneously.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# Direction keywords to look for in completions
_LONG_KEYWORDS = {"LONG", "HIGHER", "BUY", "BULLISH"}
_SHORT_KEYWORDS = {"SHORT", "LOWER", "SELL", "BEARISH"}
_FLAT_KEYWORDS = {"FLAT", "NEUTRAL", "HOLD", "WAIT"}

# Pattern to find DECISION section
_DECISION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?\s*DECISION\s*(?:\*{1,2})?[:\s](.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_direction(completion: str) -> str:
    """
    Extract trading direction from completion text.

    Looks for the DECISION section and parses direction keyword.
    Returns "FLAT" if unparseable (conservative default).

    Args:
        completion: Generated completion text

    Returns:
        Normalized direction: "LONG", "SHORT", or "FLAT"
    """
    # Try to find DECISION section
    match = _DECISION_PATTERN.search(completion)
    if not match:
        # No DECISION section found - default to FLAT (conservative)
        logger.warning("No DECISION section found in completion, defaulting to FLAT")
        return "FLAT"

    decision_text = match.group(1).upper()

    # Check for direction keywords (order matters: check specific before generic)
    for keyword in _LONG_KEYWORDS:
        if keyword in decision_text:
            return "LONG"

    for keyword in _SHORT_KEYWORDS:
        if keyword in decision_text:
            return "SHORT"

    for keyword in _FLAT_KEYWORDS:
        if keyword in decision_text:
            return "FLAT"

    # DECISION section found but no direction keyword - default to FLAT
    logger.warning("Could not parse direction from DECISION section, defaulting to FLAT")
    return "FLAT"


@dataclass
class GRPOStepResult:
    """Result of a single GRPO training step."""

    step: int
    mean_reward: float
    mean_advantage: float
    kl_divergence: float
    loss: float
    vram_mb: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON logging."""
        return {
            "step": self.step,
            "mean_reward": self.mean_reward,
            "mean_advantage": self.mean_advantage,
            "kl": self.kl_divergence,
            "loss": self.loss,
            "vram_mb": self.vram_mb,
            "timestamp": int(time.time()),
        }


@dataclass
class GRPOTrainingResult:
    """Result of full GRPO training run."""

    success: bool
    adapter_path: Optional[Path]
    steps_completed: int
    final_metrics: dict[str, Any]
    error: Optional[str]
