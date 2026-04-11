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

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

import torch
from loguru import logger

from training.grpo_config import GRPOTrainingConfig
from training.grpo_data import GRPOTrainingExample
from training.process_lock import check_can_train
from training.vram_check import check_vram_availability

# Constants
MIN_VRAM_GB = 9.0
MAX_VRAM_GB = 14.0
STOP_FILE_PATH = Path("execution/state/STOP")


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


def run_grpo_preflight(
    examples: list[GRPOTrainingExample],
) -> tuple[bool, str]:
    """
    Run all pre-flight checks before GRPO training.

    Order (fail-fast, cheap to expensive):
    1. Data check: examples list non-empty
    2. Temporal check: examples sorted by timestamp_ms
    3. VRAM check: check_vram_availability(min_free_gb=9.0)
    4. Lock check: check_can_train() returns True
    5. OLLAMA_KEEP_ALIVE=0 enforced
    6. STOP file check: execution/state/STOP does not exist

    Args:
        examples: List of training examples

    Returns:
        Tuple of (can_train: bool, reason: str)
    """
    # 1. Data check
    if not examples:
        return False, "Examples list is empty"

    # 2. Temporal check
    timestamps = [e.timestamp_ms for e in examples]
    if timestamps != sorted(timestamps):
        return False, "Examples not sorted by timestamp_ms (temporal ordering required)"

    # 3. VRAM check
    vram_status = check_vram_availability(min_free_gb=MIN_VRAM_GB)
    if not vram_status.can_train:
        return False, f"VRAM insufficient: {vram_status.reason}"

    # 4. Lock check
    can_train, lock_reason = check_can_train()
    if not can_train:
        return False, f"Lock unavailable: {lock_reason}"

    # 5. Enforce OLLAMA_KEEP_ALIVE=0
    os.environ["OLLAMA_KEEP_ALIVE"] = "0"
    logger.debug("OLLAMA_KEEP_ALIVE=0 enforced")

    # 6. STOP file check
    if STOP_FILE_PATH.exists():
        return False, "STOP file exists - refusing to train"

    logger.info(
        "GRPO preflight checks passed",
        num_examples=len(examples),
        vram_free_gb=f"{vram_status.free_mb / 1024:.1f}",
    )

    return True, "Ready to train"


def log_vram_usage(step: int) -> int:
    """
    Log VRAM usage and warn if exceeding threshold.

    Args:
        step: Current training step (for logging context)

    Returns:
        Current VRAM usage in MB
    """
    if not torch.cuda.is_available():
        return 0

    vram_bytes = torch.cuda.memory_allocated()
    vram_mb = vram_bytes // (1024 * 1024)
    vram_gb = vram_mb / 1024

    if vram_gb > MAX_VRAM_GB:
        logger.warning(
            f"VRAM exceeded {MAX_VRAM_GB}GB threshold",
            step=step,
            vram_gb=f"{vram_gb:.2f}",
            vram_mb=vram_mb,
        )
    else:
        logger.debug(
            "VRAM usage",
            step=step,
            vram_gb=f"{vram_gb:.2f}",
        )

    return vram_mb


def compute_config_hash(config: GRPOTrainingConfig) -> str:
    """
    Compute deterministic hash of training config.

    Used for reproducibility verification - checkpoints with
    different config hashes are incompatible.

    Args:
        config: GRPO training configuration

    Returns:
        16-character hex hash string
    """
    # Extract key parameters that affect training
    config_dict = {
        "group_size": config.group_size,
        "kl_penalty_beta": config.kl_penalty_beta,
        "clip_epsilon": config.clip_epsilon,
        "learning_rate": config.learning_rate,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "reward_decision_weight": config.reward.decision_weight,
        "reward_false_bullish_penalty": config.reward.false_bullish_penalty,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    full_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return full_hash[:16]


def save_grpo_checkpoint(
    model: Any,
    checkpoint_dir: Path,
    step: int,
    config: GRPOTrainingConfig,
    metrics: dict[str, float],
) -> Path:
    """
    Save GRPO checkpoint with metadata.

    Args:
        model: PEFT model to save
        checkpoint_dir: Directory for checkpoint
        step: Current training step
        config: Training configuration
        metrics: Current training metrics

    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model.save_pretrained(str(checkpoint_dir))

    # Save metadata
    metadata = {
        "step": step,
        "timestamp_ms": int(time.time() * 1000),
        "config_hash": compute_config_hash(config),
        **metrics,
    }

    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Checkpoint saved",
        checkpoint_dir=str(checkpoint_dir),
        step=step,
    )

    return checkpoint_dir


def compute_kl_divergence(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> float:
    """
    Compute KL divergence between policy and reference distributions.

    Uses the approximation: KL(π || π_ref) ≈ mean(log π - log π_ref)

    This is the standard approximation used in PPO/GRPO when we have
    log probabilities from both distributions.

    Args:
        policy_logprobs: Log probabilities from current policy
        ref_logprobs: Log probabilities from reference policy

    Returns:
        KL divergence (scalar, non-negative)
    """
    # KL divergence approximation
    kl = (policy_logprobs - ref_logprobs).mean().item()
    # KL should be non-negative (numerical errors can cause small negatives)
    return max(0.0, kl)


def compute_clipped_policy_loss(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    epsilon: float = 0.2,
) -> float:
    """
    Compute PPO-style clipped policy loss.

    loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)

    The clipping prevents too large policy updates, stabilizing training.

    Args:
        ratio: π(a|s) / π_ref(a|s) probability ratios
        advantage: Group-relative advantages
        epsilon: Clipping parameter (default: 0.2)

    Returns:
        Clipped policy loss (scalar)
    """
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    unclipped_loss = ratio * advantage
    clipped_loss = clipped_ratio * advantage
    # Take the minimum (most conservative update)
    loss = -torch.min(unclipped_loss, clipped_loss).mean()
    return loss.item()


class GRPOLogger:
    """JSONL logger for GRPO training metrics."""

    def __init__(self, log_dir: Path) -> None:
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"grpo_{timestamp}.jsonl"
        self._file: TextIO = open(self.log_path, "w")
        logger.info(f"GRPO training log: {self.log_path}")

    def log_step(self, result: GRPOStepResult) -> None:
        """
        Log a training step result.

        Args:
            result: Step result to log
        """
        self._file.write(json.dumps(result.to_dict()) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        self._file.close()


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) trainer.

    Implements the DeepSeek-R1 GRPO algorithm with:
    - Sequential G=4 completion generation (VRAM constraint)
    - Reference model weight swapping for KL penalty
    - Asymmetric reward computation
    - Checkpointing every 500 steps

    Usage:
        trainer = GRPOTrainer()
        result = trainer.train(examples)
    """

    def __init__(self, config: Optional[GRPOTrainingConfig] = None) -> None:
        """
        Initialize GRPO trainer.

        Args:
            config: Training configuration (uses defaults if None)
        """
        self.config = config or GRPOTrainingConfig()

        # Model components (lazy loaded in train())
        self._model: Any = None  # PeftModel when loaded
        self._tokenizer: Any = None  # AutoTokenizer when loaded
        self._ref_state_dict: Optional[dict[str, torch.Tensor]] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

        # Logging
        self._logger: Optional[GRPOLogger] = None
