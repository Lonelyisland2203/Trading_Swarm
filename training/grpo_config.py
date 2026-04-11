"""
GRPO (Group Relative Policy Optimization) configuration.

All tunable hyperparameters in one place for the autoresearch loop.

Reference: DeepSeek-R1 paper, Section 2.3
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GRPORewardConfig:
    """
    Reward function configuration for GRPO training.

    Asymmetric penalties ensure false bullish signals are penalized
    more heavily than false bearish — matching real-world trading costs
    where buying into a decline is typically more costly than missing
    an upturn.
    """

    # Component weights (must sum to 1.0)
    decision_weight: float = 0.6
    structure_weight: float = 0.2
    directional_weight: float = 0.2

    # Asymmetric penalties for decision reward
    # False bullish (predicted long, price down): multiply |net_return| by this
    false_bullish_penalty: float = 1.5
    # False bearish (predicted short, price up): multiply |net_return| by this
    false_bearish_penalty: float = 0.8

    # True predictions use 1.0 multiplier (no penalty/bonus beyond actual return)
    true_bullish_multiplier: float = 1.0
    true_bearish_multiplier: float = 1.0

    # Structure reward for proper reasoning format
    structure_reward_value: float = 0.2  # Reward if all sections present

    # Clipping bounds (applied to each component before combination)
    clip_min: float = -1.0
    clip_max: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        total_weight = self.decision_weight + self.structure_weight + self.directional_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total_weight:.4f} "
                f"(decision={self.decision_weight}, structure={self.structure_weight}, "
                f"directional={self.directional_weight})"
            )


@dataclass(frozen=True)
class GRPOLoRAConfig:
    """LoRA configuration for GRPO training."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # Attention
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP
    )
    bias: str = "none"


@dataclass(frozen=True)
class GRPOTrainingConfig:
    """
    Complete GRPO training configuration.

    GRPO uses group-relative advantages instead of a learned value function.
    For each prompt, G completions are generated and the advantage for each
    is computed relative to the group mean reward.

    Key differences from PPO:
    - No value network (baseline = group mean)
    - KL penalty instead of constraint (β term)
    - Same clipping as PPO (ε parameter)
    """

    # GRPO algorithm parameters
    group_size: int = 4  # G: number of completions per prompt
    kl_penalty_beta: float = 0.04  # β: KL divergence penalty coefficient
    clip_epsilon: float = 0.2  # ε: PPO-style ratio clipping

    # Learning rate
    learning_rate: float = 2e-05

    # Batch configuration
    batch_size: int = 1  # Per-device batch size
    gradient_accumulation_steps: int = 16  # Effective batch = 16

    # Training duration
    max_steps: int = 5000

    # Checkpointing
    checkpoint_interval_steps: int = 500
    checkpoint_dir: Path = field(default_factory=lambda: Path("adapters/grpo_checkpoints"))

    # Final adapter output
    output_dir: Path = field(default_factory=lambda: Path("adapters/grpo_latest"))

    # Reference model (SFT adapter from Stage 1)
    sft_adapter_path: Path = field(default_factory=lambda: Path("adapters/sft_base"))

    # Base model (HuggingFace ID)
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Generation parameters for completions
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # VRAM management
    max_vram_gb: float = 14.0  # Log warning if exceeded
    vram_log_interval_steps: int = 100

    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True

    # Logging
    log_dir: Path = field(default_factory=lambda: Path("training/logs"))

    # LoRA configuration
    lora: GRPOLoRAConfig = field(default_factory=GRPOLoRAConfig)

    # Reward configuration
    reward: GRPORewardConfig = field(default_factory=GRPORewardConfig)

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


def load_grpo_config(
    overrides: dict | None = None,
) -> GRPOTrainingConfig:
    """
    Load GRPO configuration with optional overrides.

    Args:
        overrides: Dictionary of parameter overrides

    Returns:
        GRPOTrainingConfig with applied overrides

    Example:
        >>> config = load_grpo_config({"max_steps": 1000, "learning_rate": 1e-5})
        >>> config.max_steps
        1000
        >>> config.learning_rate
        1e-5
    """
    if overrides is None:
        return GRPOTrainingConfig()

    # Handle nested configs
    reward_overrides = {}
    lora_overrides = {}
    training_overrides = {}

    for key, value in overrides.items():
        if key.startswith("reward."):
            reward_overrides[key[7:]] = value
        elif key.startswith("lora."):
            lora_overrides[key[5:]] = value
        else:
            training_overrides[key] = value

    # Build nested configs
    reward_config = GRPORewardConfig(**reward_overrides) if reward_overrides else GRPORewardConfig()
    lora_config = GRPOLoRAConfig(**lora_overrides) if lora_overrides else GRPOLoRAConfig()

    # Build training config
    training_overrides["reward"] = reward_config
    training_overrides["lora"] = lora_config

    return GRPOTrainingConfig(**training_overrides)
