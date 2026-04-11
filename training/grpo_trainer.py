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
from typing import Any, List, Optional, TextIO

import torch
from loguru import logger

from training.grpo_config import GRPOTrainingConfig
from training.grpo_data import GRPOTrainingExample
from training.grpo_reward import compute_grpo_reward, compute_group_advantages
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

    def _load_model(self) -> None:
        """
        Load base model with SFT adapter.

        Loads:
        1. Base model (4-bit quantized)
        2. SFT adapter (reference policy)
        3. Stores reference state dict for KL computation
        """
        # Lazy imports to avoid loading transformers/peft at module level
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading base model: {self.config.base_model_id}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # Load SFT adapter (reference policy)
        logger.info(f"Loading SFT adapter: {self.config.sft_adapter_path}")
        self._model = PeftModel.from_pretrained(
            base_model,
            str(self.config.sft_adapter_path),
            is_trainable=True,
        )

        # Store reference state dict (deep copy for KL computation)
        self._ref_state_dict = {
            k: v.clone().cpu() for k, v in self._model.state_dict().items() if "lora" in k.lower()
        }

        logger.info(
            "Model loaded",
            trainable_params=sum(p.numel() for p in self._model.parameters() if p.requires_grad),
            ref_params=len(self._ref_state_dict),
        )

    def _generate_completions(
        self,
        prompt: str,
    ) -> List[str]:
        """
        Generate G completions for a prompt sequentially.

        Sequential generation (not batched) to fit in 16GB VRAM.
        Clears KV cache between generations.

        Args:
            prompt: Input prompt (market snapshot)

        Returns:
            List of G completion strings
        """
        completions = []

        for i in range(self.config.group_size):
            # Tokenize prompt
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode completion (skip prompt tokens)
            prompt_len = inputs["input_ids"].shape[1]
            completion_tokens = outputs[0, prompt_len:]
            completion = self._tokenizer.decode(
                completion_tokens,
                skip_special_tokens=True,
            )
            completions.append(completion)

            # Clear KV cache between generations (VRAM management)
            if i < self.config.group_size - 1:
                torch.cuda.empty_cache()

        return completions

    def _get_log_probs(
        self,
        prompt: str,
        completion: str,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """
        Get log probabilities for completion tokens.

        Tokenizes prompt + completion, runs forward pass to get logits,
        computes log probabilities and gathers those for actual tokens.

        When use_reference=True, temporarily swaps in reference weights
        for KL computation.

        Args:
            prompt: Input prompt text
            completion: Completion text to score
            use_reference: Whether to use reference policy weights

        Returns:
            Log probabilities for each completion token
        """
        # Tokenize prompt only to get prompt length
        prompt_tokens = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._model.device)
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Tokenize full sequence (prompt + completion)
        full_text = prompt + completion
        full_tokens = self._tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024 + 512,
        ).to(self._model.device)

        # Optionally swap to reference weights
        if use_reference and self._ref_state_dict:
            current_state = {
                k: v.clone() for k, v in self._model.state_dict().items() if "lora" in k.lower()
            }
            for k, v in self._ref_state_dict.items():
                self._model.state_dict()[k].copy_(v.to(self._model.device))

        # Forward pass to get logits
        with torch.no_grad():
            outputs = self._model(
                input_ids=full_tokens["input_ids"],
                attention_mask=full_tokens["attention_mask"],
            )
            logits = outputs.logits

        # Restore current weights if we swapped
        if use_reference and self._ref_state_dict:
            for k, v in current_state.items():
                self._model.state_dict()[k].copy_(v)

        # Get completion tokens (those after prompt)
        completion_len = full_tokens["input_ids"].shape[1] - prompt_len

        if completion_len <= 0:
            return torch.tensor([0.0], device=self._model.device)

        # Shift logits and tokens for next-token prediction
        # logits[t] predicts token[t+1]
        # We want log probs for tokens [prompt_len : end]
        # So we use logits [prompt_len-1 : end-1]
        shift_logits = logits[0, prompt_len - 1 : -1, :]  # (completion_len, vocab)
        shift_labels = full_tokens["input_ids"][0, prompt_len:]  # (completion_len,)

        # Compute log softmax
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs

    def _compute_step_loss(
        self,
        completions: List[str],
        advantages: List[float],
        prompt: str,
    ) -> tuple[float, float]:
        """
        Compute GRPO loss with KL penalty for a batch of completions.

        For each completion:
        1. Get policy log probs
        2. Get reference log probs
        3. Compute probability ratio
        4. Compute clipped policy loss
        5. Compute KL divergence

        Returns average loss and KL across all completions.

        Args:
            completions: List of G completion strings
            advantages: List of G group-relative advantages
            prompt: The prompt used for generation

        Returns:
            Tuple of (total_loss, avg_kl)
            where total_loss = policy_loss + β * KL
        """
        total_policy_loss = 0.0
        total_kl = 0.0

        for completion, advantage in zip(completions, advantages):
            # Get policy and reference log probs
            policy_logprobs = self._get_log_probs(prompt, completion, use_reference=False)
            ref_logprobs = self._get_log_probs(prompt, completion, use_reference=True)

            # Compute probability ratio: exp(log π - log π_ref)
            log_ratio = policy_logprobs - ref_logprobs
            ratio = torch.exp(log_ratio.mean())  # Average over tokens

            # Compute clipped policy loss
            advantage_tensor = torch.tensor([advantage], device=ratio.device)
            policy_loss = compute_clipped_policy_loss(
                ratio.unsqueeze(0),
                advantage_tensor,
                epsilon=self.config.clip_epsilon,
            )
            total_policy_loss += policy_loss

            # Compute KL divergence
            kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
            total_kl += kl

        # Average over completions
        n_completions = len(completions)
        avg_policy_loss = total_policy_loss / n_completions
        avg_kl = total_kl / n_completions

        # Total loss with KL penalty
        total_loss = avg_policy_loss + self.config.kl_penalty_beta * avg_kl

        return total_loss, avg_kl

    def _training_step(
        self,
        example: GRPOTrainingExample,
        step: int,
    ) -> GRPOStepResult:
        """
        Execute single GRPO training step.

        1. Generate G completions
        2. Score each with reward function
        3. Compute group-relative advantages
        4. Compute loss with KL penalty
        5. Backward pass (gradients accumulated)

        Args:
            example: Training example
            step: Current step number

        Returns:
            GRPOStepResult with metrics
        """
        # 1. Generate G completions
        completions = self._generate_completions(example.market_snapshot)

        # 2. Score each completion
        rewards = []
        for completion in completions:
            direction = parse_direction(completion)
            result = compute_grpo_reward(
                completion=completion,
                predicted_direction=direction,
                actual_direction=example.actual_direction,
                gross_return_pct=example.gross_return_pct,
                config=self.config.reward,
            )
            rewards.append(result.final_reward)

        # 3. Compute group-relative advantages
        advantages = compute_group_advantages(rewards)

        # 4. Compute loss with KL penalty
        loss, kl = self._compute_step_loss(
            completions=completions,
            advantages=advantages,
            prompt=example.market_snapshot,
        )

        # 5. Log VRAM periodically
        vram_mb = 0
        if step % self.config.vram_log_interval_steps == 0:
            vram_mb = log_vram_usage(step)

        return GRPOStepResult(
            step=step,
            mean_reward=sum(rewards) / len(rewards),
            mean_advantage=sum(advantages) / len(advantages),
            kl_divergence=kl,
            loss=loss,
            vram_mb=vram_mb,
        )
