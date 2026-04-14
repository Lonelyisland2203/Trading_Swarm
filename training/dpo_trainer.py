"""
DPO Training Pipeline for Qwen3-8B fine-tuning.

CRITICAL: This module runs in Process B (training), which is mutually exclusive
with Process A (inference). Never run both simultaneously.

Architecture Decisions (Session 8):
1. Uses direct transformers + PEFT + bitsandbytes stack (not Unsloth)
   - Better debuggability, mature error handling, no extra dependencies
2. Downloads from HuggingFace (Ollama cache is GGUF, incompatible)
3. Saves final adapter only (31 steps too few for intermediate checkpoints)
4. Pre-flight checks: pairs -> temporal -> VRAM -> lock -> load
5. Automatic promotion with cooldown safeguards
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

from config.settings import settings
from training.dpo_eval import (
    AdapterEvaluation,
    should_promote_adapter,
)
from training.dpo_export import PreferencePair, export_to_huggingface_format
from training.process_lock import ProcessLockError, acquire_training_lock, check_can_train
from training.vram_check import VRAMStatus, check_vram_availability
from training.walk_forward import (
    TemporalSplitError,
    WalkForwardSplit,
    create_walk_forward_splits,
    merge_train_and_replay,
    validate_temporal_split,
)


# Promotion safeguards - prevent runaway model updates
PROMOTION_COOLDOWN_HOURS = 24
MAX_PROMOTIONS_PER_WEEK = 3
PROMOTIONS_LOG_FILE = Path("models/adapters/promotions.jsonl")


class CudaCacheFlushCallback(TrainerCallback):
    """
    Flush the CUDA memory allocator cache after every optimizer step.

    Without this, near-maxed VRAM (16 GB on RTX 5070 Ti) causes progressive
    slowdown: fragmented free blocks force increasingly expensive coalescing
    on each gradient accumulation sub-step. Calling empty_cache() after each
    optimizer step returns fragmented free blocks to the pool, keeping step
    time constant across the training run.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration derived from settings."""

    # Model
    base_model_id: str
    max_length: int

    # LoRA
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

    # DPO
    dpo_beta: float
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int

    # Paths
    adapter_dir: Path
    output_dir: Path

    @classmethod
    def from_settings(cls) -> "TrainingConfig":
        """Create config from application settings."""
        return cls(
            base_model_id=settings.dpo.base_model_id,
            max_length=settings.dpo.max_length,
            lora_rank=settings.dpo.lora_rank,
            lora_alpha=settings.dpo.lora_alpha,
            lora_dropout=settings.dpo.lora_dropout,
            lora_target_modules=list(settings.dpo.lora_target_modules),
            dpo_beta=settings.dpo.dpo_beta,
            learning_rate=settings.dpo.learning_rate,
            batch_size=settings.dpo.batch_size,
            gradient_accumulation_steps=settings.dpo.gradient_accumulation_steps,
            num_epochs=settings.dpo.num_epochs,
            adapter_dir=settings.dpo.adapter_dir,
            output_dir=settings.output_dir,
        )


@dataclass
class TrainingResult:
    """Result of a DPO training run."""

    success: bool
    adapter_path: Optional[Path]
    evaluation: Optional[AdapterEvaluation]
    promoted: bool
    promotion_reason: str
    training_time_seconds: float
    num_training_pairs: int
    num_test_pairs: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
            "training_time_seconds": self.training_time_seconds,
            "num_training_pairs": self.num_training_pairs,
            "num_test_pairs": self.num_test_pairs,
            "error": self.error,
        }
        if self.evaluation:
            result["evaluation"] = {
                "ic": self.evaluation.ic,
                "ic_pvalue": self.evaluation.ic_pvalue,
                "brier_score": self.evaluation.brier_score,
                "num_examples": self.evaluation.num_examples,
            }
        return result


class DPOTrainingError(Exception):
    """Raised when DPO training fails."""

    pass


class InsufficientDataError(DPOTrainingError):
    """Raised when there's not enough data for training."""

    pass


class VRAMError(DPOTrainingError):
    """Raised when VRAM is insufficient."""

    pass


def _check_promotion_cooldown() -> tuple[bool, str]:
    """
    Check if promotion is allowed based on cooldown and rate limits.

    Returns:
        Tuple of (can_promote: bool, reason: str)
    """
    if not PROMOTIONS_LOG_FILE.exists():
        return True, "No previous promotions"

    now = time.time()
    cooldown_threshold = now - (PROMOTION_COOLDOWN_HOURS * 3600)
    week_threshold = now - (7 * 24 * 3600)

    recent_promotions = []
    week_promotions = 0

    try:
        with open(PROMOTIONS_LOG_FILE) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    timestamp = record.get("timestamp", 0)

                    if timestamp > cooldown_threshold:
                        recent_promotions.append(record)

                    if timestamp > week_threshold:
                        week_promotions += 1

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error reading promotions log: {e}")
        return True, "Promotions log unreadable, allowing promotion"

    # Check cooldown
    if recent_promotions:
        last_promotion = max(recent_promotions, key=lambda r: r.get("timestamp", 0))
        hours_ago = (now - last_promotion.get("timestamp", 0)) / 3600
        return (
            False,
            f"Cooldown active: last promotion {hours_ago:.1f}h ago (need {PROMOTION_COOLDOWN_HOURS}h)",
        )

    # Check weekly rate limit
    if week_promotions >= MAX_PROMOTIONS_PER_WEEK:
        return (
            False,
            f"Weekly limit reached: {week_promotions}/{MAX_PROMOTIONS_PER_WEEK} promotions this week",
        )

    return True, "Promotion allowed"


def _log_promotion(adapter_path: Path, evaluation: AdapterEvaluation) -> None:
    """Log a successful promotion for rate limiting."""
    PROMOTIONS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": time.time(),
        "adapter_path": str(adapter_path),
        "ic": evaluation.ic,
        "brier_score": evaluation.brier_score,
        "datetime": datetime.now().isoformat(),
    }

    with open(PROMOTIONS_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _create_quantization_config() -> BitsAndBytesConfig:
    """
    Create 4-bit quantization config for QLoRA.

    Uses NF4 quantization with double quantization for maximum VRAM efficiency.
    bfloat16 compute dtype for numerical stability on modern GPUs.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,  # Better precision than float16
        bnb_4bit_use_double_quant=True,  # Nested quantization saves ~0.4GB
    )


def _create_lora_config(config: TrainingConfig) -> LoraConfig:
    """
    Create LoRA configuration for DPO training.

    Targets attention + MLP layers for maximum expressiveness within VRAM budget.
    """
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",  # No bias training for stability
        task_type=TaskType.CAUSAL_LM,
    )


def _create_dpo_config(config: TrainingConfig, output_dir: Path) -> DPOConfig:
    """
    Create DPO training configuration.

    Key settings:
    - precompute_ref_log_probs=True: Computes reference logprobs upfront, saves VRAM
    - gradient_checkpointing + use_reentrant=False: Non-reentrant checkpointing avoids
      stale graph reference accumulation that causes progressive slowdown
    - max_length=1200: Our prompts are ~968 tokens + ~150 token responses.
      Padding to 2048 wastes 4x attention compute and fills VRAM with padding ops.
    - optim=adamw_bnb_8bit: bitsandbytes 8-bit Adam reduces optimizer state memory
      by ~4x vs full-precision AdamW, freeing VRAM headroom.
    - bf16=True: Better numerical stability than fp16 on Blackwell
    """
    return DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.dpo_beta,
        loss_type="sigmoid",  # Standard DPO loss
        max_length=config.max_length,
        truncation_mode="keep_start",  # Keep prompt start when truncating
        num_train_epochs=config.num_epochs,
        # Memory optimization
        precompute_ref_log_probs=True,  # Avoids loading ref model during training
        gradient_checkpointing=True,
        # Non-reentrant checkpointing: prevents stale graph reference accumulation
        # that causes progressive slowdown across steps on near-maxed VRAM.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",  # 8-bit Adam: ~4x smaller optimizer state vs fp32 AdamW
        bf16=True,
        # Logging
        logging_steps=1,
        logging_first_step=True,
        # Checkpointing - save only at end
        save_strategy="no",
        # Evaluation
        eval_strategy="no",  # We evaluate manually after training
        # Optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Disable integrations we don't need
        report_to="none",
        # Seed for reproducibility
        seed=42,
    )


def _load_model_and_tokenizer(
    config: TrainingConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load quantized model and tokenizer from HuggingFace.

    Downloads on first run (~16GB), uses cache subsequently.
    """
    logger.info(
        "Loading model and tokenizer",
        model_id=config.base_model_id,
    )

    bnb_config = _create_quantization_config()

    # Load tokenizer first (fast)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        trust_remote_code=True,  # Qwen requires this
        padding_side="left",  # Required for batch generation
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Automatic GPU placement
        trust_remote_code=True,
        attn_implementation="sdpa",  # Scaled dot product attention for efficiency
    )

    # Prepare for k-bit training (enables gradient computation on quantized model)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    logger.info(
        "Model loaded successfully",
        model_dtype=str(model.dtype),
        device=str(model.device),
    )

    return model, tokenizer


def _prepare_dataset(
    preference_pairs: List[PreferencePair],
    tokenizer: AutoTokenizer,
) -> Dataset:
    """
    Convert PreferencePairs to HuggingFace Dataset for DPOTrainer.

    Uses the existing export_to_huggingface_format function which produces
    {"prompt": str, "chosen": str, "rejected": str} format.
    """
    # Convert to HuggingFace format
    hf_data = export_to_huggingface_format(preference_pairs)

    # Create Dataset
    dataset = Dataset.from_list(hf_data)

    logger.info(
        "Dataset prepared",
        num_examples=len(dataset),
        columns=dataset.column_names,
    )

    return dataset


def _save_adapter_with_metadata(
    trainer: DPOTrainer,
    output_dir: Path,
    config: TrainingConfig,
    split: WalkForwardSplit,
    training_time_seconds: float,
) -> Path:
    """
    Save LoRA adapter and metadata to disk.

    Adapter files:
    - adapter_model.safetensors (~100-200 MB)
    - adapter_config.json
    - metadata.json (our custom metadata)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights
    trainer.save_model(str(output_dir))

    # Save custom metadata
    metadata = {
        "timestamp_ms": int(time.time() * 1000),
        "base_model_id": config.base_model_id,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "dpo_beta": config.dpo_beta,
        "learning_rate": config.learning_rate,
        "num_training_pairs": split.total_pairs,
        "num_test_pairs": len(split.test_pairs),
        "train_start_ms": split.train_start_ms,
        "train_end_ms": split.train_end_ms,
        "test_start_ms": split.test_start_ms,
        "test_end_ms": split.test_end_ms,
        "training_time_seconds": training_time_seconds,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Adapter saved",
        output_dir=str(output_dir),
        metadata=metadata,
    )

    return output_dir


def run_preflight_checks(
    preference_pairs: List[PreferencePair],
    min_vram_gb: float = 9.0,
) -> tuple[WalkForwardSplit, VRAMStatus]:
    """
    Run all pre-flight checks before training.

    Order (fail-fast, cheap to expensive):
    1. Check sufficient preference pairs
    2. Validate temporal split
    3. Check VRAM availability
    4. Training lock is acquired separately (in train_dpo)

    Args:
        preference_pairs: All available preference pairs
        min_vram_gb: Minimum VRAM required for training

    Returns:
        Tuple of (walk_forward_split, vram_status)

    Raises:
        InsufficientDataError: If not enough data
        TemporalSplitError: If temporal split fails
        VRAMError: If insufficient VRAM
    """
    # Check 1: Sufficient data
    min_pairs = settings.dpo.train_window + settings.dpo.test_window
    if len(preference_pairs) < min_pairs:
        raise InsufficientDataError(
            f"Insufficient preference pairs: {len(preference_pairs)} < {min_pairs} required"
        )

    logger.info(
        "Data check passed",
        total_pairs=len(preference_pairs),
        required=min_pairs,
    )

    # Check 2: Temporal split validation
    try:
        split = create_walk_forward_splits(
            preference_pairs,
            train_window=settings.dpo.train_window,
            test_window=settings.dpo.test_window,
            replay_ratio=settings.dpo.replay_ratio,
            replay_buffer_size=settings.dpo.replay_buffer_size,
            min_training_pairs=settings.dpo.min_training_pairs,
        )
        validate_temporal_split(split)
    except TemporalSplitError as e:
        raise TemporalSplitError(f"Temporal split validation failed: {e}")

    logger.info(
        "Temporal split validated",
        train_pairs=len(split.train_pairs),
        test_pairs=len(split.test_pairs),
        replay_pairs=len(split.replay_pairs),
    )

    # Check 3: VRAM availability
    vram_status = check_vram_availability(min_free_gb=min_vram_gb)
    if not vram_status.can_train:
        raise VRAMError(f"Insufficient VRAM: {vram_status.reason}")

    logger.info(
        "VRAM check passed",
        free_gb=f"{vram_status.free_mb / 1024:.1f}",
        required_gb=f"{min_vram_gb:.1f}",
    )

    return split, vram_status


def train_dpo(
    preference_pairs: List[PreferencePair],
    baseline_eval: Optional[AdapterEvaluation] = None,
    force_training: bool = False,
) -> TrainingResult:
    """
    Execute full DPO training pipeline.

    Pipeline:
    1. Pre-flight checks (data, temporal, VRAM)
    2. Acquire training lock
    3. Load model and tokenizer
    4. Prepare dataset
    5. Train with DPOTrainer
    6. Save adapter
    7. Evaluate on test set
    8. Promote if criteria met

    Args:
        preference_pairs: All preference pairs (will be split temporally)
        baseline_eval: Optional baseline evaluation for comparison
        force_training: Skip promotion cooldown check

    Returns:
        TrainingResult with all metrics and paths

    Raises:
        DPOTrainingError: If training fails
        ProcessLockError: If lock cannot be acquired
    """
    start_time = time.time()
    config = TrainingConfig.from_settings()

    logger.info(
        "Starting DPO training pipeline",
        num_pairs=len(preference_pairs),
        base_model=config.base_model_id,
    )

    # Pre-flight checks (before acquiring lock)
    try:
        split, vram_status = run_preflight_checks(preference_pairs)
    except (InsufficientDataError, TemporalSplitError, VRAMError) as e:
        logger.error(f"Pre-flight check failed: {e}")
        return TrainingResult(
            success=False,
            adapter_path=None,
            evaluation=None,
            promoted=False,
            promotion_reason=str(e),
            training_time_seconds=time.time() - start_time,
            num_training_pairs=0,
            num_test_pairs=0,
            error=str(e),
        )

    # Check training lock availability (before acquiring)
    can_train, lock_reason = check_can_train()
    if not can_train:
        logger.error(f"Cannot acquire training lock: {lock_reason}")
        return TrainingResult(
            success=False,
            adapter_path=None,
            evaluation=None,
            promoted=False,
            promotion_reason=lock_reason,
            training_time_seconds=time.time() - start_time,
            num_training_pairs=split.total_pairs,
            num_test_pairs=len(split.test_pairs),
            error=f"Lock unavailable: {lock_reason}",
        )

    # Acquire training lock and execute training
    try:
        with acquire_training_lock():
            logger.info("Training lock acquired, beginning training")

            # Generate unique adapter name
            timestamp_ms = int(time.time() * 1000)
            adapter_name = f"adapter-DPO-{timestamp_ms}"
            adapter_output_dir = config.adapter_dir / adapter_name

            # Load model and tokenizer
            model, tokenizer = _load_model_and_tokenizer(config)

            # Prepare training data (train + replay, shuffled)
            training_pairs = merge_train_and_replay(split, shuffle=True)
            train_dataset = _prepare_dataset(training_pairs, tokenizer)

            # Create configs
            lora_config = _create_lora_config(config)
            dpo_config = _create_dpo_config(config, adapter_output_dir)

            # Initialize DPOTrainer
            # Note: With precompute_ref_log_probs=True, reference model logprobs
            # are computed once before training, avoiding VRAM overhead.
            # CudaCacheFlushCallback prevents progressive slowdown on near-maxed VRAM.
            trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                peft_config=lora_config,
                callbacks=[CudaCacheFlushCallback()],
            )

            logger.info(
                "Starting DPO training",
                num_examples=len(train_dataset),
                effective_batch_size=config.batch_size * config.gradient_accumulation_steps,
            )

            # Train
            train_result = trainer.train()

            training_time = time.time() - start_time
            logger.info(
                "Training complete",
                training_time_seconds=f"{training_time:.1f}",
                train_loss=train_result.training_loss,
            )

            # Save adapter with metadata
            adapter_path = _save_adapter_with_metadata(
                trainer=trainer,
                output_dir=adapter_output_dir,
                config=config,
                split=split,
                training_time_seconds=training_time,
            )

            # Clean up model from GPU to free VRAM for evaluation
            del model
            del trainer
            torch.cuda.empty_cache()

            # Evaluate on test set
            # Note: This requires generating predictions with the fine-tuned model
            # For now, we return a placeholder - full eval requires inference
            logger.info("Training complete, adapter saved for evaluation")

            # Check promotion eligibility
            can_promote, cooldown_reason = _check_promotion_cooldown()
            if not can_promote and not force_training:
                logger.info(f"Promotion blocked by cooldown: {cooldown_reason}")
                return TrainingResult(
                    success=True,
                    adapter_path=adapter_path,
                    evaluation=None,  # Eval requires separate inference pass
                    promoted=False,
                    promotion_reason=f"Training succeeded but promotion blocked: {cooldown_reason}",
                    training_time_seconds=training_time,
                    num_training_pairs=split.total_pairs,
                    num_test_pairs=len(split.test_pairs),
                )

            # Mark as candidate (not promoted until evaluated)
            candidate_path = adapter_path.parent / f"{adapter_path.name}.candidate"
            adapter_path.rename(candidate_path)

            return TrainingResult(
                success=True,
                adapter_path=candidate_path,
                evaluation=None,  # Will be filled by evaluation pass
                promoted=False,  # Promotion happens after evaluation
                promotion_reason="Training complete, awaiting evaluation",
                training_time_seconds=training_time,
                num_training_pairs=split.total_pairs,
                num_test_pairs=len(split.test_pairs),
            )

    except ProcessLockError as e:
        logger.error(f"Failed to acquire training lock: {e}")
        return TrainingResult(
            success=False,
            adapter_path=None,
            evaluation=None,
            promoted=False,
            promotion_reason=str(e),
            training_time_seconds=time.time() - start_time,
            num_training_pairs=split.total_pairs if split else 0,
            num_test_pairs=len(split.test_pairs) if split else 0,
            error=str(e),
        )
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        return TrainingResult(
            success=False,
            adapter_path=None,
            evaluation=None,
            promoted=False,
            promotion_reason=f"Training error: {e}",
            training_time_seconds=time.time() - start_time,
            num_training_pairs=split.total_pairs if split else 0,
            num_test_pairs=len(split.test_pairs) if split else 0,
            error=str(e),
        )


def promote_adapter(
    candidate_path: Path,
    evaluation: AdapterEvaluation,
    baseline_eval: Optional[AdapterEvaluation] = None,
) -> tuple[bool, str, Optional[Path]]:
    """
    Evaluate and promote a candidate adapter to production.

    Args:
        candidate_path: Path to candidate adapter (*.candidate directory)
        evaluation: Evaluation results for this adapter
        baseline_eval: Optional baseline for comparison

    Returns:
        Tuple of (promoted: bool, reason: str, promoted_path: Optional[Path])
    """
    if not candidate_path.exists():
        return False, f"Candidate path does not exist: {candidate_path}", None

    # Check cooldown
    can_promote, cooldown_reason = _check_promotion_cooldown()
    if not can_promote:
        return False, cooldown_reason, None

    # Check promotion criteria
    if baseline_eval is not None:
        should_promote, promotion_reason = should_promote_adapter(
            candidate_eval=evaluation,
            baseline_eval=baseline_eval,
            min_ic_improvement=settings.dpo.min_ic_delta,
            min_brier_improvement=0.01,
            min_test_samples=settings.dpo.test_window,
        )
    else:
        # No baseline - check absolute metrics
        should_promote = (
            evaluation.ic >= settings.dpo.min_oos_ic
            and evaluation.brier_score <= settings.dpo.max_brier_score
            and evaluation.ic_pvalue < 0.05
        )
        if should_promote:
            promotion_reason = (
                f"Meets absolute criteria: IC={evaluation.ic:.4f}, "
                f"Brier={evaluation.brier_score:.4f}"
            )
        else:
            promotion_reason = (
                f"Below absolute criteria: IC={evaluation.ic:.4f} "
                f"(need {settings.dpo.min_oos_ic}), "
                f"Brier={evaluation.brier_score:.4f} "
                f"(need <{settings.dpo.max_brier_score})"
            )

    if not should_promote:
        return False, promotion_reason, None

    # Promote: rename from .candidate to .promoted
    adapter_name = candidate_path.name.replace(".candidate", "")
    promoted_path = candidate_path.parent / f"{adapter_name}.promoted"

    candidate_path.rename(promoted_path)
    _log_promotion(promoted_path, evaluation)

    logger.info(
        "Adapter promoted",
        adapter_path=str(promoted_path),
        ic=evaluation.ic,
        brier=evaluation.brier_score,
    )

    return True, promotion_reason, promoted_path


def check_should_retrain(
    current_pair_count: int,
    last_training_pair_count: int,
) -> tuple[bool, str]:
    """
    Check if retraining should be triggered based on new data.

    Args:
        current_pair_count: Current total preference pairs
        last_training_pair_count: Pair count at last training

    Returns:
        Tuple of (should_retrain: bool, reason: str)
    """
    new_pairs = current_pair_count - last_training_pair_count

    # Not enough total data
    min_total = settings.dpo.train_window + settings.dpo.test_window
    if current_pair_count < min_total:
        return False, f"Insufficient total pairs: {current_pair_count} < {min_total}"

    # Not enough new data since last training
    if new_pairs < settings.dpo.retrain_threshold:
        return (
            False,
            f"Insufficient new pairs: {new_pairs} < {settings.dpo.retrain_threshold}",
        )

    return True, f"Retrain triggered: {new_pairs} new pairs since last training"
