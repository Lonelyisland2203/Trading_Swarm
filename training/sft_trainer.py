"""
SFT Training Pipeline for Qwen3-8B fine-tuning on reasoning traces.

CRITICAL: This module runs in Process B (training), which is mutually exclusive
with Process A (inference). Never run both simultaneously.

Architecture Decisions:
1. Uses direct transformers + PEFT + bitsandbytes stack (not Unsloth)
   - Better debuggability, mature error handling, no extra dependencies
2. Downloads from HuggingFace (Ollama cache is GGUF, incompatible)
3. LoRA config matches DPO trainer (r=32, alpha=64, 7 target modules)
4. Pre-flight checks: data -> temporal -> VRAM -> lock -> load
5. Training: lr=2e-5, batch_size=1, grad_accum=16, epochs=3
6. 10% validation split with early stopping (patience=2)

Usage:
    python -m training.sft_trainer --data data/sft_training_data.jsonl
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    EarlyStoppingCallback,
)

from config.settings import settings
from training.process_lock import acquire_training_lock, check_can_train
from training.vram_check import check_vram_availability


# VRAM budget constraints (GB)
MIN_VRAM_GB = 9.0
MAX_VRAM_GB = 11.0


class CudaCacheFlushCallback(TrainerCallback):
    """
    Flush the CUDA memory allocator cache after every optimizer step.

    Without this, near-maxed VRAM (16 GB on RTX 5070 Ti) causes progressive
    slowdown: fragmented free blocks force increasingly expensive coalescing
    on each gradient accumulation sub-step.
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


class VRAMLoggingCallback(TrainerCallback):
    """Log VRAM usage during training."""

    def __init__(self, log_interval_steps: int = 10):
        self.log_interval_steps = log_interval_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step % self.log_interval_steps == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(
                    "VRAM usage",
                    step=state.global_step,
                    allocated_gb=f"{allocated:.2f}",
                    reserved_gb=f"{reserved:.2f}",
                )
        return control


@dataclass
class SFTTrainingConfig:
    """Configuration for SFT training."""

    # Model
    base_model_id: str = field(default_factory=lambda: settings.dpo.base_model_id)
    max_length: int = 1400  # Market snapshot (~1000) + reasoning trace (~300)

    # LoRA (matches DPO config exactly)
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP
        ]
    )

    # Training (SFT-specific, higher LR than DPO)
    learning_rate: float = 2e-5  # Higher than DPO's 5e-6 since this is supervised
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3

    # Validation
    validation_ratio: float = 0.1  # 10% held out

    # Early stopping
    early_stopping_patience: int = 2

    # Adapter output
    adapter_dir: Path = field(default_factory=lambda: Path("models/adapters/sft_base"))

    # Evaluation
    run_eval_after_training: bool = True


@dataclass
class SFTTrainingResult:
    """Result of an SFT training run."""

    success: bool
    adapter_path: Optional[Path]
    training_loss: Optional[float]
    validation_loss: Optional[float]
    epochs_completed: int
    early_stopped: bool
    training_time_seconds: float
    num_examples: int
    error: Optional[str] = None
    evaluation: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "epochs_completed": self.epochs_completed,
            "early_stopped": self.early_stopped,
            "training_time_seconds": self.training_time_seconds,
            "num_examples": self.num_examples,
            "error": self.error,
        }


class EarlyStoppingTracker:
    """Track validation loss for early stopping."""

    def __init__(self, patience: int = 2):
        self.patience = patience
        self.best_loss = float("inf")
        self.consecutive_increases = 0

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.consecutive_increases = 0
            return False
        else:
            self.consecutive_increases += 1
            if self.consecutive_increases >= self.patience:
                return True
            return False


def create_lora_config() -> LoraConfig:
    """
    Create LoRA configuration for SFT training.

    CRITICAL: Must match DPO trainer config exactly for adapter compatibility.
    """
    config = SFTTrainingConfig()
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_sft_data(data_path: Path) -> List[Dict[str, Any]]:
    """
    Load SFT training data from JSONL file.

    Args:
        data_path: Path to JSONL file with SFT examples

    Returns:
        List of example dictionaries

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not data_path.exists():
        raise FileNotFoundError(f"SFT data file not found: {data_path}")

    examples = []
    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                if "market_snapshot" in example and "reasoning_trace" in example:
                    examples.append(example)
                else:
                    logger.warning(f"Line {line_num}: Missing required fields, skipping")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON ({e}), skipping")

    logger.info(f"Loaded {len(examples)} SFT examples from {data_path}")
    return examples


def format_for_sft(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format examples as instruction-following pairs.

    Input: market snapshot with technical indicators
    Output: structured reasoning trace (THESIS→EVIDENCE→RISK→DECISION)

    Args:
        examples: Raw examples with market_snapshot and reasoning_trace

    Returns:
        List of formatted examples with 'input' and 'output' keys
    """
    formatted = []
    for ex in examples:
        # Build instruction prompt
        instruction = (
            "Analyze the following market snapshot and provide a structured "
            "trading analysis with your thesis, supporting evidence, risks, "
            "and final decision.\n\n"
        )

        formatted.append(
            {
                "input": instruction + ex["market_snapshot"],
                "output": ex["reasoning_trace"],
            }
        )

    return formatted


def create_train_val_split(
    examples: List[Dict[str, Any]],
    val_ratio: float = 0.1,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split examples into training and validation sets.

    Args:
        examples: All examples
        val_ratio: Fraction for validation (default 0.1 = 10%)

    Returns:
        Tuple of (train_examples, val_examples)
    """
    import random

    # Shuffle deterministically
    shuffled = examples.copy()
    random.seed(42)
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_examples = shuffled[:val_size]
    train_examples = shuffled[val_size:]

    logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} validation")

    return train_examples, val_examples


def generate_adapter_name() -> str:
    """Generate unique adapter name with timestamp."""
    timestamp_ms = int(time.time() * 1000)
    return f"adapter-SFT-{timestamp_ms}"


def run_preflight_checks() -> tuple[bool, str]:
    """
    Run all pre-flight checks before training.

    Order (fail-fast, cheap to expensive):
    1. Check training lock availability
    2. Check VRAM availability
    3. Enforce OLLAMA_KEEP_ALIVE=0

    Returns:
        Tuple of (can_train: bool, reason: str)
    """
    # Check 1: Training lock
    can_train, lock_reason = check_can_train()
    if not can_train:
        return False, f"Lock unavailable: {lock_reason}"

    # Check 2: VRAM availability
    vram_status = check_vram_availability(min_free_gb=MIN_VRAM_GB)
    if not vram_status.can_train:
        return False, f"VRAM insufficient: {vram_status.reason}"

    # Check 3: Enforce OLLAMA_KEEP_ALIVE=0
    os.environ["OLLAMA_KEEP_ALIVE"] = "0"
    logger.info("OLLAMA_KEEP_ALIVE=0 enforced")

    # Check for STOP file
    stop_file = Path("execution/state/STOP")
    if stop_file.exists():
        return False, "STOP file exists - refusing to train"

    logger.info(
        "Pre-flight checks passed",
        vram_free_gb=f"{vram_status.free_mb / 1024:.1f}",
    )

    return True, "Ready to train"


def _create_quantization_config() -> BitsAndBytesConfig:
    """
    Create 4-bit quantization config for QLoRA.

    Uses NF4 quantization with double quantization for maximum VRAM efficiency.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_model_and_tokenizer(
    config: SFTTrainingConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load quantized model and tokenizer from HuggingFace.

    Downloads on first run (~16GB), uses cache subsequently.
    """
    logger.info(f"Loading model: {config.base_model_id}")

    bnb_config = _create_quantization_config()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        trust_remote_code=True,
        padding_side="right",  # SFT uses right padding
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    logger.info(f"Model loaded: {model.dtype}, device={model.device}")

    return model, tokenizer


def _prepare_dataset(
    examples: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """
    Prepare HuggingFace Dataset for SFT training.

    Tokenizes input-output pairs and creates labels for causal LM training.
    """

    def tokenize_function(example):
        # Combine input and output with separator
        full_text = example["input"] + "\n\n" + example["output"] + tokenizer.eos_token

        # Tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels = input_ids (model predicts next token)
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Mask the input portion (don't compute loss on prompt)
        # Find where output starts
        input_text = example["input"] + "\n\n"
        input_tokens = tokenizer(input_text, truncation=True, max_length=max_length)
        input_len = len(input_tokens["input_ids"])

        # Set label to -100 for input portion (ignored in loss)
        tokenized["labels"][:input_len] = [-100] * input_len

        return tokenized

    # Create dataset
    dataset = Dataset.from_list(examples)
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["input", "output"])

    return tokenized_dataset


def _create_training_args(
    config: SFTTrainingConfig,
    output_dir: Path,
    num_train_examples: int,
) -> TrainingArguments:
    """Create training arguments for SFT."""
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        bf16=True,
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        # Regularization
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Reproducibility
        seed=42,
    )


def _save_adapter_with_metadata(
    model: AutoModelForCausalLM,
    output_dir: Path,
    config: SFTTrainingConfig,
    training_time_seconds: float,
    num_examples: int,
    epochs_completed: int,
) -> Path:
    """Save LoRA adapter and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights
    model.save_pretrained(str(output_dir))

    # Save metadata
    metadata = {
        "timestamp_ms": int(time.time() * 1000),
        "base_model_id": config.base_model_id,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "learning_rate": config.learning_rate,
        "num_examples": num_examples,
        "epochs_completed": epochs_completed,
        "training_time_seconds": training_time_seconds,
        "training_type": "SFT",
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Adapter saved to {output_dir}")

    return output_dir


def train_sft(
    data_path: Path,
    output_dir: Optional[Path] = None,
    run_eval: bool = True,
) -> SFTTrainingResult:
    """
    Execute full SFT training pipeline.

    Pipeline:
    1. Pre-flight checks (lock, VRAM, OLLAMA_KEEP_ALIVE)
    2. Load training data from JSONL
    3. Split into train/validation (90/10)
    4. Acquire training lock
    5. Load model and tokenizer
    6. Prepare datasets
    7. Train with early stopping
    8. Save adapter
    9. Optionally evaluate

    Args:
        data_path: Path to SFT training data JSONL
        output_dir: Directory for saving adapter (default: models/adapters/sft_base)
        run_eval: Whether to run evaluation after training

    Returns:
        SFTTrainingResult with metrics and paths
    """
    start_time = time.time()
    config = SFTTrainingConfig()

    if output_dir is None:
        output_dir = config.adapter_dir

    logger.info(f"Starting SFT training pipeline with data from {data_path}")

    # Pre-flight checks
    can_train, reason = run_preflight_checks()
    if not can_train:
        return SFTTrainingResult(
            success=False,
            adapter_path=None,
            training_loss=None,
            validation_loss=None,
            epochs_completed=0,
            early_stopped=False,
            training_time_seconds=time.time() - start_time,
            num_examples=0,
            error=f"Preflight failed: {reason}",
        )

    # Load and prepare data
    try:
        examples = load_sft_data(data_path)
        formatted = format_for_sft(examples)
        train_examples, val_examples = create_train_val_split(
            formatted, val_ratio=config.validation_ratio
        )
    except Exception as e:
        return SFTTrainingResult(
            success=False,
            adapter_path=None,
            training_loss=None,
            validation_loss=None,
            epochs_completed=0,
            early_stopped=False,
            training_time_seconds=time.time() - start_time,
            num_examples=0,
            error=f"Data loading failed: {e}",
        )

    # Acquire lock and train
    try:
        with acquire_training_lock():
            logger.info("Training lock acquired")

            # Generate adapter name
            adapter_name = generate_adapter_name()
            adapter_output_dir = output_dir / adapter_name

            # Load model and tokenizer
            model, tokenizer = _load_model_and_tokenizer(config)

            # Apply LoRA
            lora_config = create_lora_config()
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Prepare datasets
            train_dataset = _prepare_dataset(train_examples, tokenizer, config.max_length)
            val_dataset = _prepare_dataset(val_examples, tokenizer, config.max_length)

            # Create training arguments
            training_args = _create_training_args(config, adapter_output_dir, len(train_examples))

            # Create trainer with callbacks
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[
                    CudaCacheFlushCallback(),
                    VRAMLoggingCallback(log_interval_steps=10),
                    EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
                ],
            )

            logger.info(
                "Starting training",
                train_examples=len(train_examples),
                val_examples=len(val_examples),
                epochs=config.num_epochs,
            )

            # Train
            train_result = trainer.train()

            # Get final losses
            training_loss = train_result.training_loss
            eval_result = trainer.evaluate()
            validation_loss = eval_result.get("eval_loss")

            training_time = time.time() - start_time

            # Check if early stopped
            steps_per_epoch = max(
                1, len(train_dataset) // config.batch_size // config.gradient_accumulation_steps
            )
            epochs_completed = (
                int(train_result.global_step / steps_per_epoch)
                if steps_per_epoch > 0
                else config.num_epochs
            )
            early_stopped = epochs_completed < config.num_epochs

            logger.info(
                "Training complete",
                training_loss=training_loss,
                validation_loss=validation_loss,
                epochs_completed=epochs_completed,
                early_stopped=early_stopped,
            )

            # Save adapter
            adapter_path = _save_adapter_with_metadata(
                model=model,
                output_dir=adapter_output_dir,
                config=config,
                training_time_seconds=training_time,
                num_examples=len(examples),
                epochs_completed=epochs_completed,
            )

            # Clean up GPU memory
            del model
            del trainer
            torch.cuda.empty_cache()

            # Run evaluation if requested
            evaluation = None
            if run_eval and config.run_eval_after_training:
                try:
                    from training.dpo_eval import evaluate_adapter

                    evaluation = evaluate_adapter(adapter_path)
                    logger.info(
                        "Evaluation complete",
                        ic=evaluation.ic,
                        brier_score=evaluation.brier_score,
                    )
                except ImportError:
                    logger.warning("Could not import evaluate_adapter, skipping evaluation")
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")

            return SFTTrainingResult(
                success=True,
                adapter_path=adapter_path,
                training_loss=training_loss,
                validation_loss=validation_loss,
                epochs_completed=epochs_completed,
                early_stopped=early_stopped,
                training_time_seconds=training_time,
                num_examples=len(examples),
                evaluation=evaluation,
            )

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return SFTTrainingResult(
            success=False,
            adapter_path=None,
            training_loss=None,
            validation_loss=None,
            epochs_completed=0,
            early_stopped=False,
            training_time_seconds=time.time() - start_time,
            num_examples=len(examples) if examples else 0,
            error=str(e),
        )


def main() -> None:
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Train SFT model on reasoning traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sft_training_data.jsonl"),
        help="Path to SFT training data JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saving adapter (default: models/adapters/sft_base)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Run training
    result = train_sft(
        data_path=args.data,
        output_dir=args.output_dir,
        run_eval=not args.no_eval,
    )

    # Print result summary
    if result.success:
        logger.info(
            "Training completed successfully",
            adapter_path=str(result.adapter_path),
            training_loss=result.training_loss,
            validation_loss=result.validation_loss,
            epochs=result.epochs_completed,
        )
    else:
        logger.error(f"Training failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
