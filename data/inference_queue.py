"""
Sequential inference job queue with resume capability.

Processes inference jobs one at a time through VRAM-constrained GPU,
with incremental JSONL saving and resume support.
"""

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger


class EnumJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Enum values by converting to their .value."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


from data.prompt_builder import TaskType
from data.regime_filter import MarketRegime
from swarm.orchestrator import run_multi_persona_workflow
from swarm.training_capture import TrainingExample


@dataclass(slots=True)
class InferenceJob:
    """
    A single inference job for one market context.

    Groups 5 persona generations under shared context_id for DPO pairing.
    """

    job_id: str  # Unique ID for tracking
    context_id: str  # Groups personas (for DPO pairing)
    symbol: str
    timeframe: str
    timestamp_ms: int
    ohlcv_df: pd.DataFrame
    market_regime: MarketRegime
    task_type: TaskType
    task_prompt: str


@dataclass(slots=True)
class ProcessingStats:
    """Statistics from processing job queue."""

    total_jobs: int
    completed: int
    failed: int
    skipped_resume: int
    total_examples: int
    elapsed_seconds: float


class InferenceQueue:
    """
    Sequential job processor with resume capability.

    Features:
    - FIFO processing through single GPU
    - Incremental JSONL saving (after each context)
    - Progress tracking with ETA
    - Error handling with retry logic
    - Skip completed context_ids on resume
    """

    def __init__(
        self,
        output_file: Path,
        resume: bool = False,
        max_retries: int = 2,
        retry_delay_seconds: int = 10,
    ):
        """
        Initialize inference queue.

        Args:
            output_file: Path to JSONL output file
            resume: If True, load completed contexts from output_file
            max_retries: Maximum retries per failed job
            retry_delay_seconds: Delay between retries
        """
        self.output_file = output_file
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load resume state or truncate stale output from prior runs
        if resume:
            self.completed_contexts = self._load_resume_state()
        else:
            self.completed_contexts = set()
            # Truncate existing file so append-mode writes start fresh
            if self.output_file.exists():
                self.output_file.unlink()
                logger.info("Cleared stale output file", file=str(self.output_file))

        if resume and self.completed_contexts:
            logger.info(
                "Resume mode enabled",
                completed_contexts=len(self.completed_contexts),
            )

    def _load_resume_state(self) -> set[str]:
        """
        Load completed context_ids from existing JSONL.

        Only considers contexts with all 5 personas complete (valid for DPO).

        Returns:
            Set of completed context_ids
        """
        if not self.output_file.exists():
            return set()

        context_counts = Counter()

        try:
            with open(self.output_file) as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        context_id = ex.get("context_id")
                        if context_id:
                            context_counts[context_id] += 1
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSONL line")
                        continue

            # Only consider contexts with all 5 personas complete
            completed = {ctx for ctx, count in context_counts.items() if count == 5}

            logger.info(
                "Resume state loaded",
                total_contexts=len(context_counts),
                complete_contexts=len(completed),
                incomplete_contexts=len(context_counts) - len(completed),
            )

            return completed

        except Exception as e:
            logger.error("Failed to load resume state", error=str(e))
            return set()

    async def _process_single_job(
        self,
        job: InferenceJob,
        retry_count: int = 0,
    ) -> tuple[bool, list[TrainingExample]]:
        """
        Process single inference job.

        Args:
            job: Inference job to process
            retry_count: Current retry attempt (0 = first try)

        Returns:
            Tuple of (success: bool, examples: list)
        """
        try:
            logger.info(
                "Processing job",
                symbol=job.symbol,
                timeframe=job.timeframe,
                task_type=job.task_type.value,
                stride_idx=job.context_id.split("_")[-1] if "_" in job.context_id else "?",
                retry=retry_count,
            )

            # Run multi-persona workflow (5 personas)
            summary, examples = await run_multi_persona_workflow(
                symbol=job.symbol,
                timeframe=job.timeframe,
                ohlcv_df=job.ohlcv_df,
                market_regime=job.market_regime,
                task_prompt=job.task_prompt,
                task_type=job.task_type,
            )

            # Override context_id to match job (for proper grouping)
            for ex in examples:
                ex.context_id = job.context_id

            # Check workflow success
            if summary["workflow_status"] in ("success", "partial_success"):
                logger.info(
                    "Job completed",
                    job_id=job.job_id,
                    signals_generated=summary["signals_generated"],
                    signals_accepted=summary["signals_accepted"],
                )
                return True, examples
            else:
                logger.warning(
                    "Job failed",
                    job_id=job.job_id,
                    status=summary["workflow_status"],
                    errors=summary.get("errors", []),
                )
                return False, []

        except Exception as e:
            logger.error(
                "Job exception",
                job_id=job.job_id,
                error=str(e),
                retry=retry_count,
            )
            return False, []

    def _save_examples(self, examples: list[TrainingExample]) -> None:
        """
        Save examples to JSONL file (append mode).

        Args:
            examples: List of training examples to save
        """
        try:
            with open(self.output_file, "a") as f:
                for ex in examples:
                    # Convert to dict and write as JSONL
                    # EnumJSONEncoder handles Enum values (TaskType, TradingPersona)
                    ex_dict = ex.to_dict()
                    f.write(json.dumps(ex_dict, cls=EnumJSONEncoder) + "\n")

            logger.debug(
                "Examples saved",
                count=len(examples),
                file=str(self.output_file),
            )

        except Exception as e:
            logger.error(
                "Failed to save examples",
                count=len(examples),
                error=str(e),
            )

    async def process_all(
        self,
        jobs: list[InferenceJob],
        progress_callback: Optional[callable] = None,
    ) -> ProcessingStats:
        """
        Process all jobs sequentially through model.

        Workflow per job:
        1. Skip if context_id in completed_contexts (resume)
        2. Call run_multi_persona_workflow()
        3. Save 5 examples to JSONL (append mode)
        4. Retry on failure up to max_retries
        5. Invoke progress_callback if provided

        Args:
            jobs: List of inference jobs
            progress_callback: Optional callback(context_id, duration_sec, success)

        Returns:
            Processing statistics
        """
        start_time = time.time()

        stats = ProcessingStats(
            total_jobs=len(jobs),
            completed=0,
            failed=0,
            skipped_resume=0,
            total_examples=0,
            elapsed_seconds=0.0,
        )

        for idx, job in enumerate(jobs):
            job_start = time.time()

            # Skip if already completed (resume)
            if job.context_id in self.completed_contexts:
                stats.skipped_resume += 1
                logger.debug(
                    "Skipping completed context",
                    context_id=job.context_id,
                    progress=f"{idx + 1}/{len(jobs)}",
                )
                continue

            # Process with retry logic
            success = False
            examples = []

            for retry in range(self.max_retries + 1):
                success, examples = await self._process_single_job(job, retry)

                if success:
                    break

                # Retry delay
                if retry < self.max_retries:
                    logger.info(
                        "Retrying job",
                        job_id=job.job_id,
                        retry=retry + 1,
                        max_retries=self.max_retries,
                        delay_sec=self.retry_delay_seconds,
                    )
                    await asyncio.sleep(self.retry_delay_seconds)

            # Update stats
            if success:
                stats.completed += 1
                stats.total_examples += len(examples)

                # Save examples incrementally
                self._save_examples(examples)

                # Mark as completed
                self.completed_contexts.add(job.context_id)
            else:
                stats.failed += 1
                logger.warning(
                    "Job failed after retries",
                    job_id=job.job_id,
                    max_retries=self.max_retries,
                )

            # Progress callback
            job_duration = time.time() - job_start
            if progress_callback:
                progress_callback(job.context_id, job_duration, success)

        # Final stats
        stats.elapsed_seconds = time.time() - start_time

        logger.info(
            "Queue processing complete",
            total=stats.total_jobs,
            completed=stats.completed,
            failed=stats.failed,
            skipped=stats.skipped_resume,
            examples=stats.total_examples,
            elapsed_min=stats.elapsed_seconds / 60,
        )

        return stats
