"""
Unified preflight checks for the signal loop.

Implements the required check order: STOP file -> Process lock -> VRAM.
Reuses existing modules: training.process_lock, training.vram_check.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from training.process_lock import check_can_infer
from training.vram_check import check_vram_availability, VRAMStatus


STOP_FILE_PATH = Path("execution/state/STOP")

# Inference requires less VRAM than training (no gradients, no optimizer state)
# Training needs 9GB, inference needs ~6GB for single model
DEFAULT_MIN_VRAM_GB = 6.0


@dataclass(frozen=True)
class PreflightResult:
    """Result of preflight checks."""

    passed: bool
    reason: str
    vram_status: VRAMStatus | None = None


def check_stop_file() -> bool:
    """
    Check if STOP file exists (kill switch).

    The STOP file at execution/state/STOP halts all trading immediately.
    Can be created manually or by StateManager.activate_kill_switch().

    Returns:
        True if STOP file exists (trading should halt)
    """
    exists = STOP_FILE_PATH.exists()
    if exists:
        logger.warning("Kill switch active: STOP file exists", path=str(STOP_FILE_PATH))
    return exists


def run_preflight_checks(
    min_vram_gb: float = DEFAULT_MIN_VRAM_GB,
    skip_vram_check: bool = False,
) -> PreflightResult:
    """
    Run all preflight checks in required order.

    Order: STOP file -> Process lock -> VRAM

    This matches the Data -> Temporal -> VRAM -> Lock -> Load order
    from training preflight, adapted for inference:
    - STOP file (emergency halt)
    - Process lock (no training running)
    - VRAM (sufficient memory)

    Args:
        min_vram_gb: Minimum VRAM required (default: 6.0GB for inference)
        skip_vram_check: Skip VRAM check (for testing)

    Returns:
        PreflightResult with pass/fail status and reason
    """
    # Stage 1: Kill switch check (highest priority)
    if check_stop_file():
        return PreflightResult(
            passed=False,
            reason="STOP file exists - trading halted",
        )

    # Stage 2: Process lock check (can we infer?)
    # Inference is blocked when training is running
    can_infer, lock_reason = check_can_infer()
    if not can_infer:
        return PreflightResult(
            passed=False,
            reason=f"Process lock blocked: {lock_reason}",
        )

    # Stage 3: VRAM check (optional, can skip for testing)
    if not skip_vram_check:
        vram_status = check_vram_availability(min_free_gb=min_vram_gb)

        # Note: VRAMStatus has can_train field, but for inference we need less
        # We use the same check since can_train just checks min_free_gb
        if not vram_status.can_train:
            return PreflightResult(
                passed=False,
                reason=f"Insufficient VRAM: {vram_status.reason}",
                vram_status=vram_status,
            )

        logger.debug(
            "VRAM check passed",
            free_gb=f"{vram_status.free_mb / 1024:.1f}",
            gpu=vram_status.gpu_name,
        )

        return PreflightResult(
            passed=True,
            reason="All preflight checks passed",
            vram_status=vram_status,
        )

    # If VRAM check skipped
    return PreflightResult(
        passed=True,
        reason="All preflight checks passed (VRAM check skipped)",
    )


def enforce_ollama_keep_alive() -> None:
    """
    Enforce OLLAMA_KEEP_ALIVE=0 environment variable.

    Required for VRAM management - models must unload immediately
    to allow switching between generator and critic.
    """
    current = os.environ.get("OLLAMA_KEEP_ALIVE")
    if current != "0":
        os.environ["OLLAMA_KEEP_ALIVE"] = "0"
        logger.info("Set OLLAMA_KEEP_ALIVE=0 for VRAM management")
