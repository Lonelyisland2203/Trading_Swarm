"""
VRAM availability check before training.

CRITICAL: Training requires ~9-11 GB VRAM (4-bit model + LoRA + gradients).
This module provides pre-flight checks to prevent OOM errors.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class VRAMStatus:
    """VRAM availability status."""

    total_mb: int
    used_mb: int
    free_mb: int
    gpu_name: str
    can_train: bool
    reason: str


def check_vram_availability(min_free_gb: float = 9.0) -> VRAMStatus:
    """
    Check if sufficient VRAM is available for training.

    Args:
        min_free_gb: Minimum free VRAM in GB required for training (default: 9.0)

    Returns:
        VRAMStatus with availability information

    Example:
        >>> status = check_vram_availability(min_free_gb=9.0)
        >>> if not status.can_train:
        ...     print(f"Cannot train: {status.reason}")
        ...     print(f"Free VRAM: {status.free_mb / 1024:.1f} GB")
    """
    min_free_mb = int(min_free_gb * 1024)

    try:
        # Try nvidia-smi first (NVIDIA GPUs)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Parse nvidia-smi output
            line = result.stdout.strip().split("\n")[0]  # First GPU
            total, used, free, name = [x.strip() for x in line.split(",")]

            total_mb = int(total)
            used_mb = int(used)
            free_mb = int(free)
            gpu_name = name

            can_train = free_mb >= min_free_mb

            if can_train:
                reason = f"Sufficient VRAM available: {free_mb / 1024:.1f} GB free"
            else:
                reason = (
                    f"Insufficient VRAM: {free_mb / 1024:.1f} GB free, "
                    f"need {min_free_gb:.1f} GB minimum"
                )

            return VRAMStatus(
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
                gpu_name=gpu_name,
                can_train=can_train,
                reason=reason,
            )

    except FileNotFoundError:
        logger.warning("nvidia-smi not found, trying Metal Performance Shaders")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except Exception as e:
        logger.warning(f"Error checking NVIDIA GPU: {e}")

    # Try Metal (Apple Silicon)
    try:
        # On macOS with Apple Silicon, we don't have direct VRAM query tools
        # But we can check if we're on Apple Silicon and make a conservative estimate
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and "Mac" in result.stdout:
            # Apple Silicon - unified memory
            # Get total physical memory
            mem_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if mem_result.returncode == 0:
                total_bytes = int(mem_result.stdout.strip())
                total_mb = total_bytes // (1024 * 1024)

                # Conservative estimate: assume 50% of RAM is available for GPU
                # (actual availability depends on macOS memory pressure)
                estimated_gpu_mb = total_mb // 2
                estimated_used_mb = estimated_gpu_mb // 4  # Very rough estimate
                estimated_free_mb = estimated_gpu_mb - estimated_used_mb

                can_train = estimated_free_mb >= min_free_mb

                if can_train:
                    reason = (
                        f"Apple Silicon detected, estimated {estimated_free_mb / 1024:.1f} GB available "
                        "(unified memory, conservative estimate)"
                    )
                else:
                    reason = (
                        f"Apple Silicon: estimated {estimated_free_mb / 1024:.1f} GB available, "
                        f"need {min_free_gb:.1f} GB minimum (unified memory)"
                    )

                return VRAMStatus(
                    total_mb=estimated_gpu_mb,
                    used_mb=estimated_used_mb,
                    free_mb=estimated_free_mb,
                    gpu_name="Apple Silicon (unified memory)",
                    can_train=can_train,
                    reason=reason,
                )

    except Exception as e:
        logger.warning(f"Error checking Apple Silicon: {e}")

    # Fallback: cannot determine VRAM
    return VRAMStatus(
        total_mb=0,
        used_mb=0,
        free_mb=0,
        gpu_name="Unknown",
        can_train=False,
        reason="Cannot determine VRAM availability (no nvidia-smi or Metal detection)",
    )


def log_vram_status() -> VRAMStatus:
    """
    Check and log VRAM status.

    Returns:
        VRAMStatus with availability information
    """
    status = check_vram_availability()

    logger.info(
        "VRAM status check",
        gpu=status.gpu_name,
        total_gb=f"{status.total_mb / 1024:.1f}",
        used_gb=f"{status.used_mb / 1024:.1f}",
        free_gb=f"{status.free_mb / 1024:.1f}",
        can_train=status.can_train,
    )

    if not status.can_train:
        logger.warning(f"Training not recommended: {status.reason}")
    else:
        logger.info(f"Training feasible: {status.reason}")

    return status
