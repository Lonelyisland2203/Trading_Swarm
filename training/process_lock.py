"""
Process-level locking to prevent concurrent inference and training.

CRITICAL: Process A (inference) and Process B (training) must NEVER run simultaneously.
This module provides file-based locking with exclusive/shared semantics to enforce isolation.
"""

import fcntl
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from loguru import logger

# Lock directory
LOCK_DIR = Path("/Users/javierlee/Trading Swarm/.locks")


class ProcessLockError(Exception):
    """Raised when process lock cannot be acquired."""
    pass


@contextmanager
def acquire_training_lock() -> Generator[None, None, None]:
    """
    Acquire exclusive training lock (Process B).

    Fails if:
    - Inference process (Process A) is currently running
    - Another training process is already running

    Raises:
        ProcessLockError: If lock cannot be acquired

    Example:
        >>> with acquire_training_lock():
        ...     # Training code here
        ...     train_model()
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    inference_lock_path = LOCK_DIR / "inference.lock"
    training_lock_path = LOCK_DIR / "training.lock"

    # Check inference not running
    if inference_lock_path.exists():
        try:
            with open(inference_lock_path, "r") as f:
                # Try to acquire non-blocking exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # If we got the lock, inference is not running, release it
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            # Lock is held - inference is running
            raise ProcessLockError(
                "Inference process (Process A) is currently running. "
                "Cannot start training. Wait for inference to complete or stop it manually."
            )
        except Exception as e:
            logger.warning(f"Error checking inference lock: {e}")

    # Acquire training lock
    training_lock_path.touch()
    training_lock_file = open(training_lock_path, "r+")

    try:
        # Try to acquire exclusive lock
        fcntl.flock(training_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.info("Training lock acquired", lock_path=str(training_lock_path))

        yield

    except BlockingIOError:
        training_lock_file.close()
        raise ProcessLockError(
            "Another training process (Process B) is already running. "
            "Only one training process allowed at a time."
        )
    finally:
        # Release lock and close file
        try:
            fcntl.flock(training_lock_file.fileno(), fcntl.LOCK_UN)
            training_lock_file.close()
            logger.info("Training lock released")
        except Exception as e:
            logger.error(f"Error releasing training lock: {e}")


@contextmanager
def acquire_inference_lock() -> Generator[None, None, None]:
    """
    Acquire shared inference lock (Process A).

    Fails if:
    - Training process (Process B) is currently running

    Multiple inference processes can run concurrently (shared lock),
    but inference and training are mutually exclusive.

    Raises:
        ProcessLockError: If lock cannot be acquired

    Example:
        >>> with acquire_inference_lock():
        ...     # Inference code here
        ...     generate_signal()
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    training_lock_path = LOCK_DIR / "training.lock"
    inference_lock_path = LOCK_DIR / "inference.lock"

    # Check training not running
    if training_lock_path.exists():
        try:
            with open(training_lock_path, "r") as f:
                # Try to acquire non-blocking exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # If we got the lock, training is not running, release it
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            # Lock is held - training is running
            raise ProcessLockError(
                "Training process (Process B) is currently running. "
                "Cannot start inference. Wait for training to complete."
            )
        except Exception as e:
            logger.warning(f"Error checking training lock: {e}")

    # Acquire inference lock (shared)
    inference_lock_path.touch()
    inference_lock_file = open(inference_lock_path, "r+")

    try:
        # Acquire shared lock (multiple inference processes allowed)
        fcntl.flock(inference_lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        logger.debug("Inference lock acquired (shared)", lock_path=str(inference_lock_path))

        yield

    except BlockingIOError:
        inference_lock_file.close()
        raise ProcessLockError(
            "Cannot acquire inference lock. This should not happen (shared locks)."
        )
    finally:
        # Release lock and close file
        try:
            fcntl.flock(inference_lock_file.fileno(), fcntl.LOCK_UN)
            inference_lock_file.close()
            logger.debug("Inference lock released")
        except Exception as e:
            logger.error(f"Error releasing inference lock: {e}")


def check_can_train() -> tuple[bool, str]:
    """
    Check if training can start (non-blocking check).

    Returns:
        Tuple of (can_train: bool, reason: str)

    Example:
        >>> can_train, reason = check_can_train()
        >>> if not can_train:
        ...     print(f"Cannot train: {reason}")
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    inference_lock_path = LOCK_DIR / "inference.lock"
    training_lock_path = LOCK_DIR / "training.lock"

    # Check inference not running
    if inference_lock_path.exists():
        try:
            with open(inference_lock_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            return False, "Inference process is running"
        except Exception as e:
            return False, f"Error checking inference lock: {e}"

    # Check another training not running
    if training_lock_path.exists():
        try:
            with open(training_lock_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            return False, "Another training process is running"
        except Exception as e:
            return False, f"Error checking training lock: {e}"

    return True, "Ready to train"


def check_can_infer() -> tuple[bool, str]:
    """
    Check if inference can start (non-blocking check).

    Returns:
        Tuple of (can_infer: bool, reason: str)

    Example:
        >>> can_infer, reason = check_can_infer()
        >>> if not can_infer:
        ...     print(f"Cannot infer: {reason}")
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    training_lock_path = LOCK_DIR / "training.lock"

    # Check training not running
    if training_lock_path.exists():
        try:
            with open(training_lock_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            return False, "Training process is running"
        except Exception as e:
            return False, f"Error checking training lock: {e}"

    return True, "Ready to infer"
