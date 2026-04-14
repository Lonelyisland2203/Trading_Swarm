"""
Process-level locking to prevent concurrent inference and training.

CRITICAL: Process A (inference) and Process B (training) must NEVER run simultaneously.
This module provides file-based locking with exclusive/shared semantics to enforce isolation.

Cross-platform: uses msvcrt on Windows, fcntl on Unix/macOS.
"""

import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from loguru import logger

# Platform-specific locking
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes
    import msvcrt as _msvcrt

    _k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    # LockFileEx flags
    _LOCKFILE_EXCLUSIVE_LOCK = 0x0002
    _LOCKFILE_FAIL_IMMEDIATELY = 0x0001

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_ulong),
            ("InternalHigh", ctypes.c_ulong),
            ("Offset", ctypes.c_ulong),
            ("OffsetHigh", ctypes.c_ulong),
            ("hEvent", ctypes.c_void_p),
        ]

    def _lock_exclusive(f):
        handle = _msvcrt.get_osfhandle(f.fileno())
        ovlp = _OVERLAPPED()
        if not _k32.LockFileEx(
            handle,
            _LOCKFILE_EXCLUSIVE_LOCK | _LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            ctypes.byref(ovlp),
        ):
            raise OSError("Could not acquire exclusive lock")

    def _lock_shared(f):
        handle = _msvcrt.get_osfhandle(f.fileno())
        ovlp = _OVERLAPPED()
        if not _k32.LockFileEx(
            handle,
            _LOCKFILE_FAIL_IMMEDIATELY,  # No EXCLUSIVE flag → shared
            0,
            1,
            0,
            ctypes.byref(ovlp),
        ):
            raise OSError("Could not acquire shared lock")

    def _unlock(f):
        try:
            handle = _msvcrt.get_osfhandle(f.fileno())
            ovlp = _OVERLAPPED()
            _k32.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(ovlp))
        except Exception:
            pass

    _LockError = OSError
else:
    import fcntl

    def _lock_exclusive(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _lock_shared(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)

    def _unlock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    _LockError = BlockingIOError

# Lock directory — works on both Windows and Unix
LOCK_DIR = Path.home() / "Trading Swarm" / ".locks"


class ProcessLockError(Exception):
    """Raised when process lock cannot be acquired."""

    pass


def _is_locked(lock_path: Path) -> bool:
    """Check if a lock file is currently held by another process."""
    if not lock_path.exists():
        return False
    try:
        with open(lock_path, "r+") as f:
            _lock_exclusive(f)
            _unlock(f)
        return False  # We got it — not locked
    except _LockError:
        return True  # Someone else holds it
    except Exception as e:
        logger.warning(f"Error checking lock {lock_path}: {e}")
        return False


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
        ...     train_model()
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    inference_lock_path = LOCK_DIR / "inference.lock"
    training_lock_path = LOCK_DIR / "training.lock"

    if _is_locked(inference_lock_path):
        raise ProcessLockError(
            "Inference process (Process A) is currently running. "
            "Cannot start training. Wait for inference to complete or stop it manually."
        )

    training_lock_path.touch()
    training_lock_file = open(training_lock_path, "r+")

    try:
        _lock_exclusive(training_lock_file)
        logger.info("Training lock acquired", lock_path=str(training_lock_path))
        yield
    except _LockError:
        training_lock_file.close()
        raise ProcessLockError(
            "Another training process (Process B) is already running. "
            "Only one training process allowed at a time."
        )
    finally:
        try:
            _unlock(training_lock_file)
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
        ...     generate_signal()
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    training_lock_path = LOCK_DIR / "training.lock"
    inference_lock_path = LOCK_DIR / "inference.lock"

    if _is_locked(training_lock_path):
        raise ProcessLockError(
            "Training process (Process B) is currently running. "
            "Cannot start inference. Wait for training to complete."
        )

    inference_lock_path.touch()
    inference_lock_file = open(inference_lock_path, "r+")

    try:
        _lock_shared(inference_lock_file)
        logger.debug("Inference lock acquired", lock_path=str(inference_lock_path))
        yield
    except _LockError:
        inference_lock_file.close()
        raise ProcessLockError(
            "Cannot acquire inference lock. This should not happen (shared locks)."
        )
    finally:
        try:
            _unlock(inference_lock_file)
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

    if _is_locked(LOCK_DIR / "inference.lock"):
        return False, "Inference process is running"

    if _is_locked(LOCK_DIR / "training.lock"):
        return False, "Another training process is running"

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

    if _is_locked(LOCK_DIR / "training.lock"):
        return False, "Training process is running"

    return True, "Ready to infer"
