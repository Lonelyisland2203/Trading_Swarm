"""Tests for process-level locking."""

import pytest
from pathlib import Path

from training.process_lock import (
    ProcessLockError,
    acquire_training_lock,
    acquire_inference_lock,
    check_can_train,
    check_can_infer,
    LOCK_DIR,
)


# Module-level helper functions for multiprocessing (must be picklable)
def _hold_training_lock(ready_event=None):
    """Subprocess: hold training lock for 2 seconds."""
    import time
    with acquire_training_lock():
        if ready_event is not None:
            ready_event.set()  # Signal that lock is acquired
        time.sleep(2)


def _hold_inference_lock(ready_event=None):
    """Subprocess: hold inference lock for 2 seconds."""
    import time
    with acquire_inference_lock():
        if ready_event is not None:
            ready_event.set()  # Signal that lock is acquired
        time.sleep(2)


@pytest.fixture(autouse=True)
def cleanup_locks():
    """Clean up lock files before and after each test."""
    # Before test
    if LOCK_DIR.exists():
        for lock_file in LOCK_DIR.glob("*.lock"):
            try:
                lock_file.unlink()
            except:
                pass

    yield

    # After test
    if LOCK_DIR.exists():
        for lock_file in LOCK_DIR.glob("*.lock"):
            try:
                lock_file.unlink()
            except:
                pass


class TestTrainingLock:
    """Test training lock acquisition."""

    def test_acquire_training_lock_succeeds(self):
        """Test acquiring training lock when no other process is running."""
        with acquire_training_lock():
            # Lock should be held
            assert (LOCK_DIR / "training.lock").exists()

        # Lock should be released after context exit
        # (file still exists, but lock is released)

    def test_cannot_acquire_training_lock_if_inference_running(self):
        """Test training lock fails if inference is running."""
        # Simulate inference running
        with acquire_inference_lock():
            # Try to acquire training lock
            with pytest.raises(ProcessLockError, match="Inference process.*is currently running"):
                with acquire_training_lock():
                    pass

    def test_cannot_acquire_two_training_locks(self):
        """Test only one training process can run at a time."""
        import multiprocessing

        # Use Event to signal when subprocess has acquired lock
        ready_event = multiprocessing.Event()

        # Start subprocess holding training lock
        proc = multiprocessing.Process(target=_hold_training_lock, args=(ready_event,))
        proc.start()

        # Wait for subprocess to acquire lock (with timeout)
        assert ready_event.wait(timeout=5), "Subprocess failed to acquire lock"

        # Try to acquire training lock in main process
        with pytest.raises(ProcessLockError, match="Another training process.*is already running"):
            with acquire_training_lock():
                pass

        # Wait for subprocess to finish
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


class TestInferenceLock:
    """Test inference lock acquisition."""

    def test_acquire_inference_lock_succeeds(self):
        """Test acquiring inference lock when no training is running."""
        with acquire_inference_lock():
            # Lock should be held
            assert (LOCK_DIR / "inference.lock").exists()

    def test_cannot_acquire_inference_lock_if_training_running(self):
        """Test inference lock fails if training is running."""
        # Simulate training running
        with acquire_training_lock():
            # Try to acquire inference lock
            with pytest.raises(ProcessLockError, match="Training process.*is currently running"):
                with acquire_inference_lock():
                    pass

    def test_can_acquire_multiple_inference_locks(self):
        """Test multiple inference processes can run concurrently (shared lock)."""
        import multiprocessing

        # Use Event to signal when subprocess has acquired lock
        ready_event = multiprocessing.Event()

        # Start subprocess holding inference lock
        proc = multiprocessing.Process(target=_hold_inference_lock, args=(ready_event,))
        proc.start()

        # Wait for subprocess to acquire lock (with timeout)
        assert ready_event.wait(timeout=5), "Subprocess failed to acquire lock"

        # Acquire inference lock in main process (should succeed - shared lock)
        with acquire_inference_lock():
            pass  # Should not raise

        # Wait for subprocess to finish
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)


class TestLockChecks:
    """Test non-blocking lock checks."""

    def test_check_can_train_when_idle(self):
        """Test check_can_train returns True when idle."""
        can_train, reason = check_can_train()
        assert can_train
        assert reason == "Ready to train"

    def test_check_can_train_when_inference_running(self):
        """Test check_can_train returns False when inference is running."""
        with acquire_inference_lock():
            can_train, reason = check_can_train()
            assert not can_train
            assert "Inference process is running" in reason

    def test_check_can_train_when_training_running(self):
        """Test check_can_train returns False when another training is running."""
        import multiprocessing

        # Use Event to signal when subprocess has acquired lock
        ready_event = multiprocessing.Event()

        proc = multiprocessing.Process(target=_hold_training_lock, args=(ready_event,))
        proc.start()

        # Wait for subprocess to acquire lock (with timeout)
        assert ready_event.wait(timeout=5), "Subprocess failed to acquire lock"

        can_train, reason = check_can_train()
        assert not can_train
        assert "training process is running" in reason.lower()

        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)

    def test_check_can_infer_when_idle(self):
        """Test check_can_infer returns True when idle."""
        can_infer, reason = check_can_infer()
        assert can_infer
        assert reason == "Ready to infer"

    def test_check_can_infer_when_training_running(self):
        """Test check_can_infer returns False when training is running."""
        with acquire_training_lock():
            can_infer, reason = check_can_infer()
            assert not can_infer
            assert "Training process is running" in reason
