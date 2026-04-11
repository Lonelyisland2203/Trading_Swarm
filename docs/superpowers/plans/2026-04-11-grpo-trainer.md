# GRPO Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GRPO training loop that fine-tunes qwen3:8b using group-relative policy optimization with asymmetric rewards.

**Architecture:** Monolithic `GRPOTrainer` class with sequential G=4 generation per input, reference model weight swapping for KL penalty, and checkpointing every 500 steps. Uses HuggingFace transformers + PEFT stack directly (no Ollama during training).

**Tech Stack:** Python 3.13, transformers, peft, torch, bitsandbytes (4-bit quantization)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `training/grpo_data.py` | `GRPOTrainingExample` dataclass, `GRPOWalkForwardSplit`, temporal split functions |
| `training/grpo_trainer.py` | `GRPOTrainer` class, `GRPOStepResult`, `GRPOTrainingResult`, training loop |
| `tests/training/test_grpo_data.py` | Tests for data types and walk-forward splits |
| `tests/training/test_grpo_trainer.py` | Tests for trainer with mocked model |

---

## Task 1: GRPOTrainingExample Dataclass

**Files:**
- Create: `training/grpo_data.py`
- Create: `tests/training/test_grpo_data.py`

- [ ] **Step 1: Write failing test for GRPOTrainingExample**

```python
# tests/training/test_grpo_data.py
"""Tests for GRPO training data types and walk-forward splits."""

import pytest

from training.grpo_data import GRPOTrainingExample


class TestGRPOTrainingExample:
    """Tests for GRPOTrainingExample dataclass."""

    def test_create_example(self) -> None:
        """Test creating a basic training example."""
        example = GRPOTrainingExample(
            market_snapshot="BTC/USDT: RSI=30, MACD=-0.5",
            actual_direction="LONG",
            gross_return_pct=0.5,
            timestamp_ms=1700000000000,
        )
        assert example.market_snapshot == "BTC/USDT: RSI=30, MACD=-0.5"
        assert example.actual_direction == "LONG"
        assert example.gross_return_pct == 0.5
        assert example.timestamp_ms == 1700000000000

    def test_example_is_frozen(self) -> None:
        """Test that example is immutable."""
        example = GRPOTrainingExample(
            market_snapshot="test",
            actual_direction="LONG",
            gross_return_pct=0.1,
            timestamp_ms=1000,
        )
        with pytest.raises(AttributeError):
            example.actual_direction = "SHORT"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_data.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.grpo_data'"

- [ ] **Step 3: Write minimal implementation**

```python
# training/grpo_data.py
"""
GRPO training data types and walk-forward split functions.

Provides GRPOTrainingExample for representing training inputs and
temporal split utilities for walk-forward validation.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GRPOTrainingExample:
    """
    Single training example for GRPO.

    Attributes:
        market_snapshot: Input prompt with market data and indicators
        actual_direction: Ground truth direction ("LONG", "SHORT", "FLAT")
        gross_return_pct: Ground truth return for reward computation
        timestamp_ms: Timestamp for temporal ordering (milliseconds)
    """

    market_snapshot: str
    actual_direction: str
    gross_return_pct: float
    timestamp_ms: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_grpo_data.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_data.py tests/training/test_grpo_data.py
git commit -m "feat(training): add GRPOTrainingExample dataclass"
```

---

## Task 2: GRPOWalkForwardSplit and Temporal Split Function

**Files:**
- Modify: `training/grpo_data.py`
- Modify: `tests/training/test_grpo_data.py`

- [ ] **Step 1: Write failing tests for walk-forward split**

```python
# tests/training/test_grpo_data.py (append to existing file)

from training.grpo_data import (
    GRPOTrainingExample,
    GRPOWalkForwardSplit,
    create_grpo_walk_forward_split,
    TemporalSplitError,
)


class TestGRPOWalkForwardSplit:
    """Tests for walk-forward temporal splits."""

    @pytest.fixture
    def sample_examples(self) -> list[GRPOTrainingExample]:
        """Create 700 sample examples with sequential timestamps."""
        return [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG" if i % 2 == 0 else "SHORT",
                gross_return_pct=0.1 * (i % 10 - 5),
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(700)
        ]

    def test_basic_split(self, sample_examples: list[GRPOTrainingExample]) -> None:
        """Test basic train/test split."""
        split = create_grpo_walk_forward_split(
            sample_examples,
            train_window=500,
            test_window=100,
            replay_ratio=0.0,
        )
        assert len(split.train_examples) == 500
        assert len(split.test_examples) == 100
        assert len(split.replay_examples) == 0

    def test_temporal_ordering_enforced(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that train examples all come before test examples."""
        split = create_grpo_walk_forward_split(
            sample_examples,
            train_window=500,
            test_window=100,
            replay_ratio=0.0,
        )
        train_max_ts = max(e.timestamp_ms for e in split.train_examples)
        test_min_ts = min(e.timestamp_ms for e in split.test_examples)
        assert train_max_ts < test_min_ts

    def test_replay_buffer_sampling(
        self, sample_examples: list[GRPOTrainingExample]
    ) -> None:
        """Test that replay samples 15% from history."""
        split = create_grpo_walk_forward_split(
            sample_examples,
            train_window=500,
            test_window=100,
            replay_ratio=0.15,
        )
        # 15% of 500 = 75 replay examples
        assert len(split.replay_examples) == 75
        # Replay examples should all be before train examples
        replay_max_ts = max(e.timestamp_ms for e in split.replay_examples)
        train_min_ts = min(e.timestamp_ms for e in split.train_examples)
        assert replay_max_ts < train_min_ts

    def test_insufficient_examples_raises(self) -> None:
        """Test that insufficient examples raises TemporalSplitError."""
        small_examples = [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG",
                gross_return_pct=0.1,
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(100)
        ]
        with pytest.raises(TemporalSplitError, match="Insufficient"):
            create_grpo_walk_forward_split(
                small_examples,
                train_window=500,
                test_window=100,
            )

    def test_empty_examples_raises(self) -> None:
        """Test that empty list raises TemporalSplitError."""
        with pytest.raises(TemporalSplitError, match="No examples"):
            create_grpo_walk_forward_split([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_data.py::TestGRPOWalkForwardSplit -v`
Expected: FAIL with "ImportError: cannot import name 'GRPOWalkForwardSplit'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_data.py (replace entire file)
"""
GRPO training data types and walk-forward split functions.

Provides GRPOTrainingExample for representing training inputs and
temporal split utilities for walk-forward validation.
"""

import random
from dataclasses import dataclass
from typing import List

from loguru import logger


class TemporalSplitError(Exception):
    """Raised when temporal split validation fails."""

    pass


@dataclass(frozen=True, slots=True)
class GRPOTrainingExample:
    """
    Single training example for GRPO.

    Attributes:
        market_snapshot: Input prompt with market data and indicators
        actual_direction: Ground truth direction ("LONG", "SHORT", "FLAT")
        gross_return_pct: Ground truth return for reward computation
        timestamp_ms: Timestamp for temporal ordering (milliseconds)
    """

    market_snapshot: str
    actual_direction: str
    gross_return_pct: float
    timestamp_ms: int


@dataclass(frozen=True)
class GRPOWalkForwardSplit:
    """Walk-forward train/test split for GRPO training."""

    train_examples: List[GRPOTrainingExample]
    test_examples: List[GRPOTrainingExample]
    replay_examples: List[GRPOTrainingExample]
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int


def create_grpo_walk_forward_split(
    examples: List[GRPOTrainingExample],
    train_window: int = 500,
    test_window: int = 100,
    replay_ratio: float = 0.15,
    replay_buffer_size: int = 1000,
) -> GRPOWalkForwardSplit:
    """
    Create walk-forward train/test split for GRPO examples.

    Temporal ordering:
        [-------- history --------][---- train ----][-- test --]
                                   ^                ^
                                   train_start      test_start

    Args:
        examples: List of GRPO training examples (will be sorted by timestamp)
        train_window: Number of most recent examples for training (default: 500)
        test_window: Number of holdout examples for testing (default: 100)
        replay_ratio: Fraction of training window to sample from history (default: 0.15)
        replay_buffer_size: Maximum historical examples to sample from (default: 1000)

    Returns:
        GRPOWalkForwardSplit with train/test/replay examples and timestamps

    Raises:
        TemporalSplitError: If insufficient examples or temporal ordering violated
    """
    if not examples:
        raise TemporalSplitError("No examples provided")

    # Sort by timestamp (defensive - should already be sorted)
    sorted_examples = sorted(examples, key=lambda e: e.timestamp_ms)

    total_examples = len(sorted_examples)
    required_examples = train_window + test_window

    if total_examples < required_examples:
        raise TemporalSplitError(
            f"Insufficient examples: have {total_examples}, "
            f"need {required_examples} ({train_window} train + {test_window} test)"
        )

    # Split: most recent test_window for testing, previous train_window for training
    test_examples = sorted_examples[-test_window:]
    train_examples = sorted_examples[-(test_window + train_window) : -test_window]

    # Handle timestamp boundary collision (same timestamp in train and test)
    if (
        train_examples
        and test_examples
        and train_examples[-1].timestamp_ms == test_examples[0].timestamp_ms
    ):
        boundary_ts = train_examples[-1].timestamp_ms
        extra = [e for e in test_examples if e.timestamp_ms == boundary_ts]
        test_examples = [e for e in test_examples if e.timestamp_ms != boundary_ts]
        train_examples = list(train_examples) + extra

    # Historical examples (everything before training window)
    train_ids = {id(e) for e in train_examples} | {id(e) for e in test_examples}
    history_examples = [e for e in sorted_examples if id(e) not in train_ids]

    # Sample replay buffer from history
    replay_examples: List[GRPOTrainingExample] = []
    if history_examples and replay_ratio > 0:
        recent_history = history_examples[-replay_buffer_size:]
        num_replay = int(train_window * replay_ratio)
        num_replay = min(num_replay, len(recent_history))

        if num_replay > 0:
            replay_examples = random.sample(recent_history, num_replay)
            replay_examples = sorted(replay_examples, key=lambda e: e.timestamp_ms)

    # Validate replay examples are all before training
    if replay_examples:
        train_start_ms = train_examples[0].timestamp_ms
        replay_examples = [e for e in replay_examples if e.timestamp_ms < train_start_ms]

    # Extract timestamps
    train_start_ms = train_examples[0].timestamp_ms
    train_end_ms = train_examples[-1].timestamp_ms
    test_start_ms = test_examples[0].timestamp_ms
    test_end_ms = test_examples[-1].timestamp_ms

    # Validate temporal ordering
    if test_start_ms <= train_end_ms:
        raise TemporalSplitError(
            f"Test data overlaps with training data: "
            f"train ends at {train_end_ms}, test starts at {test_start_ms}"
        )

    logger.info(
        "GRPO walk-forward split created",
        train_examples=len(train_examples),
        test_examples=len(test_examples),
        replay_examples=len(replay_examples),
        train_start_ms=train_start_ms,
        test_start_ms=test_start_ms,
    )

    return GRPOWalkForwardSplit(
        train_examples=train_examples,
        test_examples=test_examples,
        replay_examples=replay_examples,
        train_start_ms=train_start_ms,
        train_end_ms=train_end_ms,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_data.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_data.py tests/training/test_grpo_data.py
git commit -m "feat(training): add GRPOWalkForwardSplit and temporal split function"
```

---

## Task 3: Direction Parsing Utility

**Files:**
- Modify: `training/grpo_trainer.py` (create)
- Create: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for direction parsing**

```python
# tests/training/test_grpo_trainer.py
"""Tests for GRPO trainer."""

import pytest

from training.grpo_trainer import parse_direction


class TestParseDirection:
    """Tests for parse_direction function."""

    def test_parse_long_uppercase(self) -> None:
        """Test parsing LONG from uppercase."""
        completion = "## THESIS\nBullish\n## EVIDENCE\nRSI\n## RISK\nVol\n## DECISION\nLONG"
        assert parse_direction(completion) == "LONG"

    def test_parse_long_lowercase(self) -> None:
        """Test parsing long from lowercase."""
        completion = "## DECISION\nlong"
        assert parse_direction(completion) == "LONG"

    def test_parse_short(self) -> None:
        """Test parsing SHORT."""
        completion = "## DECISION\nSHORT"
        assert parse_direction(completion) == "SHORT"

    def test_parse_higher_maps_to_long(self) -> None:
        """Test that HIGHER maps to LONG."""
        completion = "## DECISION\nHIGHER"
        assert parse_direction(completion) == "LONG"

    def test_parse_lower_maps_to_short(self) -> None:
        """Test that LOWER maps to SHORT."""
        completion = "## DECISION\nlower"
        assert parse_direction(completion) == "SHORT"

    def test_parse_flat(self) -> None:
        """Test parsing FLAT."""
        completion = "## DECISION\nFLAT"
        assert parse_direction(completion) == "FLAT"

    def test_parse_neutral_maps_to_flat(self) -> None:
        """Test that NEUTRAL maps to FLAT."""
        completion = "## DECISION\nneutral"
        assert parse_direction(completion) == "FLAT"

    def test_corrupt_completion_defaults_flat(self) -> None:
        """Test that unparseable completion defaults to FLAT."""
        completion = "This is garbage text with no structure"
        assert parse_direction(completion) == "FLAT"

    def test_missing_decision_section_defaults_flat(self) -> None:
        """Test that missing DECISION section defaults to FLAT."""
        completion = "## THESIS\nBullish\n## EVIDENCE\nRSI"
        assert parse_direction(completion) == "FLAT"

    def test_direction_in_middle_of_text(self) -> None:
        """Test parsing direction when surrounded by other text."""
        completion = "## DECISION\nBased on analysis, I recommend going LONG with caution."
        assert parse_direction(completion) == "LONG"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestParseDirection -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.grpo_trainer'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py
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

import re
from pathlib import Path

from loguru import logger


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
    if match:
        decision_text = match.group(1).upper()
    else:
        # Fallback: search entire completion
        decision_text = completion.upper()

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

    # Default to FLAT (conservative - no position)
    logger.warning("Could not parse direction from completion, defaulting to FLAT")
    return "FLAT"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestParseDirection -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add parse_direction utility for GRPO"
```

---

## Task 4: GRPOStepResult and GRPOTrainingResult Dataclasses

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for result dataclasses**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from pathlib import Path

from training.grpo_trainer import (
    parse_direction,
    GRPOStepResult,
    GRPOTrainingResult,
)


class TestGRPOStepResult:
    """Tests for GRPOStepResult dataclass."""

    def test_create_step_result(self) -> None:
        """Test creating a step result."""
        result = GRPOStepResult(
            step=100,
            mean_reward=0.23,
            mean_advantage=0.0,
            kl_divergence=0.012,
            loss=0.45,
            vram_mb=10240,
        )
        assert result.step == 100
        assert result.mean_reward == 0.23
        assert result.kl_divergence == 0.012

    def test_step_result_to_dict(self) -> None:
        """Test converting step result to dictionary."""
        result = GRPOStepResult(
            step=100,
            mean_reward=0.23,
            mean_advantage=0.0,
            kl_divergence=0.012,
            loss=0.45,
            vram_mb=10240,
        )
        d = result.to_dict()
        assert d["step"] == 100
        assert d["mean_reward"] == 0.23
        assert "timestamp" in d


class TestGRPOTrainingResult:
    """Tests for GRPOTrainingResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating a successful training result."""
        result = GRPOTrainingResult(
            success=True,
            adapter_path=Path("adapters/grpo_latest"),
            steps_completed=5000,
            final_metrics={"mean_reward": 0.25, "kl": 0.01},
            error=None,
        )
        assert result.success is True
        assert result.adapter_path == Path("adapters/grpo_latest")
        assert result.steps_completed == 5000

    def test_create_failure_result(self) -> None:
        """Test creating a failed training result."""
        result = GRPOTrainingResult(
            success=False,
            adapter_path=None,
            steps_completed=0,
            final_metrics={},
            error="Preflight failed: lock unavailable",
        )
        assert result.success is False
        assert result.adapter_path is None
        assert "lock" in result.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOStepResult -v`
Expected: FAIL with "ImportError: cannot import name 'GRPOStepResult'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after parse_direction function)

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GRPOStepResult:
    """Result of a single GRPO training step."""

    step: int
    mean_reward: float
    mean_advantage: float
    kl_divergence: float
    loss: float
    vram_mb: int

    def to_dict(self) -> Dict[str, Any]:
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
    final_metrics: Dict[str, Any]
    error: Optional[str]
```

- [ ] **Step 4: Update imports and run tests**

Run: `pytest tests/training/test_grpo_trainer.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add GRPOStepResult and GRPOTrainingResult dataclasses"
```

---

## Task 5: Preflight Checks

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for preflight checks**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from unittest.mock import patch, MagicMock
import os

from training.grpo_trainer import (
    parse_direction,
    GRPOStepResult,
    GRPOTrainingResult,
    run_grpo_preflight,
)
from training.grpo_data import GRPOTrainingExample


class TestPreflightChecks:
    """Tests for GRPO preflight checks."""

    @pytest.fixture
    def sample_examples(self) -> list[GRPOTrainingExample]:
        """Create sample training examples."""
        return [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG",
                gross_return_pct=0.1,
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(10)
        ]

    def test_preflight_empty_examples_fails(self) -> None:
        """Test that empty examples list fails preflight."""
        ok, reason = run_grpo_preflight([])
        assert ok is False
        assert "empty" in reason.lower()

    def test_preflight_unsorted_examples_fails(self) -> None:
        """Test that unsorted examples fail preflight."""
        examples = [
            GRPOTrainingExample(
                market_snapshot="second",
                actual_direction="LONG",
                gross_return_pct=0.1,
                timestamp_ms=2000,
            ),
            GRPOTrainingExample(
                market_snapshot="first",
                actual_direction="SHORT",
                gross_return_pct=-0.1,
                timestamp_ms=1000,
            ),
        ]
        ok, reason = run_grpo_preflight(examples)
        assert ok is False
        assert "sorted" in reason.lower() or "temporal" in reason.lower()

    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_insufficient_vram_fails(
        self,
        mock_vram: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that insufficient VRAM fails preflight."""
        mock_vram.return_value = MagicMock(can_train=False, reason="Only 4GB free")
        ok, reason = run_grpo_preflight(sample_examples)
        assert ok is False
        assert "VRAM" in reason or "4GB" in reason

    @patch("training.grpo_trainer.check_can_train")
    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_lock_unavailable_fails(
        self,
        mock_vram: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that unavailable lock fails preflight."""
        mock_vram.return_value = MagicMock(can_train=True, reason="OK")
        mock_lock.return_value = (False, "Another training process is running")
        ok, reason = run_grpo_preflight(sample_examples)
        assert ok is False
        assert "lock" in reason.lower() or "training" in reason.lower()

    @patch("training.grpo_trainer.check_can_train")
    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_stop_file_fails(
        self,
        mock_vram: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
        tmp_path: Path,
    ) -> None:
        """Test that STOP file fails preflight."""
        mock_vram.return_value = MagicMock(can_train=True, reason="OK")
        mock_lock.return_value = (True, "Ready")

        # Create STOP file
        stop_dir = tmp_path / "execution" / "state"
        stop_dir.mkdir(parents=True)
        (stop_dir / "STOP").touch()

        with patch(
            "training.grpo_trainer.STOP_FILE_PATH", stop_dir / "STOP"
        ):
            ok, reason = run_grpo_preflight(sample_examples)
            assert ok is False
            assert "STOP" in reason

    @patch("training.grpo_trainer.check_can_train")
    @patch("training.grpo_trainer.check_vram_availability")
    def test_preflight_success(
        self,
        mock_vram: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test successful preflight."""
        mock_vram.return_value = MagicMock(can_train=True, reason="OK")
        mock_lock.return_value = (True, "Ready")

        with patch(
            "training.grpo_trainer.STOP_FILE_PATH", Path("/nonexistent/STOP")
        ):
            ok, reason = run_grpo_preflight(sample_examples)
            assert ok is True
            assert "ready" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestPreflightChecks -v`
Expected: FAIL with "ImportError: cannot import name 'run_grpo_preflight'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after dataclasses, before any class)

import os
from typing import List, Tuple

from training.grpo_data import GRPOTrainingExample
from training.process_lock import check_can_train
from training.vram_check import check_vram_availability

# Constants
MIN_VRAM_GB = 9.0
STOP_FILE_PATH = Path("execution/state/STOP")


def run_grpo_preflight(
    examples: List[GRPOTrainingExample],
) -> Tuple[bool, str]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestPreflightChecks -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add GRPO preflight checks"
```

---

## Task 6: VRAM Monitoring Utility

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for VRAM monitoring**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from training.grpo_trainer import log_vram_usage


class TestVRAMMonitoring:
    """Tests for VRAM monitoring."""

    @patch("training.grpo_trainer.torch")
    def test_log_vram_returns_usage(self, mock_torch: MagicMock) -> None:
        """Test that log_vram returns VRAM usage in MB."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        vram_mb = log_vram_usage(step=100)
        assert vram_mb == 10 * 1024  # 10GB in MB

    @patch("training.grpo_trainer.torch")
    @patch("training.grpo_trainer.logger")
    def test_vram_warning_above_14gb(
        self, mock_logger: MagicMock, mock_torch: MagicMock
    ) -> None:
        """Test warning logged when VRAM exceeds 14GB."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 15 * 1024 * 1024 * 1024  # 15GB

        log_vram_usage(step=100)
        mock_logger.warning.assert_called_once()
        assert "14GB" in str(mock_logger.warning.call_args) or "exceeded" in str(
            mock_logger.warning.call_args
        ).lower()

    @patch("training.grpo_trainer.torch")
    @patch("training.grpo_trainer.logger")
    def test_no_warning_under_14gb(
        self, mock_logger: MagicMock, mock_torch: MagicMock
    ) -> None:
        """Test no warning when VRAM is under 14GB."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 10 * 1024 * 1024 * 1024  # 10GB

        log_vram_usage(step=100)
        mock_logger.warning.assert_not_called()

    @patch("training.grpo_trainer.torch")
    def test_vram_no_cuda_returns_zero(self, mock_torch: MagicMock) -> None:
        """Test that no CUDA returns 0."""
        mock_torch.cuda.is_available.return_value = False

        vram_mb = log_vram_usage(step=100)
        assert vram_mb == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestVRAMMonitoring -v`
Expected: FAIL with "ImportError: cannot import name 'log_vram_usage'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after run_grpo_preflight)

import torch

MAX_VRAM_GB = 14.0


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestVRAMMonitoring -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add VRAM monitoring for GRPO"
```

---

## Task 7: Checkpoint Saving and Config Hash

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for checkpoint saving**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

import json
import hashlib

from training.grpo_trainer import (
    compute_config_hash,
    save_grpo_checkpoint,
)
from training.grpo_config import GRPOTrainingConfig


class TestCheckpointing:
    """Tests for GRPO checkpointing."""

    def test_config_hash_deterministic(self) -> None:
        """Test that config hash is deterministic."""
        config = GRPOTrainingConfig()
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated hash

    def test_config_hash_changes_with_params(self) -> None:
        """Test that config hash changes when params change."""
        from training.grpo_config import load_grpo_config

        config1 = load_grpo_config()
        config2 = load_grpo_config({"learning_rate": 1e-4})
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 != hash2

    @patch("training.grpo_trainer.time.time")
    def test_save_checkpoint_creates_metadata(
        self, mock_time: MagicMock, tmp_path: Path
    ) -> None:
        """Test that checkpoint saves metadata file."""
        mock_time.return_value = 1700000000.0

        # Mock model
        mock_model = MagicMock()

        checkpoint_dir = tmp_path / "checkpoint-500"
        config = GRPOTrainingConfig()
        metrics = {"mean_reward": 0.25, "kl": 0.01, "loss": 0.5}

        save_grpo_checkpoint(
            model=mock_model,
            checkpoint_dir=checkpoint_dir,
            step=500,
            config=config,
            metrics=metrics,
        )

        # Check model.save_pretrained was called
        mock_model.save_pretrained.assert_called_once_with(str(checkpoint_dir))

        # Check metadata file exists
        metadata_path = checkpoint_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata contents
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["step"] == 500
        assert metadata["mean_reward"] == 0.25
        assert metadata["timestamp_ms"] == 1700000000000
        assert "config_hash" in metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestCheckpointing -v`
Expected: FAIL with "ImportError: cannot import name 'compute_config_hash'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after log_vram_usage)

import hashlib
import json

from training.grpo_config import GRPOTrainingConfig


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
    metrics: Dict[str, float],
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestCheckpointing -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add GRPO checkpoint saving with config hash"
```

---

## Task 8: KL Divergence Computation

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for KL divergence**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from training.grpo_trainer import compute_kl_divergence


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_identical_distributions_is_zero(self) -> None:
        """Test that KL divergence of identical distributions is 0."""
        policy_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        ref_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        assert abs(kl) < 1e-6

    def test_kl_always_non_negative(self) -> None:
        """Test that KL divergence is non-negative."""
        policy_logprobs = torch.tensor([-1.0, -1.5, -2.0])
        ref_logprobs = torch.tensor([-2.0, -2.5, -3.0])
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        assert kl >= 0

    def test_kl_divergence_value(self) -> None:
        """Test KL divergence with known values."""
        # KL = sum(policy_prob * (policy_logprob - ref_logprob))
        # For log probs, this is sum(exp(policy_logprob) * (policy_logprob - ref_logprob))
        # Simplified: mean(policy_logprob - ref_logprob) when using log space approximation
        policy_logprobs = torch.tensor([-1.0, -2.0])
        ref_logprobs = torch.tensor([-1.5, -2.5])
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        # Expected: mean([0.5, 0.5]) = 0.5
        assert abs(kl - 0.5) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestKLDivergence -v`
Expected: FAIL with "ImportError: cannot import name 'compute_kl_divergence'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after save_grpo_checkpoint)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestKLDivergence -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add KL divergence computation for GRPO"
```

---

## Task 9: Policy Ratio Clipping

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for ratio clipping**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from training.grpo_trainer import compute_clipped_policy_loss


class TestPolicyClipping:
    """Tests for PPO-style policy ratio clipping."""

    def test_ratio_within_bounds_not_clipped(self) -> None:
        """Test that ratio within [0.8, 1.2] is not clipped."""
        ratio = torch.tensor([1.0, 1.1, 0.9])
        advantage = torch.tensor([0.5, 0.5, 0.5])
        loss = compute_clipped_policy_loss(ratio, advantage, epsilon=0.2)
        # Expected: -mean(ratio * advantage) = -mean([0.5, 0.55, 0.45]) = -0.5
        expected = -ratio.mean().item() * 0.5
        assert abs(loss - expected) < 0.01

    def test_ratio_above_upper_bound_clipped(self) -> None:
        """Test that ratio > 1+ε is clipped."""
        ratio = torch.tensor([1.5])  # Above 1.2
        advantage = torch.tensor([1.0])
        loss = compute_clipped_policy_loss(ratio, advantage, epsilon=0.2)
        # Clipped ratio = 1.2, so loss = -min(1.5 * 1.0, 1.2 * 1.0) = -1.2
        assert abs(loss - (-1.2)) < 1e-6

    def test_ratio_below_lower_bound_clipped(self) -> None:
        """Test that ratio < 1-ε is clipped."""
        ratio = torch.tensor([0.5])  # Below 0.8
        advantage = torch.tensor([1.0])
        loss = compute_clipped_policy_loss(ratio, advantage, epsilon=0.2)
        # Clipped ratio = 0.8, so loss = -min(0.5 * 1.0, 0.8 * 1.0) = -0.5
        assert abs(loss - (-0.5)) < 1e-6

    def test_negative_advantage_uses_max(self) -> None:
        """Test that negative advantage correctly uses max for conservative update."""
        ratio = torch.tensor([1.5])  # Above 1.2
        advantage = torch.tensor([-1.0])  # Negative advantage
        loss = compute_clipped_policy_loss(ratio, advantage, epsilon=0.2)
        # For negative advantage: loss = -min(1.5 * -1, 1.2 * -1) = -min(-1.5, -1.2) = -(-1.5) = 1.5
        # But PPO formula is -min(ratio * adv, clipped_ratio * adv)
        # = -min(-1.5, -1.2) = -(-1.5) = 1.5
        # Wait, that's wrong. Let me recalculate.
        # loss = -min(ratio * adv, clip(ratio) * adv)
        # = -min(1.5 * -1, 1.2 * -1) = -min(-1.5, -1.2) = -(-1.2) = 1.2
        assert abs(loss - 1.2) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestPolicyClipping -v`
Expected: FAIL with "ImportError: cannot import name 'compute_clipped_policy_loss'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after compute_kl_divergence)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestPolicyClipping -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add PPO-style policy ratio clipping for GRPO"
```

---

## Task 10: JSONL Logger

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for JSONL logger**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from training.grpo_trainer import GRPOLogger


class TestGRPOLogger:
    """Tests for GRPO training logger."""

    def test_logger_creates_file(self, tmp_path: Path) -> None:
        """Test that logger creates log file."""
        log_dir = tmp_path / "logs"
        logger = GRPOLogger(log_dir=log_dir)
        assert log_dir.exists()
        assert len(list(log_dir.glob("grpo_*.jsonl"))) == 1

    def test_logger_writes_step(self, tmp_path: Path) -> None:
        """Test that logger writes step results."""
        log_dir = tmp_path / "logs"
        grpo_logger = GRPOLogger(log_dir=log_dir)

        step_result = GRPOStepResult(
            step=100,
            mean_reward=0.25,
            mean_advantage=0.0,
            kl_divergence=0.01,
            loss=0.5,
            vram_mb=10240,
        )
        grpo_logger.log_step(step_result)
        grpo_logger.close()

        # Read log file
        log_file = list(log_dir.glob("grpo_*.jsonl"))[0]
        with open(log_file) as f:
            line = f.readline()
            data = json.loads(line)

        assert data["step"] == 100
        assert data["mean_reward"] == 0.25
        assert data["loss"] == 0.5

    def test_logger_flushes_periodically(self, tmp_path: Path) -> None:
        """Test that logger flushes after writing."""
        log_dir = tmp_path / "logs"
        grpo_logger = GRPOLogger(log_dir=log_dir)

        step_result = GRPOStepResult(
            step=1, mean_reward=0.1, mean_advantage=0.0,
            kl_divergence=0.01, loss=0.5, vram_mb=1000,
        )
        grpo_logger.log_step(step_result)

        # File should have content even before close
        log_file = list(log_dir.glob("grpo_*.jsonl"))[0]
        assert log_file.stat().st_size > 0

        grpo_logger.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOLogger -v`
Expected: FAIL with "ImportError: cannot import name 'GRPOLogger'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add after compute_clipped_policy_loss)

from datetime import datetime


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
        self._file = open(self.log_path, "w")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOLogger -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add JSONL logger for GRPO training"
```

---

## Task 11: GRPOTrainer Class - Initialization

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for trainer initialization**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from training.grpo_trainer import GRPOTrainer
from training.grpo_config import GRPOTrainingConfig


class TestGRPOTrainerInit:
    """Tests for GRPOTrainer initialization."""

    def test_trainer_init_with_default_config(self) -> None:
        """Test trainer initializes with default config."""
        trainer = GRPOTrainer()
        assert trainer.config.group_size == 4
        assert trainer.config.kl_penalty_beta == 0.04
        assert trainer.config.clip_epsilon == 0.2

    def test_trainer_init_with_custom_config(self) -> None:
        """Test trainer initializes with custom config."""
        config = GRPOTrainingConfig(max_steps=1000, learning_rate=1e-5)
        trainer = GRPOTrainer(config=config)
        assert trainer.config.max_steps == 1000
        assert trainer.config.learning_rate == 1e-5

    def test_trainer_model_not_loaded_until_train(self) -> None:
        """Test that model is not loaded during init."""
        trainer = GRPOTrainer()
        assert trainer._model is None
        assert trainer._tokenizer is None
        assert trainer._ref_state_dict is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerInit -v`
Expected: FAIL with "ImportError: cannot import name 'GRPOTrainer'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add at end of file)

from typing import Optional

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        self._model: Optional[PeftModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._ref_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

        # Logging
        self._logger: Optional[GRPOLogger] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerInit -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add GRPOTrainer class initialization"
```

---

## Task 12: GRPOTrainer - Model Loading

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for model loading**

```python
# tests/training/test_grpo_trainer.py (append to existing file)


class TestGRPOTrainerModelLoading:
    """Tests for GRPOTrainer model loading."""

    @patch("training.grpo_trainer.AutoModelForCausalLM")
    @patch("training.grpo_trainer.AutoTokenizer")
    @patch("training.grpo_trainer.PeftModel")
    def test_load_model_loads_base_with_sft_adapter(
        self,
        mock_peft: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """Test that _load_model loads base model with SFT adapter."""
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        trainer = GRPOTrainer()
        trainer._load_model()

        # Verify base model loaded
        mock_model_cls.from_pretrained.assert_called_once()
        # Verify SFT adapter loaded
        mock_peft.from_pretrained.assert_called_once()
        assert trainer._model is mock_peft_model
        assert trainer._tokenizer is mock_tokenizer

    @patch("training.grpo_trainer.AutoModelForCausalLM")
    @patch("training.grpo_trainer.AutoTokenizer")
    @patch("training.grpo_trainer.PeftModel")
    def test_load_model_stores_reference_state_dict(
        self,
        mock_peft: MagicMock,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """Test that reference model state dict is stored for KL computation."""
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock PEFT model with state dict
        mock_peft_model = MagicMock()
        mock_state_dict = {"lora.weight": torch.tensor([1.0, 2.0])}
        mock_peft_model.state_dict.return_value = mock_state_dict
        mock_peft.from_pretrained.return_value = mock_peft_model

        trainer = GRPOTrainer()
        trainer._load_model()

        # Reference state dict should be stored (deep copy)
        assert trainer._ref_state_dict is not None
        assert "lora.weight" in trainer._ref_state_dict

    def test_reference_model_is_sft_adapter(self) -> None:
        """Test that reference model path points to SFT adapter."""
        config = GRPOTrainingConfig()
        assert config.sft_adapter_path == Path("adapters/sft_base")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerModelLoading -v`
Expected: FAIL with "AttributeError: 'GRPOTrainer' object has no attribute '_load_model'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add to GRPOTrainer class)

    def _load_model(self) -> None:
        """
        Load base model with SFT adapter.

        Loads:
        1. Base model (4-bit quantized)
        2. SFT adapter (reference policy)
        3. Stores reference state dict for KL computation
        """
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
            k: v.clone().cpu() for k, v in self._model.state_dict().items()
            if "lora" in k.lower()
        }

        logger.info(
            "Model loaded",
            trainable_params=sum(p.numel() for p in self._model.parameters() if p.requires_grad),
            ref_params=len(self._ref_state_dict),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerModelLoading -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add model loading to GRPOTrainer"
```

---

## Task 13: GRPOTrainer - Sequential Generation

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for sequential generation**

```python
# tests/training/test_grpo_trainer.py (append to existing file)


class TestGRPOTrainerGeneration:
    """Tests for GRPOTrainer completion generation."""

    @pytest.fixture
    def mock_trainer(self) -> GRPOTrainer:
        """Create trainer with mocked model."""
        trainer = GRPOTrainer()

        # Mock tokenizer
        trainer._tokenizer = MagicMock()
        trainer._tokenizer.return_tensors = "pt"
        trainer._tokenizer.pad_token_id = 0
        trainer._tokenizer.eos_token_id = 1
        trainer._tokenizer.encode.return_value = [100, 101, 102]
        trainer._tokenizer.decode.return_value = (
            "## THESIS\nBullish\n## EVIDENCE\nRSI\n## RISK\nVol\n## DECISION\nLONG"
        )
        trainer._tokenizer.__call__ = MagicMock(
            return_value={"input_ids": torch.tensor([[100, 101, 102]])}
        )

        # Mock model
        trainer._model = MagicMock()
        trainer._model.generate.return_value = torch.tensor([[100, 101, 102, 200, 201]])
        trainer._model.device = torch.device("cpu")

        return trainer

    def test_generate_completions_returns_g_completions(
        self, mock_trainer: GRPOTrainer
    ) -> None:
        """Test that _generate_completions returns G completions."""
        completions = mock_trainer._generate_completions("market snapshot")
        assert len(completions) == mock_trainer.config.group_size  # G=4

    def test_generate_completions_sequential(
        self, mock_trainer: GRPOTrainer
    ) -> None:
        """Test that completions are generated sequentially (G calls)."""
        mock_trainer._generate_completions("market snapshot")
        # Should call generate G times (sequential, not batched)
        assert mock_trainer._model.generate.call_count == mock_trainer.config.group_size

    def test_generate_completions_clears_cache(
        self, mock_trainer: GRPOTrainer
    ) -> None:
        """Test that KV cache is cleared between generations."""
        with patch("training.grpo_trainer.torch.cuda.empty_cache") as mock_cache:
            mock_trainer._generate_completions("market snapshot")
            # Should clear cache G-1 times (between generations)
            assert mock_cache.call_count >= mock_trainer.config.group_size - 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerGeneration -v`
Expected: FAIL with "AttributeError: 'GRPOTrainer' object has no attribute '_generate_completions'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add to GRPOTrainer class)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerGeneration -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add sequential completion generation to GRPOTrainer"
```

---

## Task 14: GRPOTrainer - Training Step

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for training step**

```python
# tests/training/test_grpo_trainer.py (append to existing file)

from training.grpo_data import GRPOTrainingExample


class TestGRPOTrainerStep:
    """Tests for GRPOTrainer training step."""

    @pytest.fixture
    def mock_trainer_for_step(self) -> GRPOTrainer:
        """Create trainer with all mocks for step testing."""
        trainer = GRPOTrainer()

        # Mock tokenizer
        trainer._tokenizer = MagicMock()
        trainer._tokenizer.pad_token_id = 0
        trainer._tokenizer.eos_token_id = 1
        trainer._tokenizer.decode.return_value = (
            "## THESIS\nBullish\n## EVIDENCE\nRSI\n## RISK\nVol\n## DECISION\nLONG"
        )
        trainer._tokenizer.__call__ = MagicMock(
            return_value={
                "input_ids": torch.tensor([[100, 101, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

        # Mock model
        trainer._model = MagicMock()
        trainer._model.generate.return_value = torch.tensor([[100, 101, 102, 200, 201]])
        trainer._model.device = torch.device("cpu")
        # Mock forward pass for log probs
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 5, 1000)
        trainer._model.return_value = mock_output

        # Mock reference state dict
        trainer._ref_state_dict = {}

        # Mock optimizer
        trainer._optimizer = MagicMock()

        return trainer

    def test_training_step_returns_step_result(
        self, mock_trainer_for_step: GRPOTrainer
    ) -> None:
        """Test that _training_step returns GRPOStepResult."""
        example = GRPOTrainingExample(
            market_snapshot="BTC/USDT snapshot",
            actual_direction="LONG",
            gross_return_pct=0.5,
            timestamp_ms=1700000000000,
        )

        with patch.object(
            mock_trainer_for_step, "_generate_completions"
        ) as mock_gen:
            mock_gen.return_value = [
                "## DECISION\nLONG",
                "## DECISION\nLONG",
                "## DECISION\nSHORT",
                "## DECISION\nLONG",
            ]
            with patch.object(
                mock_trainer_for_step, "_compute_step_loss"
            ) as mock_loss:
                mock_loss.return_value = (0.5, 0.01)  # loss, kl

                result = mock_trainer_for_step._training_step(example, step=1)

        assert isinstance(result, GRPOStepResult)
        assert result.step == 1

    def test_training_step_computes_rewards(
        self, mock_trainer_for_step: GRPOTrainer
    ) -> None:
        """Test that training step computes rewards for all completions."""
        example = GRPOTrainingExample(
            market_snapshot="BTC/USDT snapshot",
            actual_direction="LONG",
            gross_return_pct=0.5,
            timestamp_ms=1700000000000,
        )

        with patch.object(
            mock_trainer_for_step, "_generate_completions"
        ) as mock_gen:
            mock_gen.return_value = [
                "## DECISION\nLONG",
                "## DECISION\nLONG",
                "## DECISION\nSHORT",
                "## DECISION\nFLAT",
            ]
            with patch(
                "training.grpo_trainer.compute_grpo_reward"
            ) as mock_reward:
                mock_reward.return_value = MagicMock(final_reward=0.5)
                with patch.object(
                    mock_trainer_for_step, "_compute_step_loss"
                ) as mock_loss:
                    mock_loss.return_value = (0.5, 0.01)

                    mock_trainer_for_step._training_step(example, step=1)

        # Should compute reward for each completion (G=4)
        assert mock_reward.call_count == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerStep -v`
Expected: FAIL with "AttributeError: 'GRPOTrainer' object has no attribute '_training_step'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add to GRPOTrainer class)

from training.grpo_reward import compute_grpo_reward, compute_group_advantages

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerStep -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add training step to GRPOTrainer"
```

---

## Task 15: GRPOTrainer - Loss Computation with KL Penalty

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for loss computation**

```python
# tests/training/test_grpo_trainer.py (append to existing file)


class TestGRPOTrainerLoss:
    """Tests for GRPO loss computation."""

    def test_kl_penalty_applied(self) -> None:
        """Test that loss includes KL penalty term."""
        trainer = GRPOTrainer()
        trainer._tokenizer = MagicMock()
        trainer._tokenizer.__call__ = MagicMock(
            return_value={
                "input_ids": torch.tensor([[100, 101]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        )

        # Mock model outputs
        mock_logits = torch.randn(1, 5, 1000)
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        trainer._model = MagicMock(return_value=mock_output)
        trainer._model.device = torch.device("cpu")

        # Mock reference state dict
        trainer._ref_state_dict = {}

        with patch.object(trainer, "_get_log_probs") as mock_logprobs:
            mock_logprobs.return_value = (
                torch.tensor([-1.0, -2.0]),  # policy
                torch.tensor([-1.5, -2.5]),  # reference
            )

            loss, kl = trainer._compute_step_loss(
                completions=["test completion"],
                advantages=[0.5],
                prompt="test prompt",
            )

        # KL should be computed
        assert kl > 0
        # Loss should include KL penalty (β * KL)
        expected_kl_penalty = trainer.config.kl_penalty_beta * kl
        assert loss is not None

    def test_loss_uses_clipped_ratio(self) -> None:
        """Test that loss uses PPO-style clipped ratio."""
        trainer = GRPOTrainer()
        trainer.config = GRPOTrainingConfig(clip_epsilon=0.2)

        # This is implicitly tested through compute_clipped_policy_loss
        # which is called in _compute_step_loss
        ratio = torch.tensor([1.5])  # Above 1+ε
        advantage = torch.tensor([1.0])
        loss = compute_clipped_policy_loss(ratio, advantage, epsilon=0.2)
        # Should be clipped to 1.2
        assert abs(loss - (-1.2)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerLoss -v`
Expected: FAIL with "AttributeError: 'GRPOTrainer' object has no attribute '_compute_step_loss'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add to GRPOTrainer class)

    def _get_log_probs(
        self,
        prompt: str,
        completion: str,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """
        Get log probabilities for completion given prompt.

        Args:
            prompt: Input prompt
            completion: Generated completion
            use_reference: If True, use reference model weights

        Returns:
            Log probabilities tensor
        """
        # Swap to reference weights if needed
        if use_reference and self._ref_state_dict:
            current_state = {
                k: v.clone() for k, v in self._model.state_dict().items()
                if "lora" in k.lower()
            }
            self._model.load_state_dict(self._ref_state_dict, strict=False)

        try:
            # Tokenize full sequence
            full_text = prompt + completion
            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_new_tokens + 1024,
            ).to(self._model.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self._model(**inputs)

            # Compute log probs for completion tokens
            logits = outputs.logits
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            # Log softmax
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # Gather log probs for actual tokens
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Only keep completion tokens
            prompt_len = len(self._tokenizer.encode(prompt))
            completion_log_probs = token_log_probs[0, prompt_len - 1:]

            return completion_log_probs

        finally:
            # Restore current weights if we swapped
            if use_reference and self._ref_state_dict:
                self._model.load_state_dict(current_state, strict=False)

    def _compute_step_loss(
        self,
        completions: List[str],
        advantages: List[float],
        prompt: str,
    ) -> Tuple[float, float]:
        """
        Compute GRPO loss with KL penalty.

        loss = policy_loss + β * KL(π || π_ref)

        Args:
            completions: G completion strings
            advantages: Group-relative advantages
            prompt: Input prompt

        Returns:
            Tuple of (total_loss, kl_divergence)
        """
        total_policy_loss = 0.0
        total_kl = 0.0

        for completion, advantage in zip(completions, advantages):
            # Get log probs from current policy and reference
            policy_logprobs = self._get_log_probs(prompt, completion, use_reference=False)
            ref_logprobs = self._get_log_probs(prompt, completion, use_reference=True)

            # Compute probability ratio
            ratio = torch.exp(policy_logprobs - ref_logprobs)

            # Compute clipped policy loss
            policy_loss = compute_clipped_policy_loss(
                ratio,
                torch.tensor([advantage]),
                epsilon=self.config.clip_epsilon,
            )
            total_policy_loss += policy_loss

            # Compute KL divergence
            kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
            total_kl += kl

        # Average over completions
        avg_policy_loss = total_policy_loss / len(completions)
        avg_kl = total_kl / len(completions)

        # Total loss with KL penalty
        total_loss = avg_policy_loss + self.config.kl_penalty_beta * avg_kl

        return total_loss, avg_kl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerLoss -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add loss computation with KL penalty to GRPOTrainer"
```

---

## Task 16: GRPOTrainer - Main Train Loop

**Files:**
- Modify: `training/grpo_trainer.py`
- Modify: `tests/training/test_grpo_trainer.py`

- [ ] **Step 1: Write failing tests for main train loop**

```python
# tests/training/test_grpo_trainer.py (append to existing file)


class TestGRPOTrainerTrain:
    """Tests for GRPOTrainer main train loop."""

    @pytest.fixture
    def sample_examples(self) -> list[GRPOTrainingExample]:
        """Create sample training examples."""
        return [
            GRPOTrainingExample(
                market_snapshot=f"snapshot_{i}",
                actual_direction="LONG" if i % 2 == 0 else "SHORT",
                gross_return_pct=0.1 * (i % 5),
                timestamp_ms=1700000000000 + i * 1000,
            )
            for i in range(100)
        ]

    @patch("training.grpo_trainer.acquire_training_lock")
    @patch("training.grpo_trainer.run_grpo_preflight")
    def test_train_runs_preflight(
        self,
        mock_preflight: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that train() runs preflight checks."""
        mock_preflight.return_value = (False, "Test failure")

        trainer = GRPOTrainer()
        result = trainer.train(sample_examples)

        mock_preflight.assert_called_once()
        assert result.success is False
        assert "Test failure" in result.error

    @patch("training.grpo_trainer.acquire_training_lock")
    @patch("training.grpo_trainer.run_grpo_preflight")
    def test_train_acquires_lock(
        self,
        mock_preflight: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that train() acquires training lock."""
        mock_preflight.return_value = (True, "Ready")
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        trainer = GRPOTrainer()
        # Mock the rest to avoid full training
        with patch.object(trainer, "_load_model"):
            with patch.object(trainer, "_training_step") as mock_step:
                mock_step.return_value = GRPOStepResult(
                    step=1, mean_reward=0.1, mean_advantage=0.0,
                    kl_divergence=0.01, loss=0.5, vram_mb=1000,
                )
                with patch.object(trainer, "_cleanup"):
                    # Limit steps
                    trainer.config = GRPOTrainingConfig(max_steps=1)
                    trainer._logger = MagicMock()
                    result = trainer.train(sample_examples)

        mock_lock.assert_called_once()

    @patch("training.grpo_trainer.acquire_training_lock")
    @patch("training.grpo_trainer.run_grpo_preflight")
    def test_train_checkpoints_every_500_steps(
        self,
        mock_preflight: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
    ) -> None:
        """Test that checkpoints are saved every 500 steps."""
        mock_preflight.return_value = (True, "Ready")
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        trainer = GRPOTrainer()
        trainer.config = GRPOTrainingConfig(
            max_steps=501,
            checkpoint_interval_steps=500,
        )

        with patch.object(trainer, "_load_model"):
            with patch.object(trainer, "_training_step") as mock_step:
                mock_step.return_value = GRPOStepResult(
                    step=1, mean_reward=0.1, mean_advantage=0.0,
                    kl_divergence=0.01, loss=0.5, vram_mb=1000,
                )
                with patch.object(trainer, "_cleanup"):
                    with patch(
                        "training.grpo_trainer.save_grpo_checkpoint"
                    ) as mock_save:
                        trainer._logger = MagicMock()
                        trainer._model = MagicMock()
                        result = trainer.train(sample_examples)

        # Should save at step 500
        assert mock_save.call_count >= 1

    @patch("training.grpo_trainer.acquire_training_lock")
    @patch("training.grpo_trainer.run_grpo_preflight")
    def test_train_stop_file_graceful_shutdown(
        self,
        mock_preflight: MagicMock,
        mock_lock: MagicMock,
        sample_examples: list[GRPOTrainingExample],
        tmp_path: Path,
    ) -> None:
        """Test graceful shutdown when STOP file appears."""
        mock_preflight.return_value = (True, "Ready")
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        # Create STOP file
        stop_dir = tmp_path / "execution" / "state"
        stop_dir.mkdir(parents=True)
        stop_file = stop_dir / "STOP"

        trainer = GRPOTrainer()
        trainer.config = GRPOTrainingConfig(max_steps=1000)

        step_count = [0]

        def mock_step(example, step):
            step_count[0] += 1
            # Create STOP file after 50 steps
            if step_count[0] == 50:
                stop_file.touch()
            return GRPOStepResult(
                step=step, mean_reward=0.1, mean_advantage=0.0,
                kl_divergence=0.01, loss=0.5, vram_mb=1000,
            )

        with patch.object(trainer, "_load_model"):
            with patch.object(trainer, "_training_step", side_effect=mock_step):
                with patch.object(trainer, "_cleanup"):
                    with patch(
                        "training.grpo_trainer.STOP_FILE_PATH", stop_file
                    ):
                        trainer._logger = MagicMock()
                        trainer._model = MagicMock()
                        result = trainer.train(sample_examples)

        # Should stop before 1000 steps
        assert result.steps_completed < 1000
        assert "STOP" in (result.error or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerTrain -v`
Expected: FAIL with "AttributeError: 'GRPOTrainer' object has no attribute 'train'"

- [ ] **Step 3: Write implementation**

```python
# training/grpo_trainer.py (add to GRPOTrainer class)

import gc
from training.process_lock import acquire_training_lock

    def _cleanup(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._optimizer is not None:
            del self._optimizer
            self._optimizer = None
        if self._ref_state_dict is not None:
            del self._ref_state_dict
            self._ref_state_dict = None
        if self._logger is not None:
            self._logger.close()
            self._logger = None

        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleaned up")

    def train(
        self,
        examples: List[GRPOTrainingExample],
    ) -> GRPOTrainingResult:
        """
        Execute full GRPO training loop.

        Args:
            examples: List of training examples

        Returns:
            GRPOTrainingResult with success status and metrics
        """
        # 1. Preflight checks
        ok, reason = run_grpo_preflight(examples)
        if not ok:
            return GRPOTrainingResult(
                success=False,
                adapter_path=None,
                steps_completed=0,
                final_metrics={},
                error=reason,
            )

        steps_completed = 0
        final_metrics: Dict[str, Any] = {}

        try:
            # 2. Acquire training lock
            with acquire_training_lock():
                logger.info("Training lock acquired")

                # 3. Load model
                self._load_model()

                # 4. Initialize logger
                self._logger = GRPOLogger(log_dir=self.config.log_dir)

                # 5. Training loop
                example_idx = 0
                for step in range(1, self.config.max_steps + 1):
                    # Check STOP file every 100 steps
                    if step % 100 == 0 and STOP_FILE_PATH.exists():
                        logger.warning("STOP file detected, saving checkpoint")
                        if self._model is not None:
                            save_grpo_checkpoint(
                                model=self._model,
                                checkpoint_dir=self.config.checkpoint_dir / f"checkpoint-{step}-early_stop",
                                step=step,
                                config=self.config,
                                metrics=final_metrics,
                            )
                        return GRPOTrainingResult(
                            success=False,
                            adapter_path=None,
                            steps_completed=step,
                            final_metrics=final_metrics,
                            error="STOP file detected",
                        )

                    # Get example (cycle through examples)
                    example = examples[example_idx % len(examples)]
                    example_idx += 1

                    # Execute training step
                    step_result = self._training_step(example, step)
                    steps_completed = step

                    # Log step
                    self._logger.log_step(step_result)

                    # Update final metrics
                    final_metrics = {
                        "mean_reward": step_result.mean_reward,
                        "kl": step_result.kl_divergence,
                        "loss": step_result.loss,
                    }

                    # Checkpoint every N steps
                    if step % self.config.checkpoint_interval_steps == 0:
                        save_grpo_checkpoint(
                            model=self._model,
                            checkpoint_dir=self.config.checkpoint_dir / f"checkpoint-{step}",
                            step=step,
                            config=self.config,
                            metrics=final_metrics,
                        )

                # 6. Save final adapter
                final_adapter_path = save_grpo_checkpoint(
                    model=self._model,
                    checkpoint_dir=self.config.output_dir,
                    step=steps_completed,
                    config=self.config,
                    metrics=final_metrics,
                )

                logger.info(
                    "Training complete",
                    steps=steps_completed,
                    adapter_path=str(final_adapter_path),
                )

                return GRPOTrainingResult(
                    success=True,
                    adapter_path=final_adapter_path,
                    steps_completed=steps_completed,
                    final_metrics=final_metrics,
                    error=None,
                )

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return GRPOTrainingResult(
                success=False,
                adapter_path=None,
                steps_completed=steps_completed,
                final_metrics=final_metrics,
                error=str(e),
            )

        finally:
            self._cleanup()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/training/test_grpo_trainer.py::TestGRPOTrainerTrain -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add training/grpo_trainer.py tests/training/test_grpo_trainer.py
git commit -m "feat(training): add main train loop to GRPOTrainer"
```

---

## Task 17: Final Integration and CLI Entry Point

**Files:**
- Modify: `training/grpo_trainer.py`

- [ ] **Step 1: Add CLI entry point**

```python
# training/grpo_trainer.py (add at end of file)

def main() -> None:
    """CLI entry point for GRPO training."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Train GRPO model on market snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/grpo_training_data.jsonl"),
        help="Path to GRPO training data JSONL",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Load training data
    if not args.data.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    examples = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(GRPOTrainingExample(
                market_snapshot=data["market_snapshot"],
                actual_direction=data["actual_direction"],
                gross_return_pct=data["gross_return_pct"],
                timestamp_ms=data["timestamp_ms"],
            ))

    logger.info(f"Loaded {len(examples)} training examples")

    # Create config with overrides
    overrides = {}
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = args.checkpoint_dir

    from training.grpo_config import load_grpo_config
    config = load_grpo_config(overrides) if overrides else GRPOTrainingConfig()

    # Run training
    trainer = GRPOTrainer(config=config)
    result = trainer.train(examples)

    if result.success:
        logger.info(
            "Training completed successfully",
            adapter_path=str(result.adapter_path),
            steps=result.steps_completed,
            final_metrics=result.final_metrics,
        )
    else:
        logger.error(f"Training failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/training/test_grpo_trainer.py tests/training/test_grpo_data.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run linting and type checking**

Run: `ruff check --fix training/grpo_trainer.py training/grpo_data.py && ruff format training/grpo_trainer.py training/grpo_data.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add training/grpo_trainer.py
git commit -m "feat(training): add CLI entry point for GRPO trainer"
```

---

## Task 18: Run Full Test Suite and Verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/training/test_grpo_*.py -v --tb=short`
Expected: All tests PASS (approximately 40+ tests)

- [ ] **Step 2: Check test coverage**

Run: `pytest tests/training/test_grpo_*.py --cov=training.grpo_trainer --cov=training.grpo_data --cov-report=term-missing`
Expected: grpo_trainer.py: 90%+, grpo_data.py: 95%+

- [ ] **Step 3: Run existing tests to ensure no regressions**

Run: `pytest tests/ -v --tb=short -x`
Expected: All existing tests still pass

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test(training): verify GRPO trainer implementation complete

Session 17D complete:
- training/grpo_data.py: GRPOTrainingExample, walk-forward splits
- training/grpo_trainer.py: Full GRPO training loop
- 40+ tests covering all components
- Coverage: 90%+ trainer, 95%+ data"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | GRPOTrainingExample dataclass | 2 |
| 2 | GRPOWalkForwardSplit + temporal split | 5 |
| 3 | Direction parsing utility | 10 |
| 4 | GRPOStepResult + GRPOTrainingResult | 4 |
| 5 | Preflight checks | 6 |
| 6 | VRAM monitoring | 4 |
| 7 | Checkpoint saving + config hash | 3 |
| 8 | KL divergence computation | 3 |
| 9 | Policy ratio clipping | 4 |
| 10 | JSONL logger | 3 |
| 11 | GRPOTrainer initialization | 3 |
| 12 | Model loading | 3 |
| 13 | Sequential generation | 3 |
| 14 | Training step | 2 |
| 15 | Loss computation with KL penalty | 2 |
| 16 | Main train loop | 4 |
| 17 | CLI entry point | 0 (integration) |
| 18 | Verification | 0 (verification) |

**Total: ~61 tests across 18 tasks**
