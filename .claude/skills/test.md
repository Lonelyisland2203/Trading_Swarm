---
name: test
description: Generate pytest tests for a new module following project conventions
---

# Generate Module Tests

Create pytest tests for the specified module following project test conventions.

## File Placement

Mirror source structure exactly:
- `training/grpo_trainer.py` → `tests/training/test_grpo_trainer.py`
- `data/indicators.py` → `tests/data/test_indicators.py`
- `execution/client.py` → `tests/execution/test_client.py`

## Required Rules

### 1. Temporal Isolation
```python
# NEVER use
datetime.now()
time.time()
pd.Timestamp.now()

# ALWAYS use fixed timestamps
FIXED_TS = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")

@pytest.fixture
def as_of() -> pd.Timestamp:
    return pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
```

### 2. Mock Ollama
```python
@pytest.fixture
def mock_ollama(mocker) -> MagicMock:
    return mocker.patch("ollama.generate", return_value={
        "response": '{"signal": "neutral", "confidence": 0.5}'
    })
```
Never hit a real model in tests.

### 3. Mock Binance/CCXT
```python
@pytest.fixture
def mock_exchange(mocker) -> MagicMock:
    exchange = mocker.patch("ccxt.binance")
    exchange.return_value.fetch_ohlcv.return_value = [
        [1705320000000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
    ]
    return exchange
```
Never hit a real exchange in tests.

### 4. Pre-flight Order Tests
If module involves training or execution:
```python
def test_preflight_order_enforced() -> None:
    """Verify Data→Temporal→VRAM→Lock→Load order."""
    ...
```

### 5. VRAM Isolation Tests
If module loads models:
```python
def test_model_unloaded_after_use() -> None:
    """Verify OLLAMA_KEEP_ALIVE=0 behavior."""
    ...

def test_no_concurrent_model_loads() -> None:
    """Verify semaphore prevents dual model loading."""
    ...
```

### 6. Asymmetric Penalty Tests
If module involves reward calculations:
```python
@pytest.mark.parametrize("pred,outcome,expected", [
    ("bullish", "up", 1.0),
    ("bullish", "down", -1.5),  # Harsher penalty
    ("bearish", "down", 1.0),
    ("bearish", "up", -0.8),
])
def test_asymmetric_rewards(pred: str, outcome: str, expected: float) -> None:
    ...
```

### 7. Type Hints
All test functions must have type annotations:
```python
def test_example(mock_ollama: MagicMock, as_of: pd.Timestamp) -> None:
    ...
```

### 8. Fixtures and Parametrize
- Use `@pytest.fixture` for reusable test state
- Use `@pytest.mark.parametrize` for input variations
- Use `conftest.py` for shared fixtures

## Test Structure

```python
"""Tests for module_name."""
import pytest
from unittest.mock import MagicMock
import pandas as pd

from module_path import TargetClass, target_function


class TestTargetClass:
    """Tests for TargetClass."""

    def test_init(self) -> None:
        ...

    def test_method_happy_path(self) -> None:
        ...

    def test_method_edge_case(self) -> None:
        ...

    def test_method_error_handling(self) -> None:
        ...
```
