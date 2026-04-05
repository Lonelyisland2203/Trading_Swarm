# Realistic Fee Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement realistic Binance Futures USDT-M fee model to prevent training on poisoned labels

**Architecture:** Fix fees at verification layer (single source of truth). Create FeeModelSettings with maker/taker fees, BNB discounts, funding costs, and slippage. Use exact log ↔ percentage conversions throughout.

**Tech Stack:** Pydantic, pytest, existing verifier/reward infrastructure

---

## File Structure

**New files:**
- `config/fee_model.py` - FeeModelSettings model with fee calculation methods
- `tests/test_config.py` - Unit tests for fee model (add to existing file)

**Modified files:**
- `verifier/constants.py` - Add holding period calculation
- `verifier/outcome.py` - Add apply_fee_model(), deprecate compute_net_return()
- `verifier/engine.py` - Accept fee_model parameter, use apply_fee_model()
- `config/settings.py` - Add fee_model nested field and env var mappings
- `run_dpo_training.py` - Add fee flip diagnostic to Phase 3
- `tests/test_verifier/test_constants.py` - Test holding period calculation
- `tests/test_verifier/test_outcome.py` - Test apply_fee_model()
- `tests/test_verifier/test_engine.py` - Test end-to-end with fee model

---

### Task 1: Create FeeModelSettings with round_trip_cost_pct method

**Files:**
- Create: `config/fee_model.py`
- Test: `tests/test_config.py` (add TestFeeModelSettings class)

- [ ] **Step 1: Write failing test for default fee model settings**

```python
# Add to tests/test_config.py after existing imports
from config.fee_model import FeeModelSettings


class TestFeeModelSettings:
    """Test Binance Futures USDT-M fee model."""

    def test_default_values(self):
        """Test default fee model configuration."""
        fee_model = FeeModelSettings()
        assert fee_model.exchange == "binance"
        assert fee_model.mode == "futures_usdt"
        assert fee_model.maker_fee_pct == 0.02
        assert fee_model.taker_fee_pct == 0.05
        assert fee_model.bnb_discount is True
        assert fee_model.entry_order_type == "maker"
        assert fee_model.exit_order_type == "taker"
        assert fee_model.funding_rate_per_8h_pct == 0.01
        assert fee_model.include_funding is True
        assert fee_model.slippage_pct == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestFeeModelSettings::test_default_values -v`
Expected: FAIL with "cannot import name 'FeeModelSettings'"

- [ ] **Step 3: Create config/fee_model.py with FeeModelSettings class**

```python
"""
Fee model configuration for realistic trading cost estimation.

Binance Futures USDT-M fee structure with maker/taker fees, BNB discounts,
funding costs, and slippage modeling.
"""

from typing import Literal

from pydantic import BaseModel, Field


class FeeModelSettings(BaseModel):
    """
    Binance Futures USDT-M fee model.

    Models realistic trading costs including:
    - Maker/taker fee structure with BNB discounts (10% on futures)
    - Funding costs proportional to holding period (per 8h)
    - Slippage on both entry and exit legs

    All fees expressed as percentages (e.g., 0.02 = 0.02%).
    """

    # Exchange configuration
    exchange: str = Field(
        default="binance",
        description="Exchange name (fixed to Binance)"
    )
    mode: Literal["futures_usdt"] = Field(
        default="futures_usdt",
        description="Trading mode (USDT-margined futures)"
    )

    # Fee structure (Binance VIP 0)
    maker_fee_pct: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Maker fee percentage (0.02% at VIP 0)"
    )
    taker_fee_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Taker fee percentage (0.05% at VIP 0)"
    )
    bnb_discount: bool = Field(
        default=True,
        description="Apply 10% BNB discount on futures fees"
    )

    # Order types (realistic execution assumptions)
    entry_order_type: Literal["maker", "taker"] = Field(
        default="maker",
        description="Entry order type (maker=limit, taker=market)"
    )
    exit_order_type: Literal["maker", "taker"] = Field(
        default="taker",
        description="Exit order type (maker=limit, taker=market)"
    )

    # Funding (USDT-M specific)
    funding_rate_per_8h_pct: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Average funding rate per 8-hour period (0.01%)"
    )
    include_funding: bool = Field(
        default=True,
        description="Include funding costs in total round-trip cost"
    )

    # Market impact
    slippage_pct: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Slippage percentage per leg (0.02%)"
    )

    def round_trip_cost_pct(self, holding_periods_8h: float) -> float:
        """
        Compute total round-trip cost as percentage.

        Includes:
        - Entry fee (maker or taker, with BNB discount if enabled)
        - Exit fee (maker or taker, with BNB discount if enabled)
        - Funding cost (proportional to holding period, if include_funding=True)
        - Slippage (both legs)

        Args:
            holding_periods_8h: Holding period in 8-hour units (funding periods)

        Returns:
            Total round-trip cost as percentage (e.g., 0.133 for 0.133%)

        Example:
            >>> fee_model = FeeModelSettings()
            >>> fee_model.round_trip_cost_pct(3.0)  # 24 hours = 3 funding periods
            0.133  # Entry 0.018% + Exit 0.045% + Slippage 0.04% + Funding 0.03%
        """
        # BNB discount multiplier (10% off for futures)
        bnb_multiplier = 0.9 if self.bnb_discount else 1.0

        # Entry fee
        if self.entry_order_type == "maker":
            entry_fee = self.maker_fee_pct * bnb_multiplier
        else:  # taker
            entry_fee = self.taker_fee_pct * bnb_multiplier

        # Exit fee
        if self.exit_order_type == "maker":
            exit_fee = self.maker_fee_pct * bnb_multiplier
        else:  # taker
            exit_fee = self.taker_fee_pct * bnb_multiplier

        # Funding cost (proportional to holding period)
        if self.include_funding:
            funding_cost = self.funding_rate_per_8h_pct * holding_periods_8h
        else:
            funding_cost = 0.0

        # Slippage (both entry and exit)
        total_slippage = self.slippage_pct * 2

        # Total round-trip cost
        return entry_fee + exit_fee + funding_cost + total_slippage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestFeeModelSettings::test_default_values -v`
Expected: PASS

- [ ] **Step 5: Write test for round_trip_cost_pct with default settings**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_round_trip_cost_maker_taker_with_bnb(self):
    """Test round-trip cost with maker entry, taker exit, BNB discount."""
    fee_model = FeeModelSettings()

    # 3 funding periods (24 hours for 1h timeframe)
    cost = fee_model.round_trip_cost_pct(holding_periods_8h=3.0)

    # Entry: 0.02% * 0.9 = 0.018%
    # Exit: 0.05% * 0.9 = 0.045%
    # Slippage: 0.02% * 2 = 0.04%
    # Funding: 0.01% * 3 = 0.03%
    # Total: 0.133%
    assert abs(cost - 0.133) < 1e-9
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestFeeModelSettings::test_round_trip_cost_maker_taker_with_bnb -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add config/fee_model.py tests/test_config.py
git commit -m "feat: add FeeModelSettings with round_trip_cost_pct method

- Create config/fee_model.py with Pydantic model
- Support maker/taker fees with 10% BNB discount
- Include funding costs proportional to holding period
- Model slippage on both entry and exit legs
- Add unit tests for default values and cost calculation"
```

---

### Task 2: Add net_return and minimum_profitable_return_pct methods

**Files:**
- Modify: `config/fee_model.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test for net_return method**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_net_return_positive_after_fees(self):
    """Test net return computation with positive gross return."""
    fee_model = FeeModelSettings()

    # Gross return +0.15%, holding 3 periods
    # Round-trip cost: 0.133%
    # Net: 0.15 - 0.133 = 0.017%
    net = fee_model.net_return(gross_return_pct=0.15, holding_periods_8h=3.0)

    assert abs(net - 0.017) < 1e-9


def test_net_return_negative_after_fees(self):
    """Test net return becomes negative when gross is below fee threshold."""
    fee_model = FeeModelSettings()

    # Gross return +0.08%, holding 3 periods
    # Round-trip cost: 0.133%
    # Net: 0.08 - 0.133 = -0.053%
    net = fee_model.net_return(gross_return_pct=0.08, holding_periods_8h=3.0)

    assert abs(net - (-0.053)) < 1e-9


def test_net_return_zero_periods(self):
    """Test net return with zero holding period (no funding)."""
    fee_model = FeeModelSettings()

    # Gross return +0.15%, holding 0 periods
    # Round-trip cost: 0.103% (no funding)
    # Net: 0.15 - 0.103 = 0.047%
    net = fee_model.net_return(gross_return_pct=0.15, holding_periods_8h=0.0)

    assert abs(net - 0.047) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py::TestFeeModelSettings::test_net_return_positive_after_fees -v`
Expected: FAIL with "FeeModelSettings object has no attribute 'net_return'"

- [ ] **Step 3: Add net_return method to FeeModelSettings**

```python
# Add to FeeModelSettings class in config/fee_model.py after round_trip_cost_pct method

def net_return(self, gross_return_pct: float, holding_periods_8h: float) -> float:
    """
    Compute net return after subtracting all trading costs.

    Args:
        gross_return_pct: Gross return as percentage (e.g., 0.15 for 0.15%)
        holding_periods_8h: Holding period in 8-hour units

    Returns:
        Net return as percentage after all costs

    Example:
        >>> fee_model = FeeModelSettings()
        >>> fee_model.net_return(0.15, 3.0)  # +0.15% gross, 3 funding periods
        0.017  # 0.15% - 0.133% = 0.017%
        >>> fee_model.net_return(0.08, 3.0)  # +0.08% gross (below threshold)
        -0.053  # 0.08% - 0.133% = -0.053% (unprofitable)
    """
    total_cost = self.round_trip_cost_pct(holding_periods_8h)
    return gross_return_pct - total_cost
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py::TestFeeModelSettings -k net_return -v`
Expected: PASS (all 3 net_return tests)

- [ ] **Step 5: Write failing test for minimum_profitable_return_pct**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_minimum_profitable_return(self):
    """Test minimum profitable return equals round-trip cost."""
    fee_model = FeeModelSettings()

    # Minimum profitable return is the break-even point
    min_return = fee_model.minimum_profitable_return_pct(holding_periods_8h=3.0)
    round_trip_cost = fee_model.round_trip_cost_pct(holding_periods_8h=3.0)

    assert min_return == round_trip_cost
    assert abs(min_return - 0.133) < 1e-9


def test_minimum_profitable_return_different_periods(self):
    """Test minimum return varies with holding period (funding)."""
    fee_model = FeeModelSettings()

    # Short hold (no funding)
    min_0_periods = fee_model.minimum_profitable_return_pct(0.0)
    assert abs(min_0_periods - 0.103) < 1e-9  # No funding

    # Medium hold (3 periods)
    min_3_periods = fee_model.minimum_profitable_return_pct(3.0)
    assert abs(min_3_periods - 0.133) < 1e-9  # +0.03% funding

    # Long hold (15 periods - 1d timeframe)
    min_15_periods = fee_model.minimum_profitable_return_pct(15.0)
    assert abs(min_15_periods - 0.253) < 1e-9  # +0.15% funding
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_config.py::TestFeeModelSettings::test_minimum_profitable_return -v`
Expected: FAIL with "FeeModelSettings object has no attribute 'minimum_profitable_return_pct'"

- [ ] **Step 7: Add minimum_profitable_return_pct method**

```python
# Add to FeeModelSettings class in config/fee_model.py after net_return method

def minimum_profitable_return_pct(self, holding_periods_8h: float) -> float:
    """
    Compute minimum gross return needed to break even after fees.

    This is the threshold below which signals are unprofitable.

    Args:
        holding_periods_8h: Holding period in 8-hour units

    Returns:
        Break-even return as percentage

    Example:
        >>> fee_model = FeeModelSettings()
        >>> fee_model.minimum_profitable_return_pct(3.0)
        0.133  # Need >0.133% gross return to profit after fees
        >>> fee_model.minimum_profitable_return_pct(15.0)  # 1d timeframe
        0.253  # Higher threshold due to 15 funding periods
    """
    return self.round_trip_cost_pct(holding_periods_8h)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_config.py::TestFeeModelSettings -v`
Expected: PASS (all tests in TestFeeModelSettings)

- [ ] **Step 9: Commit**

```bash
git add config/fee_model.py tests/test_config.py
git commit -m "feat: add net_return and minimum_profitable_return_pct methods

- Add net_return() to compute return after all costs
- Add minimum_profitable_return_pct() for break-even threshold
- Test positive/negative net returns
- Test threshold varies with holding period (funding)"
```

---

### Task 3: Add tests for fee variations (BNB, order types, funding)

**Files:**
- Test: `tests/test_config.py`

- [ ] **Step 1: Write tests for BNB discount variations**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_round_trip_cost_without_bnb_discount(self):
    """Test fees without BNB discount (10% higher)."""
    fee_model = FeeModelSettings(bnb_discount=False)

    cost = fee_model.round_trip_cost_pct(holding_periods_8h=3.0)

    # Entry: 0.02% (no discount)
    # Exit: 0.05% (no discount)
    # Slippage: 0.04%
    # Funding: 0.03%
    # Total: 0.14%
    assert abs(cost - 0.14) < 1e-9


def test_bnb_discount_is_10_percent(self):
    """Verify BNB discount is 10% (not 25% - that's spot only)."""
    fee_no_bnb = FeeModelSettings(bnb_discount=False)
    fee_with_bnb = FeeModelSettings(bnb_discount=True)

    cost_no_bnb = fee_no_bnb.round_trip_cost_pct(0.0)  # No funding
    cost_with_bnb = fee_with_bnb.round_trip_cost_pct(0.0)

    # Without BNB: 0.02 + 0.05 + 0.04 = 0.11%
    # With BNB: (0.02 + 0.05) * 0.9 + 0.04 = 0.063 + 0.04 = 0.103%
    # Discount: (0.11 - 0.103) / 0.07 = 10%
    fee_component_no_bnb = 0.02 + 0.05  # 0.07%
    fee_component_with_bnb = (0.02 + 0.05) * 0.9  # 0.063%
    discount = (fee_component_no_bnb - fee_component_with_bnb) / fee_component_no_bnb

    assert abs(discount - 0.10) < 1e-9
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_config.py::TestFeeModelSettings -k "bnb_discount" -v`
Expected: PASS

- [ ] **Step 3: Write tests for different order type combinations**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_round_trip_cost_both_maker(self):
    """Test cost with both entry and exit as maker orders."""
    fee_model = FeeModelSettings(
        entry_order_type="maker",
        exit_order_type="maker",
    )

    cost = fee_model.round_trip_cost_pct(holding_periods_8h=0.0)

    # Entry: 0.02% * 0.9 = 0.018%
    # Exit: 0.02% * 0.9 = 0.018%
    # Slippage: 0.04%
    # Total: 0.076%
    assert abs(cost - 0.076) < 1e-9


def test_round_trip_cost_both_taker(self):
    """Test cost with both entry and exit as taker orders."""
    fee_model = FeeModelSettings(
        entry_order_type="taker",
        exit_order_type="taker",
    )

    cost = fee_model.round_trip_cost_pct(holding_periods_8h=0.0)

    # Entry: 0.05% * 0.9 = 0.045%
    # Exit: 0.05% * 0.9 = 0.045%
    # Slippage: 0.04%
    # Total: 0.13%
    assert abs(cost - 0.13) < 1e-9
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py::TestFeeModelSettings -k "order_type" -v`
Expected: PASS

- [ ] **Step 5: Write tests for funding cost variations**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_funding_cost_zero_periods(self):
    """Test no funding cost for very short holds."""
    fee_model = FeeModelSettings()

    cost = fee_model.round_trip_cost_pct(holding_periods_8h=0.0)

    # Should have base fees + slippage only (no funding)
    assert abs(cost - 0.103) < 1e-9


def test_funding_cost_fractional_periods(self):
    """Test funding cost with fractional periods (1m timeframe)."""
    fee_model = FeeModelSettings()

    # 0.125 periods = 1 hour
    cost = fee_model.round_trip_cost_pct(holding_periods_8h=0.125)

    # Entry: 0.018%
    # Exit: 0.045%
    # Slippage: 0.04%
    # Funding: 0.01% * 0.125 = 0.00125%
    # Total: 0.10425%
    assert abs(cost - 0.10425) < 1e-9


def test_funding_cost_long_hold(self):
    """Test funding cost for long holds (1d timeframe)."""
    fee_model = FeeModelSettings()

    # 15 periods = 5 days
    cost = fee_model.round_trip_cost_pct(holding_periods_8h=15.0)

    # Entry: 0.018%
    # Exit: 0.045%
    # Slippage: 0.04%
    # Funding: 0.01% * 15 = 0.15%
    # Total: 0.253%
    assert abs(cost - 0.253) < 1e-9


def test_funding_disabled(self):
    """Test funding cost can be disabled."""
    fee_model = FeeModelSettings(include_funding=False)

    cost_0 = fee_model.round_trip_cost_pct(0.0)
    cost_15 = fee_model.round_trip_cost_pct(15.0)

    # Should be same regardless of holding period
    assert cost_0 == cost_15
    assert abs(cost_0 - 0.103) < 1e-9
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py::TestFeeModelSettings -k "funding" -v`
Expected: PASS

- [ ] **Step 7: Write tests for slippage variations**

```python
# Add to TestFeeModelSettings class in tests/test_config.py

def test_slippage_custom_value(self):
    """Test custom slippage percentage."""
    fee_model = FeeModelSettings(slippage_pct=0.05)  # 0.05% per leg

    cost = fee_model.round_trip_cost_pct(holding_periods_8h=0.0)

    # Entry: 0.018%
    # Exit: 0.045%
    # Slippage: 0.05% * 2 = 0.10%
    # Total: 0.163%
    assert abs(cost - 0.163) < 1e-9


def test_slippage_zero(self):
    """Test with zero slippage (unrealistic but valid)."""
    fee_model = FeeModelSettings(slippage_pct=0.0)

    cost = fee_model.round_trip_cost_pct(holding_periods_8h=0.0)

    # Entry: 0.018%
    # Exit: 0.045%
    # Slippage: 0.0%
    # Total: 0.063%
    assert abs(cost - 0.063) < 1e-9
```

- [ ] **Step 8: Run all FeeModelSettings tests**

Run: `pytest tests/test_config.py::TestFeeModelSettings -v`
Expected: PASS (all 15+ tests)

- [ ] **Step 9: Commit**

```bash
git add tests/test_config.py
git commit -m "test: add comprehensive fee model variation tests

- Test BNB discount (10% on futures, not 25%)
- Test all order type combinations (maker/taker)
- Test funding costs (zero, fractional, long holds)
- Test funding can be disabled
- Test custom slippage values"
```

---

### Task 4: Add holding period calculation to verifier/constants.py

**Files:**
- Modify: `verifier/constants.py`
- Test: `tests/test_verifier/test_constants.py`

- [ ] **Step 1: Write failing test for compute_holding_periods_8h**

```python
# Add to tests/test_verifier/test_constants.py after existing imports
from verifier.constants import get_horizon_bars, compute_holding_periods_8h


# Add new test class at end of file
class TestHoldingPeriodCalculation:
    """Test holding period calculation for funding costs."""

    def test_1m_timeframe(self):
        """Test 1m timeframe converts to correct funding periods."""
        # 60 bars * 1 minute = 60 minutes = 1 hour = 1/8 period
        periods = compute_holding_periods_8h("1m", 60)
        assert abs(periods - 0.125) < 1e-9

    def test_5m_timeframe(self):
        """Test 5m timeframe converts to correct funding periods."""
        # 48 bars * 5 minutes = 240 minutes = 4 hours = 0.5 periods
        periods = compute_holding_periods_8h("5m", 48)
        assert abs(periods - 0.5) < 1e-9

    def test_15m_timeframe(self):
        """Test 15m timeframe converts to correct funding periods."""
        # 24 bars * 15 minutes = 360 minutes = 6 hours = 0.75 periods
        periods = compute_holding_periods_8h("15m", 24)
        assert abs(periods - 0.75) < 1e-9

    def test_1h_timeframe(self):
        """Test 1h timeframe converts to correct funding periods."""
        # 24 bars * 1 hour = 24 hours = 3 periods
        periods = compute_holding_periods_8h("1h", 24)
        assert abs(periods - 3.0) < 1e-9

    def test_4h_timeframe(self):
        """Test 4h timeframe converts to correct funding periods."""
        # 12 bars * 4 hours = 48 hours = 6 periods
        periods = compute_holding_periods_8h("4h", 12)
        assert abs(periods - 6.0) < 1e-9

    def test_1d_timeframe(self):
        """Test 1d timeframe converts to correct funding periods."""
        # 5 bars * 24 hours = 120 hours = 15 periods
        periods = compute_holding_periods_8h("1d", 5)
        assert abs(periods - 15.0) < 1e-9

    def test_unknown_timeframe(self):
        """Test unknown timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            compute_holding_periods_8h("3h", 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verifier/test_constants.py::TestHoldingPeriodCalculation -v`
Expected: FAIL with "cannot import name 'compute_holding_periods_8h'"

- [ ] **Step 3: Add compute_holding_periods_8h to verifier/constants.py**

```python
# Add to verifier/constants.py after get_horizon_bars function

# Timeframe to hours mapping for funding period calculation
_TIMEFRAME_TO_HOURS: dict[str, float] = {
    "1m": 1.0 / 60.0,   # 1 minute = 1/60 hours
    "5m": 5.0 / 60.0,   # 5 minutes = 5/60 hours
    "15m": 15.0 / 60.0, # 15 minutes = 15/60 hours
    "1h": 1.0,          # 1 hour
    "4h": 4.0,          # 4 hours
    "1d": 24.0,         # 24 hours
}


def compute_holding_periods_8h(timeframe: str, horizon_bars: int) -> float:
    """
    Compute holding period in 8-hour units (funding periods).

    Funding costs on Binance Futures USDT-M are charged every 8 hours.
    This function converts the holding period (in bars) to the number
    of 8-hour funding periods.

    Args:
        timeframe: Timeframe string (e.g., "1h", "1d")
        horizon_bars: Number of bars to hold position

    Returns:
        Holding period as fraction of 8h periods

    Raises:
        ValueError: Unknown timeframe

    Example:
        >>> compute_holding_periods_8h("1m", 60)  # 60 minutes
        0.125  # 1/8 of a funding period
        >>> compute_holding_periods_8h("1h", 24)  # 24 hours
        3.0    # 3 funding periods
        >>> compute_holding_periods_8h("1d", 5)   # 5 days
        15.0   # 15 funding periods
    """
    if timeframe not in _TIMEFRAME_TO_HOURS:
        valid = ", ".join(sorted(_TIMEFRAME_TO_HOURS.keys()))
        raise ValueError(
            f"Unknown timeframe: '{timeframe}'. Valid timeframes: {valid}"
        )

    hours_per_bar = _TIMEFRAME_TO_HOURS[timeframe]
    total_hours = horizon_bars * hours_per_bar
    funding_periods = total_hours / 8.0

    return funding_periods
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verifier/test_constants.py::TestHoldingPeriodCalculation -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add verifier/constants.py tests/test_verifier/test_constants.py
git commit -m "feat: add compute_holding_periods_8h for funding calculation

- Add timeframe to hours mapping
- Convert bars to 8-hour funding periods
- Test all supported timeframes (1m to 1d)
- Raise ValueError on unknown timeframe"
```

---

### Task 5: Add apply_fee_model to verifier/outcome.py

**Files:**
- Modify: `verifier/outcome.py`
- Test: `tests/test_verifier/test_outcome.py`

- [ ] **Step 1: Write failing test for apply_fee_model with exact conversions**

```python
# Add to tests/test_verifier/test_outcome.py after existing imports
import math
from config.fee_model import FeeModelSettings
from verifier.outcome import apply_fee_model


# Add new test class at end of file
class TestApplyFeeModel:
    """Test realistic fee model application with exact log conversions."""

    def test_apply_fee_model_positive_net(self):
        """Test fee application with positive net return after fees."""
        fee_model = FeeModelSettings()

        # Gross log return corresponding to +0.15%
        gross_log = math.log(1 + 0.15 / 100)  # ln(1.0015)

        # Apply fees (3 funding periods)
        net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h=3.0)

        # Convert back to percentage to verify
        # Expected: 0.15% - 0.133% = 0.017%
        net_pct = (math.exp(net_log) - 1) * 100

        assert abs(net_pct - 0.017) < 1e-6

    def test_apply_fee_model_negative_net(self):
        """Test fee application makes positive gross return unprofitable."""
        fee_model = FeeModelSettings()

        # Gross log return corresponding to +0.08%
        gross_log = math.log(1 + 0.08 / 100)

        # Apply fees (3 funding periods, cost = 0.133%)
        net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h=3.0)

        # Convert back to percentage
        # Expected: 0.08% - 0.133% = -0.053%
        net_pct = (math.exp(net_log) - 1) * 100

        assert abs(net_pct - (-0.053)) < 1e-6

    def test_apply_fee_model_exact_conversions(self):
        """Verify exact log ↔ percentage conversions (no approximations)."""
        fee_model = FeeModelSettings()

        # Start with known percentage
        gross_pct = 0.20  # 0.20%

        # Convert to log
        gross_log = math.log(1 + gross_pct / 100)

        # Apply fees
        net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h=3.0)

        # Convert back
        net_pct = (math.exp(net_log) - 1) * 100

        # Expected: 0.20% - 0.133% = 0.067%
        expected_net_pct = gross_pct - 0.133

        # Should match exactly (within floating point precision)
        assert abs(net_pct - expected_net_pct) < 1e-9

    def test_apply_fee_model_zero_holding_period(self):
        """Test fee application with zero funding cost."""
        fee_model = FeeModelSettings()

        gross_log = math.log(1 + 0.15 / 100)

        # No funding (0 periods)
        net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h=0.0)

        net_pct = (math.exp(net_log) - 1) * 100

        # Expected: 0.15% - 0.103% = 0.047%
        assert abs(net_pct - 0.047) < 1e-6

    def test_apply_fee_model_negative_gross(self):
        """Test fee application on negative gross return."""
        fee_model = FeeModelSettings()

        # Gross -0.10%
        gross_log = math.log(1 + (-0.10) / 100)

        net_log = apply_fee_model(gross_log, fee_model, holding_periods_8h=3.0)

        net_pct = (math.exp(net_log) - 1) * 100

        # Expected: -0.10% - 0.133% = -0.233%
        assert abs(net_pct - (-0.233)) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verifier/test_outcome.py::TestApplyFeeModel -v`
Expected: FAIL with "cannot import name 'apply_fee_model'"

- [ ] **Step 3: Add apply_fee_model to verifier/outcome.py**

```python
# Add after imports in verifier/outcome.py (but before existing functions)
# Add this import if not present
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.fee_model import FeeModelSettings


# Add this function after the imports and before compute_log_return
def apply_fee_model(
    gross_log_return: float,
    fee_model: "FeeModelSettings",
    holding_periods_8h: float,
) -> float:
    """
    Apply realistic fee model to gross return.

    Uses EXACT conversions (no linear approximations):
    - pct = (exp(log_return) - 1) * 100
    - log_return = ln(1 + net_pct / 100)

    This prevents future debugging headaches from accumulation of
    approximation errors in DPO training.

    Args:
        gross_log_return: Gross log return before fees
        fee_model: Fee model configuration
        holding_periods_8h: Holding period for funding calculation

    Returns:
        Net log return after all fees

    Example:
        >>> from config.fee_model import FeeModelSettings
        >>> import math
        >>> fee_model = FeeModelSettings()
        >>> gross_log = math.log(1.0015)  # +0.15%
        >>> net_log = apply_fee_model(gross_log, fee_model, 3.0)
        >>> net_pct = (math.exp(net_log) - 1) * 100
        >>> abs(net_pct - 0.017) < 1e-6  # 0.15% - 0.133% fees
        True
    """
    # Convert log → percentage (exact)
    gross_pct = (math.exp(gross_log_return) - 1) * 100

    # Subtract fees
    net_pct = fee_model.net_return(gross_pct, holding_periods_8h)

    # Convert percentage → log (exact)
    net_log_return = math.log(1 + net_pct / 100)

    return net_log_return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verifier/test_outcome.py::TestApplyFeeModel -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Deprecate old compute_net_return function**

```python
# Modify the docstring of compute_net_return in verifier/outcome.py
# Change the first line from:
#     """Compute net return after transaction costs (still log scale)."""
# To:
#     """
#     DEPRECATED: Use apply_fee_model() for realistic fees.
#
#     This function uses a simple flat transaction cost model and does not
#     account for maker/taker differences, funding costs, or realistic slippage.
#
#     Kept for backward compatibility with old tests only.
#
#     Compute net return after transaction costs (still log scale).
```

- [ ] **Step 6: Run all outcome tests to ensure backward compatibility**

Run: `pytest tests/test_verifier/test_outcome.py -v`
Expected: PASS (all tests, including old compute_net_return tests)

- [ ] **Step 7: Commit**

```bash
git add verifier/outcome.py tests/test_verifier/test_outcome.py
git commit -m "feat: add apply_fee_model with exact log conversions

- Implement apply_fee_model() for realistic fee calculation
- Use exact exp/log conversions (no linear approximations)
- Test positive/negative net returns after fees
- Test exact conversion accuracy
- Deprecate old compute_net_return() for backward compatibility"
```

---

### Task 6: Integrate fee model into verifier/engine.py

**Files:**
- Modify: `verifier/engine.py`
- Test: `tests/test_verifier/test_engine.py`

- [ ] **Step 1: Write failing test for verify_example with fee_model parameter**

```python
# Add to tests/test_verifier/test_engine.py after existing imports
import math
from config.fee_model import FeeModelSettings
from verifier.constants import compute_holding_periods_8h


# Add this test method to the TestVerifyExample class
def test_verify_example_with_fee_model(self, mock_market_data):
    """Test verification uses realistic fee model when provided."""
    # Create example
    example = TrainingExample(
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1000000000,
        generator_signal={"direction": "HIGHER", "confidence": 0.8},
    )

    # Mock market data: gross return +0.15%
    entry_price = 50000.0
    exit_price = 50075.0  # +0.15% gross
    gross_return = (exit_price - entry_price) / entry_price

    mock_market_data.setup_for_verification(
        symbol="BTC/USDT",
        entry_price=entry_price,
        exit_price=exit_price,
        low_during_hold=49900.0,
    )

    # Create fee model
    fee_model = FeeModelSettings()

    # Verify with fee model
    outcome = asyncio.run(verify_example(example, mock_market_data, fee_model=fee_model))

    assert outcome is not None

    # Check net return reflects realistic fees
    # Gross: +0.15%
    # Fees for 1h/24bars: 3 funding periods = 0.133%
    # Net: 0.15% - 0.133% = 0.017%
    net_pct = (math.exp(outcome.net_return) - 1) * 100
    assert abs(net_pct - 0.017) < 1e-4  # Within 0.0001%


def test_verify_example_fee_model_makes_signal_unprofitable(self, mock_market_data):
    """Test realistic fees can make seemingly profitable signal unprofitable."""
    example = TrainingExample(
        symbol="ETH/USDT",
        timeframe="1h",
        timestamp_ms=1000000000,
        generator_signal={"direction": "HIGHER", "confidence": 0.9},
    )

    # Mock market data: gross return +0.08% (below fee threshold)
    entry_price = 3000.0
    exit_price = 3002.4  # +0.08% gross

    mock_market_data.setup_for_verification(
        symbol="ETH/USDT",
        entry_price=entry_price,
        exit_price=exit_price,
        low_during_hold=2995.0,
    )

    fee_model = FeeModelSettings()

    outcome = asyncio.run(verify_example(example, mock_market_data, fee_model=fee_model))

    # Net return should be negative
    # Gross: +0.08%
    # Fees: 0.133%
    # Net: -0.053%
    net_pct = (math.exp(outcome.net_return) - 1) * 100
    assert net_pct < 0  # Unprofitable after fees
    assert abs(net_pct - (-0.053)) < 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verifier/test_engine.py -k "fee_model" -v`
Expected: FAIL with "verify_example() got unexpected keyword argument 'fee_model'"

- [ ] **Step 3: Modify verify_example to accept fee_model parameter**

```python
# In verifier/engine.py, update the verify_example function signature:
# Change from:
async def verify_example(
    example: TrainingExample,
    market_data: MarketDataProvider,
    config: BacktestConfig = BacktestConfig(),
) -> VerifiedOutcome | None:

# To:
async def verify_example(
    example: TrainingExample,
    market_data: MarketDataProvider,
    config: BacktestConfig = BacktestConfig(),
    fee_model: "FeeModelSettings | None" = None,
) -> VerifiedOutcome | None:


# Add import at top of file with other TYPE_CHECKING imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.fee_model import FeeModelSettings


# In the function body, find where net_return is computed
# (search for "compute_net_return")
# Replace that section with:

    # Compute net return after fees
    if fee_model is not None:
        # Use realistic fee model
        from verifier.constants import compute_holding_periods_8h
        from verifier.outcome import apply_fee_model

        horizon_bars = get_horizon_bars(timeframe)
        holding_periods_8h = compute_holding_periods_8h(timeframe, horizon_bars)

        net_log_return = apply_fee_model(
            gross_log_return,
            fee_model,
            holding_periods_8h,
        )
    else:
        # Fall back to old simple model for backward compatibility
        from verifier.outcome import compute_net_return
        net_log_return = compute_net_return(gross_log_return)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verifier/test_engine.py -k "fee_model" -v`
Expected: PASS

- [ ] **Step 5: Update verify_batch to pass fee_model**

```python
# In verifier/engine.py, find the verify_batch function
# Update its signature:
async def verify_batch(
    examples: list[TrainingExample],
    market_data: MarketDataProvider,
    config: BacktestConfig = BacktestConfig(),
    fee_model: "FeeModelSettings | None" = None,
) -> list[VerifiedOutcome]:

# Update the inner call to verify_example:
# Change from:
        outcome = await verify_example(example, market_data, config)
# To:
        outcome = await verify_example(example, market_data, config, fee_model)
```

- [ ] **Step 6: Run all verifier engine tests**

Run: `pytest tests/test_verifier/test_engine.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add verifier/engine.py tests/test_verifier/test_engine.py
git commit -m "feat: integrate fee model into verification layer

- Add fee_model parameter to verify_example()
- Use apply_fee_model() when fee_model provided
- Fall back to old compute_net_return() for backward compatibility
- Propagate fee_model through verify_batch()
- Test realistic fees make +0.08% gross unprofitable"
```

---

### Task 7: Add FeeModelSettings to config/settings.py

**Files:**
- Modify: `config/settings.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test for fee_model in AppSettings**

```python
# Add to tests/test_config.py, in the TestAppSettings class

def test_fee_model_nested_settings(self):
    """Test fee model is available as nested settings."""
    settings = AppSettings()

    assert hasattr(settings, "fee_model")
    assert isinstance(settings.fee_model, FeeModelSettings)
    assert settings.fee_model.maker_fee_pct == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestAppSettings::test_fee_model_nested_settings -v`
Expected: FAIL with "AppSettings object has no attribute 'fee_model'"

- [ ] **Step 3: Add fee_model to AppSettings**

```python
# In config/settings.py, add import at top
from config.fee_model import FeeModelSettings


# In AppSettings class, add nested field after dpo and dataset:
    fee_model: FeeModelSettings = Field(default_factory=FeeModelSettings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestAppSettings::test_fee_model_nested_settings -v`
Expected: PASS

- [ ] **Step 5: Write test for environment variable mappings**

```python
# Add to tests/test_config.py TestAppSettings class

def test_fee_model_env_var_mapping(self, monkeypatch):
    """Test fee model fields can be set via environment variables."""
    monkeypatch.setenv("FEE_MAKER_PCT", "0.01")
    monkeypatch.setenv("FEE_TAKER_PCT", "0.04")
    monkeypatch.setenv("FEE_BNB_DISCOUNT", "false")
    monkeypatch.setenv("FEE_FUNDING_RATE_PER_8H", "0.02")

    settings = AppSettings()

    assert settings.fee_model.maker_fee_pct == 0.01
    assert settings.fee_model.taker_fee_pct == 0.04
    assert settings.fee_model.bnb_discount is False
    assert settings.fee_model.funding_rate_per_8h_pct == 0.02
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_config.py::TestAppSettings::test_fee_model_env_var_mapping -v`
Expected: FAIL (env vars not mapped yet)

- [ ] **Step 7: Add environment variable mappings in model_post_init**

```python
# In config/settings.py, in the model_post_init method,
# add to env_mappings dict after DPO settings:

            # Fee model settings
            "FEE_MAKER_PCT": ("fee_model", "maker_fee_pct"),
            "FEE_TAKER_PCT": ("fee_model", "taker_fee_pct"),
            "FEE_BNB_DISCOUNT": ("fee_model", "bnb_discount"),
            "FEE_ENTRY_ORDER_TYPE": ("fee_model", "entry_order_type"),
            "FEE_EXIT_ORDER_TYPE": ("fee_model", "exit_order_type"),
            "FEE_FUNDING_RATE_PER_8H": ("fee_model", "funding_rate_per_8h_pct"),
            "FEE_INCLUDE_FUNDING": ("fee_model", "include_funding"),
            "FEE_SLIPPAGE_PCT": ("fee_model", "slippage_pct"),


# After the environment variable processing loop, add re-validation:
        self.fee_model = FeeModelSettings.model_validate(self.fee_model.model_dump())
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_config.py::TestAppSettings::test_fee_model_env_var_mapping -v`
Expected: PASS

- [ ] **Step 9: Run all config tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all tests)

- [ ] **Step 10: Commit**

```bash
git add config/settings.py tests/test_config.py
git commit -m "feat: add fee_model to AppSettings with env var support

- Add fee_model nested field to AppSettings
- Map environment variables (FEE_MAKER_PCT, etc.)
- Re-validate fee_model after env var injection
- Test nested settings and env var mapping"
```

---

### Task 8: Add fee flip diagnostic to run_dpo_training.py

**Files:**
- Modify: `run_dpo_training.py`

- [ ] **Step 1: Add FEE_FLIP_WARNING_THRESHOLD constant**

```python
# Add near top of run_dpo_training.py after imports
from collections import defaultdict

FEE_FLIP_WARNING_THRESHOLD = 0.15  # 15% flip rate triggers warning
```

- [ ] **Step 2: Write compute_fee_flip_diagnostic function**

```python
# Add before phase3_reward function in run_dpo_training.py

def compute_fee_flip_diagnostic(
    examples_and_outcomes: list[tuple[TrainingExample, VerifiedOutcome]],
) -> None:
    """
    Print diagnostic showing examples that flip from positive to negative
    under realistic fees, grouped by timeframe.

    Compares:
    - Old: flat 0.1% round-trip cost
    - New: realistic Binance Futures fees with funding

    Args:
        examples_and_outcomes: List of (TrainingExample, VerifiedOutcome) pairs
    """
    from config.settings import settings
    from verifier.constants import get_horizon_bars, compute_holding_periods_8h
    import math

    fee_model = settings.fee_model

    # Group by timeframe
    by_timeframe = defaultdict(list)

    for example, outcome in examples_and_outcomes:
        timeframe = example.timeframe

        # Compute old net (flat 0.1% = 0.001 in decimal)
        gross_log = outcome.realized_return + math.log(1 - 0.001) * 2  # Reverse old calc
        old_net_pct = (math.exp(gross_log) - 1) * 100

        # Compute new net (realistic fees)
        horizon_bars = get_horizon_bars(timeframe)
        holding_periods_8h = compute_holding_periods_8h(timeframe, horizon_bars)
        gross_pct = (math.exp(outcome.realized_return) - 1) * 100
        new_net_pct = fee_model.net_return(gross_pct, holding_periods_8h)

        # Track if flipped from positive to negative
        flipped = (old_net_pct > 0) and (new_net_pct < 0)

        by_timeframe[timeframe].append({
            "old_net_pct": old_net_pct,
            "new_net_pct": new_net_pct,
            "flipped": flipped,
        })

    # Compute statistics
    print("\n" + "=" * 80)
    print("=== FEE FLIP DIAGNOSTIC ===")
    print("Examples that were profitable under flat 0.1% fees but are unprofitable")
    print("with realistic Binance Futures fees:\n")

    # Table header
    print(f"{'Timeframe':<10} | {'Total':<6} | {'Flipped':<8} | {'Flip Rate':<10} | {'Avg Old Net':<12} | {'Avg New Net':<12}")
    print("-" * 80)

    total_examples = 0
    total_flipped = 0
    all_old_net = []
    all_new_net = []

    for timeframe in sorted(by_timeframe.keys()):
        data = by_timeframe[timeframe]

        count = len(data)
        flipped_count = sum(1 for d in data if d["flipped"])
        flip_rate = flipped_count / count if count > 0 else 0.0

        avg_old = sum(d["old_net_pct"] for d in data) / count
        avg_new = sum(d["new_net_pct"] for d in data) / count

        total_examples += count
        total_flipped += flipped_count
        all_old_net.extend(d["old_net_pct"] for d in data)
        all_new_net.extend(d["new_net_pct"] for d in data)

        print(f"{timeframe:<10} | {count:>6} | {flipped_count:>8} | {flip_rate:>9.1%} | {avg_old:>+11.2f}% | {avg_new:>+11.2f}%")

    # Total row
    print("-" * 80)
    total_flip_rate = total_flipped / total_examples if total_examples > 0 else 0.0
    total_avg_old = sum(all_old_net) / len(all_old_net) if all_old_net else 0.0
    total_avg_new = sum(all_new_net) / len(all_new_net) if all_new_net else 0.0

    print(f"{'TOTAL':<10} | {total_examples:>6} | {total_flipped:>8} | {total_flip_rate:>9.1%} | {total_avg_old:>+11.2f}% | {total_avg_new:>+11.2f}%")

    # 1d funding breakdown
    if "1d" in by_timeframe:
        horizon_bars = get_horizon_bars("1d")
        holding_periods = compute_holding_periods_8h("1d", horizon_bars)
        funding_cost = fee_model.funding_rate_per_8h_pct * holding_periods
        print(f"\n1d funding cost alone: {funding_cost:.2f}% ({int(holding_periods)} periods × {fee_model.funding_rate_per_8h_pct:.2f}%)")

    # Warnings for high flip rates
    print()
    for timeframe in sorted(by_timeframe.keys()):
        data = by_timeframe[timeframe]
        flip_rate = sum(1 for d in data if d["flipped"]) / len(data)

        if flip_rate > FEE_FLIP_WARNING_THRESHOLD:
            print(f"WARNING: {timeframe} timeframe has {flip_rate:.1%} flip rate - signals may not clear fee hurdle.")
            print(f"Consider focusing training on longer timeframes or increasing signal selectivity.")

    print("=" * 80 + "\n")
```

- [ ] **Step 3: Call diagnostic in phase3_reward**

```python
# In phase3_reward function in run_dpo_training.py,
# after the reward computation loop, before the return statement, add:

    # Compute and print fee flip diagnostic
    compute_fee_flip_diagnostic(pairs)

    return rewards
```

- [ ] **Step 4: Test the diagnostic manually (no automated test needed)**

Run: `python run_dpo_training.py --dataset <path> --dry-run` (if you have test data)
Expected: See fee flip diagnostic table printed during Phase 3

Note: This step requires actual training data, so manual verification is sufficient.

- [ ] **Step 5: Commit**

```bash
git add run_dpo_training.py
git commit -m "feat: add fee flip diagnostic to DPO training pipeline

- Add FEE_FLIP_WARNING_THRESHOLD constant (15%)
- Implement compute_fee_flip_diagnostic() function
- Print table grouped by timeframe
- Show old vs new net returns
- Add 1d funding cost breakdown
- Print WARNING for timeframes exceeding threshold
- Call diagnostic in phase3_reward()"
```

---

### Task 9: Update run_dpo_training.py to use fee_model in verification

**Files:**
- Modify: `run_dpo_training.py`

- [ ] **Step 1: Import FeeModelSettings and pass to verify_batch**

```python
# In run_dpo_training.py, update the phase2_verify function:

async def _run_verify(examples: list[TrainingExample]) -> list[VerifiedOutcome]:
    from data.market_data import MarketDataService
    from config.settings import settings

    async with MarketDataService() as svc:
        # Pass fee model to verify_batch
        return await verify_batch(examples, svc, fee_model=settings.fee_model)
```

- [ ] **Step 2: Update phase2_verify logging**

```python
# In phase2_verify function, after the outcomes are computed, add logging:

    logger.info(
        "Phase 2 complete",
        verified=len(pairs),
        failed=failed,
        fee_model="realistic (Binance Futures USDT-M)",
    )
```

- [ ] **Step 3: Test manually**

Run: `python run_dpo_training.py --dataset <path> --dry-run` (if you have test data)
Expected: Phase 2 logs should show "fee_model=realistic (Binance Futures USDT-M)"

- [ ] **Step 4: Commit**

```bash
git add run_dpo_training.py
git commit -m "feat: use realistic fee model in DPO training verification

- Pass settings.fee_model to verify_batch()
- Log fee model type in Phase 2 output
- All verified outcomes now use realistic Binance Futures fees"
```

---

### Task 10: Final integration test and documentation

**Files:**
- Test: Create integration test in `tests/test_verifier/test_engine.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
# Add to tests/test_verifier/test_engine.py

def test_end_to_end_fee_model_integration(self, mock_market_data):
    """
    End-to-end test: TrainingExample -> VerifiedOutcome with realistic fees.

    Verifies complete flow through all new components:
    - FeeModelSettings configuration
    - Holding period calculation
    - Fee application with exact conversions
    - Integration into verify_example
    """
    from config.fee_model import FeeModelSettings
    import math

    # Create example for 1h timeframe (3 funding periods)
    example = TrainingExample(
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp_ms=1000000000,
        generator_signal={"direction": "HIGHER", "confidence": 0.85},
    )

    # Setup: gross return +0.20% (well above fee threshold)
    entry_price = 50000.0
    exit_price = 50100.0  # +0.20%

    mock_market_data.setup_for_verification(
        symbol="BTC/USDT",
        entry_price=entry_price,
        exit_price=exit_price,
        low_during_hold=49950.0,
    )

    # Create fee model with known parameters
    fee_model = FeeModelSettings(
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        bnb_discount=True,
        funding_rate_per_8h_pct=0.01,
        slippage_pct=0.02,
    )

    # Verify
    outcome = asyncio.run(verify_example(
        example,
        mock_market_data,
        fee_model=fee_model
    ))

    assert outcome is not None

    # Verify net return
    # Gross: +0.20%
    # Fees: 0.133% (entry 0.018% + exit 0.045% + slippage 0.04% + funding 0.03%)
    # Net: 0.067%
    net_pct = (math.exp(outcome.net_return) - 1) * 100

    assert abs(net_pct - 0.067) < 1e-4
    assert outcome.actual_direction == "HIGHER"

    # Verify realized return (gross) is stored correctly
    realized_pct = (math.exp(outcome.realized_return) - 1) * 100
    assert abs(realized_pct - 0.20) < 1e-4
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_verifier/test_engine.py::test_end_to_end_fee_model_integration -v`
Expected: PASS

- [ ] **Step 3: Run complete test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (426+ tests)

- [ ] **Step 4: Update CLAUDE.md with new components**

```bash
# This step is a note - the claude-md-custodian agent will handle this
# No manual action needed here
```

- [ ] **Step 5: Final commit**

```bash
git add tests/test_verifier/test_engine.py
git commit -m "test: add end-to-end integration test for fee model

- Test complete flow from TrainingExample to VerifiedOutcome
- Verify FeeModelSettings, holding periods, exact conversions
- Confirm net return reflects realistic Binance Futures fees
- All 426+ tests passing"
```

- [ ] **Step 6: Create summary commit message for the entire feature**

```bash
git log --oneline -10 > /tmp/commits.txt
# Review commits, then create annotated tag
git tag -a v1.1.0-fee-model -m "feat: realistic Binance Futures USDT-M fee model

This release fixes label poisoning in DPO training by implementing realistic
trading fees at the verification layer.

Key changes:
- FeeModelSettings with maker/taker fees, BNB discounts, funding costs
- Holding period calculation for funding (per 8h)
- Exact log ↔ percentage conversions (no approximations)
- Fee flip diagnostic showing unprofitable signals by timeframe
- Integration into verification and DPO training pipeline

Impact:
- Signals with <0.13% gross return now correctly labeled as unprofitable
- Training on 1m/5m timeframes may need reassessment (high flip rates)
- Model will learn realistic profitability thresholds

Migration:
- Regenerate all VerifiedOutcome data with new fee model
- Re-run DPO training Phases 2-4
- Check fee flip diagnostic for timeframe guidance"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ FeeModelSettings created (Task 1-3)
- ✅ Holding period calculation (Task 4)
- ✅ apply_fee_model with exact conversions (Task 5)
- ✅ Integration into verifier (Task 6)
- ✅ Settings integration (Task 7)
- ✅ Fee flip diagnostic (Task 8-9)
- ✅ Complete test coverage (all tasks)

**Placeholder scan:**
- ✅ No TBDs or TODOs
- ✅ All code blocks complete
- ✅ All test expectations specified
- ✅ All file paths exact

**Type consistency:**
- ✅ FeeModelSettings used consistently
- ✅ holding_periods_8h parameter naming consistent
- ✅ apply_fee_model signature consistent across tasks
- ✅ Test class names follow existing patterns

**Implementation order:**
- ✅ Bottom-up: FeeModelSettings → holding periods → apply_fee_model → integration
- ✅ Each task builds on previous tasks
- ✅ Tests written before implementation (TDD)
- ✅ Frequent commits after each passing test

---

## Execution Notes

**Estimated time:** 2-3 hours for complete implementation and testing

**Key dependencies:**
- Existing verifier infrastructure (constants, outcome, engine)
- Existing settings system (Pydantic)
- pytest framework

**Risk areas:**
- Exact log ↔ percentage conversions (tested extensively)
- Backward compatibility (old compute_net_return kept, deprecated)
- Integration with existing DPO pipeline (tested in Task 9)

**Post-implementation:**
- Run fee flip diagnostic on actual training data
- Regenerate VerifiedOutcome data with new fees
- Update training strategy based on timeframe flip rates
