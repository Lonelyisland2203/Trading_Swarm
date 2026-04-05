# Realistic Fee Model for Binance Futures USDT-M

**Date:** 2026-04-05
**Status:** Approved
**Author:** Claude Code

## Problem Statement

The current reward system computes raw returns without accounting for realistic trading fees. This trains the model on poisoned labels — a signal with +0.08% gross return appears profitable but is actually a loss after Binance Futures USDT-M round-trip fees.

**Current state:**
- Flat 0.1% round-trip cost in `verifier/outcome.py`
- Does not account for maker/taker differences
- Ignores funding costs (critical for futures)
- No BNB discount modeling
- No slippage modeling

**Impact:**
- Model learns to favor unprofitable signals
- DPO training reinforces money-losing behavior
- Real trading would underperform backtest by ~0.1-0.2% per trade

## Solution Overview

Fix fees at the verification layer (single source of truth). Create a realistic fee model for Binance Futures USDT-M that accounts for:
1. Maker/taker fee structure with BNB discounts
2. Funding costs based on holding period
3. Realistic slippage estimates
4. Exact log ↔ percentage conversions (no approximations)

## Architecture

### 1. Fee Model Configuration

**File:** `config/fee_model.py`

```python
class FeeModelSettings(BaseModel):
    """Binance Futures USDT-M fee model."""

    # Exchange configuration
    exchange: str = "binance"
    mode: Literal["futures_usdt"] = "futures_usdt"

    # Fee structure (Binance VIP 0)
    maker_fee_pct: float = 0.02       # 0.02% maker
    taker_fee_pct: float = 0.05       # 0.05% taker
    bnb_discount: bool = True          # 10% discount on futures

    # Order types (realistic execution)
    entry_order_type: Literal["maker", "taker"] = "maker"  # Limit orders for entry
    exit_order_type: Literal["maker", "taker"] = "taker"   # Market orders for exit

    # Funding (USDT-M specific)
    funding_rate_per_8h_pct: float = 0.01  # Average 0.01% per 8h period
    include_funding: bool = True

    # Market impact
    slippage_pct: float = 0.02         # Conservative 0.02% per leg
```

**Methods:**

1. `round_trip_cost_pct(holding_periods_8h: float) -> float`
   - Entry fee: `maker_fee_pct * (0.9 if bnb_discount else 1.0)`
   - Exit fee: `taker_fee_pct * (0.9 if bnb_discount else 1.0)`
   - Funding: `funding_rate_per_8h_pct * holding_periods_8h` (if include_funding)
   - Slippage: `slippage_pct * 2` (both legs)
   - Returns total cost as percentage

2. `net_return(gross_return_pct: float, holding_periods_8h: float) -> float`
   - Subtracts `round_trip_cost_pct()` from gross return
   - Returns net percentage return

3. `minimum_profitable_return_pct(holding_periods_8h: float) -> float`
   - Returns break-even threshold (== `round_trip_cost_pct()`)

**Fee calculation example (maker entry, taker exit, BNB enabled):**
- Entry: 0.02% × 0.9 = 0.018%
- Exit: 0.05% × 0.9 = 0.045%
- Slippage: 0.02% × 2 = 0.04%
- Subtotal: 0.103%
- Funding (3 periods): 0.01% × 3 = 0.03%
- **Total: 0.133%**

### 2. Holding Period Calculation

**File:** `verifier/constants.py`

Add function:
```python
def compute_holding_periods_8h(timeframe: str, horizon_bars: int) -> float:
    """
    Compute holding period in 8-hour units (funding periods).

    Args:
        timeframe: Timeframe string (e.g., "1h", "1d")
        horizon_bars: Number of bars to hold position

    Returns:
        Holding period as fraction of 8h periods

    Examples:
        >>> compute_holding_periods_8h("1m", 60)  # 60 minutes
        0.125  # 1/8 of a funding period
        >>> compute_holding_periods_8h("1h", 24)  # 24 hours
        3.0    # 3 funding periods
        >>> compute_holding_periods_8h("1d", 5)   # 5 days
        15.0   # 15 funding periods
    """
```

**Implementation:**
- Parse timeframe to hours per bar (1m → 1/60, 1h → 1, 1d → 24)
- Total hours = bars × hours_per_bar
- Return total_hours / 8.0

### 3. Verification Layer Integration

**File:** `verifier/outcome.py`

Add new function:
```python
def apply_fee_model(
    gross_log_return: float,
    fee_model: FeeModelSettings,
    holding_periods_8h: float,
) -> float:
    """
    Apply realistic fee model to gross return.

    Uses EXACT conversions (no linear approximations):
    - pct = (exp(log_return) - 1) * 100
    - log_return = ln(1 + net_pct / 100)

    Args:
        gross_log_return: Gross log return before fees
        fee_model: Fee model configuration
        holding_periods_8h: Holding period for funding calculation

    Returns:
        Net log return after all fees
    """
    # Convert log → percentage
    gross_pct = (math.exp(gross_log_return) - 1) * 100

    # Subtract fees
    net_pct = fee_model.net_return(gross_pct, holding_periods_8h)

    # Convert percentage → log
    net_log_return = math.log(1 + net_pct / 100)

    return net_log_return
```

**Deprecation:**
- Keep old `compute_net_return()` for backward compatibility
- Add docstring: "DEPRECATED: Use apply_fee_model() for realistic fees"

**File:** `verifier/engine.py`

Modify `verify_example()`:
```python
async def verify_example(
    example: TrainingExample,
    market_data: MarketDataProvider,
    config: BacktestConfig = BacktestConfig(),
    fee_model: FeeModelSettings = FeeModelSettings(),  # NEW PARAMETER
) -> VerifiedOutcome | None:
    # ... existing code to compute gross_log_return ...

    # Compute holding period for fee calculation
    horizon_bars = get_horizon_bars(timeframe)
    holding_periods_8h = compute_holding_periods_8h(timeframe, horizon_bars)

    # Apply realistic fees
    net_log_return = apply_fee_model(
        gross_log_return,
        fee_model,
        holding_periods_8h,
    )

    # ... rest of function ...
```

### 4. Settings Integration

**File:** `config/settings.py`

Add nested field to `AppSettings`:
```python
class AppSettings(BaseSettings):
    # ... existing fields ...
    fee_model: FeeModelSettings = Field(default_factory=FeeModelSettings)
```

Add environment variable mappings in `model_post_init()`:
- `FEE_MAKER_PCT` → ("fee_model", "maker_fee_pct")
- `FEE_TAKER_PCT` → ("fee_model", "taker_fee_pct")
- `FEE_BNB_DISCOUNT` → ("fee_model", "bnb_discount")
- `FEE_ENTRY_ORDER_TYPE` → ("fee_model", "entry_order_type")
- `FEE_EXIT_ORDER_TYPE` → ("fee_model", "exit_order_type")
- `FEE_FUNDING_RATE_PER_8H` → ("fee_model", "funding_rate_per_8h_pct")
- `FEE_INCLUDE_FUNDING` → ("fee_model", "include_funding")
- `FEE_SLIPPAGE_PCT` → ("fee_model", "slippage_pct")

### 5. Fee Flip Diagnostic

**File:** `run_dpo_training.py` (Phase 3: reward computation)

Add diagnostic output after reward computation:

```python
FEE_FLIP_WARNING_THRESHOLD = 0.15  # 15% flip rate triggers warning

def compute_fee_flip_diagnostic(
    examples_and_outcomes: list[tuple[TrainingExample, VerifiedOutcome]],
    fee_model: FeeModelSettings,
) -> None:
    """
    Print diagnostic showing examples that flip from positive to negative
    under realistic fees, grouped by timeframe.
    """
```

**Output format:**
```
=== FEE FLIP DIAGNOSTIC ===
Examples that were profitable under flat 0.1% fees but are unprofitable with realistic Binance Futures fees:

Timeframe | Total Examples | Flipped to Negative | Flip Rate | Avg Old Net | Avg New Net
----------|----------------|---------------------|-----------|-------------|-------------
1m        |           2250 |                 487 |    21.6%  |      +0.08% |      -0.05%
5m        |           2250 |                 312 |    13.9%  |      +0.09% |      -0.03%
15m       |           2250 |                 156 |     6.9%  |      +0.11% |      +0.01%
1h        |           2250 |                  89 |     4.0%  |      +0.13% |      +0.02%
4h        |           2250 |                  67 |     3.0%  |      +0.14% |      +0.03%
1d        |           2250 |                 123 |     5.5%  |      +0.12% |      -0.01%
----------|----------------|---------------------|-----------|-------------|-------------
TOTAL     |          13500 |                1234 |     9.1%  |      +0.11% |      +0.00%

1d funding cost alone: 0.15% (15 periods × 0.01%)

WARNING: 1m timeframe has 21.6% flip rate - signals may not clear fee hurdle.
Consider focusing training on 15m+ timeframes or increasing signal selectivity.
```

**Logic:**
1. For each example, compute two net returns:
   - Old: `gross - 0.001` (flat 0.1%)
   - New: `gross - fee_model.round_trip_cost_pct(holding_periods_8h)`
2. Track where `old_net > 0` AND `new_net < 0`
3. Group by timeframe, compute statistics
4. Print table to stdout
5. For 1d timeframe: add funding breakdown line
6. Print WARNING for any timeframe exceeding `FEE_FLIP_WARNING_THRESHOLD`

**Interpretation:**
- High flip rate on short timeframes indicates they're below profitability threshold
- Funding breakdown for 1d shows whether funding or base fees dominate
- Guides timeframe prioritization for training

## Testing Strategy

### Unit Tests: `tests/test_config/test_fee_model.py`

1. **Round-trip cost calculation:**
   - Test maker entry + taker exit with BNB discount
   - Test maker entry + maker exit (all limit orders)
   - Test taker entry + taker exit (all market orders)
   - Verify BNB discount is 10% (not 25% - that's spot)
   - Test with BNB disabled

2. **Funding cost calculation:**
   - 0 periods → 0% funding
   - 3 periods → 0.03% funding
   - 15 periods → 0.15% funding
   - Fractional periods (0.125, 2.5)
   - With `include_funding=False` → 0% funding

3. **Slippage:**
   - Verify 2 legs × 0.02% = 0.04% total
   - Test with different slippage values

4. **Net return computation:**
   - Gross +0.08% → verify becomes negative
   - Gross +0.15% → verify becomes positive but reduced
   - Test exact conversions (no approximations)
   - Edge cases: 0%, negative returns

5. **Minimum profitable return:**
   - Verify equals `round_trip_cost_pct()`
   - Different holding periods → different thresholds

### Integration Tests: `tests/test_verifier/`

1. **Holding period calculation:**
   - All timeframes (1m, 5m, 15m, 1h, 4h, 1d)
   - Verify correct conversion to 8h periods

2. **apply_fee_model():**
   - Test exact log ↔ percentage conversions
   - Verify no linear approximations
   - Compare against hand-calculated values

3. **End-to-end verification:**
   - Create TrainingExample with known gross return
   - Run through `verify_example()` with fee model
   - Verify `VerifiedOutcome.net_return` is correct
   - Test all timeframes

### Fee Flip Diagnostic Tests: `tests/test_training/test_dpo_pipeline.py`

1. Test diagnostic computation
2. Test table formatting
3. Test WARNING threshold logic
4. Test 1d funding breakdown

## Migration Path

**Impact:**
- Existing `VerifiedOutcome` data has incorrect `net_return` (computed with 0.1% flat fee)
- Must regenerate all verified outcomes
- Training examples (JSONL) are unaffected (contain gross returns)

**Steps:**
1. Deploy new fee model code
2. Re-run `run_dpo_training.py --dataset <path>` Phases 2-4:
   - Phase 2 (verify): Recomputes `VerifiedOutcome` with realistic fees
   - Phase 3 (reward): Shows fee flip diagnostic
   - Phase 4 (pairs): Generates new preference pairs
3. Discard old preference pairs (computed with wrong fees)
4. Proceed with Phase 5 (train) using new pairs

**Validation:**
- Run fee flip diagnostic (Phase 3)
- Verify signals with <0.13% gross return get negative rewards
- Check mean reward shifts downward
- Ensure reward clipping still works

**Backward compatibility:**
- Keep deprecated `compute_net_return()` for old tests
- Mark with docstring warning

## Success Criteria

1. Fee model correctly computes Binance Futures USDT-M costs:
   - Maker/taker fees with 10% BNB discount
   - Funding costs proportional to holding period
   - Slippage on both legs

2. Exact conversions (no approximations):
   - `pct = (exp(log_return) - 1) * 100`
   - `log_return = ln(1 + net_pct / 100)`

3. Verification layer uses realistic fees:
   - `verify_example()` accepts `fee_model` parameter
   - `VerifiedOutcome.net_return` reflects true costs

4. Fee flip diagnostic prints:
   - Table grouped by timeframe
   - WARNING for timeframes exceeding 15% flip rate
   - 1d funding breakdown

5. All tests pass:
   - Unit tests cover all fee calculations
   - Integration tests verify end-to-end flow
   - Edge cases handled correctly

6. Settings configurable via environment variables

## Non-Goals

- Multiple exchange support (Binance only for now)
- Spot trading fees (futures USDT-M only)
- Dynamic funding rates (using average 0.01%)
- Per-symbol fee tiers (VIP 0 uniform)
- Adaptive slippage modeling (fixed 0.02%)

## Future Enhancements

- Support for different VIP levels (lower fees)
- Dynamic funding rate lookup (historical actual rates)
- Per-symbol slippage calibration
- Exchange-specific fee models (Bybit, OKX)
- Spot market support
