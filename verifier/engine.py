"""
Backtesting engine - main API for verification.

Computes realized outcomes from training examples with point-in-time safety.
"""

from itertools import groupby
from typing import Protocol

import pandas as pd
from loguru import logger

from data.market_data import MarketDataService
from swarm.training_capture import TrainingExample
from .config import BacktestConfig
from .constants import get_horizon_bars
from .outcome import (
    VerifiedOutcome,
    compute_log_return,
    compute_mae,
    compute_net_return,
    determine_direction,
)
from .validator import validate_forward_data_completeness, validate_no_lookahead


class MarketDataProvider(Protocol):
    """Protocol for market data access (allows mocking in tests)."""
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
    ) -> pd.DataFrame:
        ...
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        ...


async def verify_example(
    example: TrainingExample,
    market_data: MarketDataProvider,
    config: BacktestConfig = BacktestConfig(),
) -> VerifiedOutcome | None:
    """
    Verify a single training example - compute realized outcome.
    
    Point-in-time safety:
    1. Signal was generated at example.timestamp_ms (bar close)
    2. Entry is at next bar open (or close if config.entry_on="close" for testing)
    3. Forward data starts AFTER entry
    4. Outcome measured over timeframe-adaptive horizon
    
    Args:
        example: Training example to verify
        market_data: Market data provider
        config: Backtest configuration
    
    Returns:
        VerifiedOutcome if data available, None if insufficient data
        
    Example:
        async with MarketDataService() as market_data:
            outcome = await verify_example(example, market_data)
            if outcome:
                print(f"Realized: {outcome.actual_direction}, Return: {outcome.realized_return:.4f}")
    """
    symbol = example.symbol
    timeframe = example.timeframe
    signal_timestamp_ms = example.timestamp_ms
    
    # Get horizon for this timeframe
    try:
        horizon_bars = get_horizon_bars(timeframe)
    except ValueError as e:
        logger.error("Unknown timeframe", timeframe=timeframe, error=str(e))
        return None
    
    # Calculate bar duration
    bar_duration_ms = market_data._timeframe_to_ms(timeframe)
    
    # Determine entry time
    if config.entry_on == "next_open":
        # Realistic: next bar open (first ms after signal bar closes)
        entry_timestamp_ms = signal_timestamp_ms + 1
    else:  # "close" - for testing only
        entry_timestamp_ms = signal_timestamp_ms
    
    # Fetch forward data
    # Need enough bars: entry bar + horizon bars
    # Add buffer of 2 extra bars for safety
    total_bars_needed = horizon_bars + 3
    
    try:
        # Fetch OHLCV starting from just before entry time
        # We need to get bars that include the entry bar
        all_data = await market_data.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            lookback_bars=total_bars_needed,
        )
        
        # Filter to forward data only (after signal)
        forward_data = all_data[all_data["timestamp"] > signal_timestamp_ms].copy()
        
        if forward_data.empty:
            logger.warning(
                "No forward data available",
                symbol=symbol,
                timeframe=timeframe,
                signal_ts=signal_timestamp_ms,
            )
            return None
        
        # Validate we have enough data
        if len(forward_data) < horizon_bars:
            logger.warning(
                "Insufficient forward data",
                symbol=symbol,
                expected=horizon_bars,
                actual=len(forward_data),
            )
            return None
        
        # Take exactly horizon_bars for measurement window
        holding_period = forward_data.iloc[:horizon_bars].copy()
        
        # Get entry price
        entry_bar = holding_period.iloc[0]
        if config.entry_on == "next_open":
            entry_price = entry_bar["open"]
        else:  # "close"
            # For testing: use signal bar close (requires looking back)
            signal_bar = all_data[all_data["timestamp"] == signal_timestamp_ms]
            if signal_bar.empty:
                logger.error("Signal bar not found", timestamp=signal_timestamp_ms)
                return None
            entry_price = signal_bar.iloc[0]["close"]
        
        # Get exit price (close of last bar in holding period)
        exit_price = holding_period.iloc[-1]["close"]
        
        # Point-in-time validation
        validate_no_lookahead(
            signal_timestamp_ms=signal_timestamp_ms,
            entry_timestamp_ms=entry_timestamp_ms,
            forward_data_start_ms=int(holding_period["timestamp"].iloc[0]),
            forward_data_end_ms=int(holding_period["timestamp"].iloc[-1]),
        )
        
        # Validate data completeness
        validate_forward_data_completeness(
            expected_bars=horizon_bars,
            actual_bars=len(holding_period),
            tolerance=1,
        )
        
        # Compute log return
        log_return = compute_log_return(entry_price, exit_price)
        
        # Compute MAE
        predicted_direction = example.generator_signal.get("direction", "UNKNOWN")
        if predicted_direction in ("HIGHER", "LOWER"):
            mae = compute_mae(holding_period, predicted_direction, entry_price)
        else:
            logger.warning(
                "Cannot compute MAE for unknown direction",
                direction=predicted_direction,
            )
            mae = 0.0
        
        # Compute net return after transaction costs
        net_return = compute_net_return(
            log_return,
            txn_cost_pct=config.txn_cost_pct,
            num_trades=2,
        )
        
        # Determine actual direction
        actual_direction = determine_direction(log_return)
        
        logger.debug(
            "Outcome computed",
            symbol=symbol,
            actual_direction=actual_direction,
            log_return=log_return,
            mae=mae,
            net_return=net_return,
        )
        
        return VerifiedOutcome(
            example_id=example.example_id,
            actual_direction=actual_direction,
            realized_return=log_return,
            max_adverse_excursion=mae,
            net_return=net_return,
            entry_price=entry_price,
            exit_price=exit_price,
            bars_held=len(holding_period),
        )
        
    except Exception as e:
        logger.error(
            "Verification failed",
            symbol=symbol,
            timeframe=timeframe,
            error=str(e),
            example_id=example.example_id,
        )
        return None


async def verify_batch(
    examples: list[TrainingExample],
    market_data: MarketDataProvider,
    config: BacktestConfig = BacktestConfig(),
    batch_size: int = 100,
) -> list[VerifiedOutcome]:
    """
    Verify batch of training examples with efficient grouping.
    
    Groups examples by (symbol, timeframe) to reduce redundant data fetches.
    Processes in batches to control memory usage.
    
    Args:
        examples: List of training examples to verify
        market_data: Market data provider
        config: Backtest configuration
        batch_size: Maximum examples per batch (default 100)
    
    Returns:
        List of verified outcomes (excludes examples with insufficient data)
        
    Example:
        examples = load_training_examples(Path("outputs/training"))
        outcomes = await verify_batch(examples, market_data)
        print(f"Verified {len(outcomes)} of {len(examples)} examples")
    """
    if not examples:
        logger.info("No examples to verify")
        return []
    
    logger.info(
        "Starting batch verification",
        total_examples=len(examples),
        batch_size=batch_size,
    )
    
    results: list[VerifiedOutcome] = []
    failed = 0
    
    # Group by (symbol, timeframe) for efficient processing
    sorted_examples = sorted(examples, key=lambda e: (e.symbol, e.timeframe))
    
    for (symbol, timeframe), group in groupby(
        sorted_examples, key=lambda e: (e.symbol, e.timeframe)
    ):
        group_list = list(group)
        logger.debug(
            "Processing group",
            symbol=symbol,
            timeframe=timeframe,
            count=len(group_list),
        )
        
        # Process in batches to control memory
        for i in range(0, len(group_list), batch_size):
            batch = group_list[i : i + batch_size]
            
            # Verify each example in batch
            for example in batch:
                outcome = await verify_example(example, market_data, config)
                if outcome:
                    results.append(outcome)
                else:
                    failed += 1
    
    logger.info(
        "Batch verification complete",
        verified=len(results),
        failed=failed,
        total=len(examples),
        success_rate=f"{len(results) / len(examples) * 100:.1f}%",
    )
    
    return results
