"""
Point-in-time validation for backtesting.

Ensures no lookahead bias in outcome computation through defensive assertions.
"""

from loguru import logger


def validate_no_lookahead(
    signal_timestamp_ms: int,
    entry_timestamp_ms: int,
    forward_data_start_ms: int,
    forward_data_end_ms: int,
) -> None:
    """
    Validate point-in-time correctness - no lookahead bias.
    
    Temporal ordering MUST be:
        signal_timestamp < entry_timestamp <= forward_data_start < forward_data_end
    
    Where:
    - signal_timestamp: When signal was generated (bar close)
    - entry_timestamp: When trade could be executed (next bar open)
    - forward_data_start: First bar used for outcome computation
    - forward_data_end: Last bar used for outcome computation
    
    Args:
        signal_timestamp_ms: Signal generation time
        entry_timestamp_ms: Trade entry time  
        forward_data_start_ms: Start of forward window
        forward_data_end_ms: End of forward window
    
    Raises:
        AssertionError: If temporal ordering is violated (lookahead bias detected)
        
    Example:
        >>> validate_no_lookahead(
        ...     signal_timestamp_ms=1000,
        ...     entry_timestamp_ms=1001,
        ...     forward_data_start_ms=1001,
        ...     forward_data_end_ms=2000
        ... )
        # Passes - correct ordering
        
        >>> validate_no_lookahead(
        ...     signal_timestamp_ms=1000,
        ...     entry_timestamp_ms=999,  # Entry BEFORE signal!
        ...     forward_data_start_ms=1001,
        ...     forward_data_end_ms=2000
        ... )
        AssertionError: Entry must be after signal...
    """
    # Check 1: Entry must be after signal
    assert signal_timestamp_ms < entry_timestamp_ms, (
        f"Entry must be after signal (lookahead bias detected):\n"
        f"  Signal time: {signal_timestamp_ms}\n"
        f"  Entry time:  {entry_timestamp_ms}\n"
        f"  Difference:  {entry_timestamp_ms - signal_timestamp_ms} ms"
    )
    
    # Check 2: Forward data must start at or after entry
    assert entry_timestamp_ms <= forward_data_start_ms, (
        f"Forward data must start at or after entry:\n"
        f"  Entry time:         {entry_timestamp_ms}\n"
        f"  Forward data start: {forward_data_start_ms}\n"
        f"  Difference:         {forward_data_start_ms - entry_timestamp_ms} ms"
    )
    
    # Check 3: Forward data range must be valid
    assert forward_data_start_ms < forward_data_end_ms, (
        f"Forward data range invalid:\n"
        f"  Start: {forward_data_start_ms}\n"
        f"  End:   {forward_data_end_ms}\n"
        f"  Duration: {forward_data_end_ms - forward_data_start_ms} ms"
    )
    
    logger.debug(
        "Point-in-time validation passed",
        signal_ts=signal_timestamp_ms,
        entry_ts=entry_timestamp_ms,
        forward_start=forward_data_start_ms,
        forward_end=forward_data_end_ms,
    )


def validate_forward_data_completeness(
    expected_bars: int,
    actual_bars: int,
    tolerance: int = 1,
) -> bool:
    """
    Validate that forward data contains expected number of bars.
    
    Missing bars indicate data gaps (exchange downtime, API issues, etc.).
    A small tolerance is allowed for bar-count rounding.
    
    Args:
        expected_bars: Expected number of bars in forward window
        actual_bars: Actual number of bars retrieved
        tolerance: Allowed difference (default 1 bar)
    
    Returns:
        True if bar count is within tolerance, False otherwise
        
    Example:
        >>> validate_forward_data_completeness(24, 24)  # Exact match
        True
        >>> validate_forward_data_completeness(24, 23)  # Within tolerance
        True
        >>> validate_forward_data_completeness(24, 20)  # Too many missing
        False
    """
    diff = abs(expected_bars - actual_bars)
    
    if diff > tolerance:
        logger.warning(
            "Forward data incomplete",
            expected=expected_bars,
            actual=actual_bars,
            missing=expected_bars - actual_bars,
        )
        return False
    
    if diff > 0:
        logger.debug(
            "Forward data bar count within tolerance",
            expected=expected_bars,
            actual=actual_bars,
            diff=diff,
        )
    
    return True
