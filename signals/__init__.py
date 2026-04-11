"""
Signals package for production signal loop.

This package contains the core components for generating, evaluating, logging,
and tracking trading signals in production.
"""

from signals.signal_models import (
    SignalDirection,
    Signal,
    SignalLogEntry,
    AccuracyRecord,
    map_generator_to_signal,
)
from signals.preflight import (
    PreflightResult,
    check_stop_file,
    run_preflight_checks,
)
from signals.signal_logger import log_signal
from signals.accuracy_tracker import (
    queue_for_verification,
    process_pending_verifications,
)
from signals.signal_loop import (
    run_cycle,
    run_loop,
    should_override_signal,
)

__all__ = [
    # Models
    "SignalDirection",
    "Signal",
    "SignalLogEntry",
    "AccuracyRecord",
    "map_generator_to_signal",
    # Preflight
    "PreflightResult",
    "check_stop_file",
    "run_preflight_checks",
    # Logging
    "log_signal",
    # Accuracy
    "queue_for_verification",
    "process_pending_verifications",
    # Signal loop
    "run_cycle",
    "run_loop",
    "should_override_signal",
]
