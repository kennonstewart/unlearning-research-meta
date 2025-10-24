"""
DEPRECATED: This module is replaced by experiment/run.py for experiment execution.

This stub maintains backward compatibility for tests that import CSV processing helpers.
For new experiments, use experiment/run.py instead.
"""

# Import legacy CSV helpers for backward compatibility with existing tests
from experiment.legacy_csv_helpers import (
    process_seed_output,
    process_event_output,
    process_aggregate_output,
)

__all__ = [
    "process_seed_output",
    "process_event_output", 
    "process_aggregate_output",
]

# Deprecation notice
import warnings
warnings.warn(
    "grid_runner.py is deprecated. Use experiment/run.py for running experiments. "
    "CSV processing helpers have been moved to experiment/legacy_csv_helpers.py",
    DeprecationWarning,
    stacklevel=2
)
