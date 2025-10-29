"""
DEPRECATED: This module is replaced by experiment/run.py for experiment execution.

This stub maintains backward compatibility for code that imports ExperimentRunner,
ALGO_MAP, or _get_data_stream. For new experiments, use experiment/run.py instead.
"""

import os
import sys
from typing import Dict

# Ensure code directory is on path
_EXP_DIR = os.path.dirname(__file__)
_CODE_DIR = os.path.abspath(os.path.join(_EXP_DIR, "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from memory_pair.src.memory_pair import MemoryPair
from data_loader import get_rotating_linear_stream

# Algorithm mapping - for backward compatibility
ALGO_MAP = {"memorypair": MemoryPair}


def _get_data_stream(cfg, seed: int):
    """Get synthetic data stream (rotating linear stream only).
    
    DEPRECATED: Use get_theory_stream or get_rotating_linear_stream directly.
    """
    return get_rotating_linear_stream(
        seed=seed,
        rotate_angle=getattr(cfg, "rotate_angle", 0.01),
        G_hat=getattr(cfg, "G_hat", None),
        D_hat=getattr(cfg, "D_hat", None),
        c_hat=getattr(cfg, "c_hat", None),
        C_hat=getattr(cfg, "C_hat", None),
    )


# ExperimentRunner stub for backward compatibility
class ExperimentRunner:
    """
    DEPRECATED: Use experiment/run.py for running experiments.
    
    This stub is provided for backward compatibility only.
    """
    
    def __init__(self, cfg):
        import warnings
        warnings.warn(
            "ExperimentRunner is deprecated. Use experiment/run.py for running experiments.",
            DeprecationWarning,
            stacklevel=2
        )
        self.cfg = cfg
    
    def run_all(self):
        raise NotImplementedError(
            "ExperimentRunner.run_all() is deprecated. "
            "Use experiment/run.py for running experiments."
        )
    
    def run_one_seed(self, seed: int):
        raise NotImplementedError(
            "ExperimentRunner.run_one_seed() is deprecated. "
            "Use experiment/run.py for running experiments."
        )


__all__ = [
    "ExperimentRunner",
    "ALGO_MAP",
    "_get_data_stream",
]

# Deprecation notice
import warnings
warnings.warn(
    "runner.py is deprecated. Use experiment/run.py for running experiments.",
    DeprecationWarning,
    stacklevel=2
)
