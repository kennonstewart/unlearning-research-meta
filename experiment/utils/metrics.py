"""
Minimal metrics used by phases/tests.
"""

from __future__ import annotations


def abs_error(pred: float, y: float) -> float:
    try:
        return float(abs(float(pred) - float(y)))
    except Exception:
        return float("nan")

