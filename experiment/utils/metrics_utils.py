"""
Minimal privacy metrics helper used by phases and tests.

Extracts relevant accountant/odometer metrics from a model.
"""

from __future__ import annotations

from typing import Any, Dict


def get_privacy_metrics(model: Any) -> Dict[str, Any]:
    """Return a flat dict of privacy metrics from model.odometer if available.

    Supports both a metrics() method and direct attribute access.
    """
    out: Dict[str, Any] = {}
    odo = getattr(model, "odometer", None)
    if odo is None:
        return out

    # Prefer explicit metrics() method if present
    try:
        if hasattr(odo, "metrics") and callable(odo.metrics):
            m = odo.metrics()
            if isinstance(m, dict):
                out.update(m)
    except Exception:
        pass

    # Fall back to common attributes
    for key in (
        "eps_spent",
        "rho_spent",
        "sigma_step",
        "delta_step",
        "m_capacity",
        "deletion_capacity",
        "rho_total",
    ):
        if hasattr(odo, key):
            try:
                out[key] = getattr(odo, key)
            except Exception:
                pass

    # Convenience mirror for downstream naming
    if "rho_spent" in out and "privacy_spend_running" not in out:
        out["privacy_spend_running"] = out["rho_spent"]

    return out

