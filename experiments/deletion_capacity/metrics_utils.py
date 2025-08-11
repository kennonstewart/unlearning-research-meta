"""
Utility functions for metrics computation and aggregation.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def mean_ci(values: List[float]) -> Tuple[float, float]:
    """Compute mean and 95% confidence interval."""
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    ci = float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, ci


def get_privacy_metrics(model) -> Dict[str, Any]:
    """Extract privacy metrics from model's odometer and calibration statistics."""
    odometer = getattr(model, "odometer", None)
    metrics = {}
    
    # Add calibration statistics if available
    if hasattr(model, "calibration_stats") and model.calibration_stats:
        stats = model.calibration_stats
        metrics.update({
            "G_hat": stats.get("G"),
            "D_hat": stats.get("D"),
            "c_hat": stats.get("c"),
            "C_hat": stats.get("C"),
            "N_star_theory": stats.get("N_star"),
        })
    
    # Add theoretical metrics from odometer if available
    if odometer:
        if hasattr(odometer, "deletion_capacity"):
            metrics["m_theory"] = odometer.deletion_capacity
        if hasattr(odometer, "sigma_step"):
            metrics["sigma_step_theory"] = odometer.sigma_step
    
    if odometer is None:
        return metrics
    
    # Try to get metrics from new accountant strategy interface first
    if hasattr(odometer, "metrics"):
        try:
            strategy_metrics = odometer.metrics()
            metrics.update(strategy_metrics)
            return metrics
        except Exception:
            pass  # Fallback to legacy method
    
    # Legacy fallback: extract metrics directly from odometer attributes
    # Common metrics for both accountant types
    if hasattr(odometer, "eps_spent"):
        metrics["eps_spent"] = odometer.eps_spent
    if hasattr(odometer, "delta_spent"):
        metrics["delta_spent"] = odometer.delta_spent
    
    # RDP-specific metrics
    if hasattr(odometer, "eps_converted"):
        metrics["eps_converted"] = odometer.eps_converted
    if hasattr(odometer, "delta_total"):
        metrics["delta_total"] = odometer.delta_total
    if hasattr(odometer, "eps_remaining"):
        metrics["eps_remaining"] = odometer.eps_remaining
    if hasattr(odometer, "sens_count"):
        metrics["sens_count"] = odometer.sens_count
    if hasattr(odometer, "sens_q95"):
        metrics["sens_q95"] = odometer.sens_q95
    if hasattr(odometer, "recalibrations_count"):
        metrics["recalibrations_count"] = odometer.recalibrations_count
    if hasattr(odometer, "m_current"):
        metrics["m_current"] = odometer.m_current
    if hasattr(odometer, "sigma_current"):
        metrics["sigma_current"] = odometer.sigma_current
    
    # Legacy metrics
    if hasattr(odometer, "eps_step"):
        metrics["eps_step_theory"] = odometer.eps_step
    if hasattr(odometer, "delta_step"):
        metrics["delta_step_theory"] = odometer.delta_step
    
    # Add accountant type if not already present
    if "accountant_type" not in metrics:
        if hasattr(odometer, "accountant_type"):
            metrics["accountant_type"] = odometer.accountant_type
        else:
            metrics["accountant_type"] = "unknown"
    
    return metrics


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate summary statistics across seeds."""
    if not summaries:
        return {}
    
    # Get all unique keys from summaries
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())
    
    aggregated = {}
    
    # Add all numeric keys
    for key in sorted(all_keys):
        if key in ["accountant_type"]:  # Skip non-numeric keys
            continue
        
        values = [
            s.get(key)
            for s in summaries
            if s.get(key) is not None and isinstance(s.get(key), (int, float))
        ]
        
        if values:
            mean, ci = mean_ci(values)
            aggregated[f"{key}_mean"] = mean
            aggregated[f"{key}_ci95"] = ci
    
    return aggregated