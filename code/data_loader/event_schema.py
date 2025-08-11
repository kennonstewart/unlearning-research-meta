"""
Shared event schema for unified data loading and processing.

This module defines the canonical event format that all loaders should emit
and provides utilities for parsing and working with event records.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


def create_event_record(
    x: np.ndarray,
    y: Union[float, int],
    sample_id: str,
    event_id: int,
    segment_id: int = 0,
    metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a canonical event record.
    
    Args:
        x: Input feature vector
        y: Target value
        sample_id: Stable string identifier for this sample
        event_id: Monotonic integer event identifier
        segment_id: Segment identifier (default 0)
        metrics: Optional metrics dict (x_norm will be computed if not provided)
    
    Returns:
        Canonical event record dictionary
    """
    if metrics is None:
        metrics = {}
    
    # Compute x_norm if not provided
    if "x_norm" not in metrics:
        try:
            metrics["x_norm"] = float(np.linalg.norm(x))
        except Exception:
            metrics["x_norm"] = None
    
    return {
        "x": x,
        "y": y,
        "sample_id": sample_id,
        "event_id": event_id,
        "segment_id": segment_id,
        "metrics": metrics
    }


def create_event_record_with_diagnostics(
    x: np.ndarray,
    y: Union[float, int],
    sample_id: str,
    event_id: int,
    segment_id: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    # M9 diagnostics
    gamma_bar: Optional[float] = None,
    gamma_split: Optional[float] = None,
    gamma_ins: Optional[float] = None,
    gamma_del: Optional[float] = None,
    N_star_live: Optional[int] = None,
    m_theory_live: Optional[int] = None,
    blocked_reason: Optional[str] = None,
    # M7 diagnostics (CovType)
    mean_l2: Optional[float] = None,
    std_l2: Optional[float] = None,
    clip_rate: Optional[float] = None,
    # M8 diagnostics (Linear)
    lambda_est: Optional[float] = None,
    P_T_true: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create event record with extended diagnostics for M7-M10.
    
    All diagnostic parameters are optional and will only be included
    if they are not None.
    """
    if metrics is None:
        metrics = {}
    
    # Compute x_norm if not provided
    if "x_norm" not in metrics:
        try:
            metrics["x_norm"] = float(np.linalg.norm(x))
        except Exception:
            metrics["x_norm"] = None
    
    # Add diagnostic fields if provided
    diagnostics = {}
    if gamma_bar is not None:
        diagnostics["gamma_bar"] = gamma_bar
    if gamma_split is not None:
        diagnostics["gamma_split"] = gamma_split
    if gamma_ins is not None:
        diagnostics["gamma_ins"] = gamma_ins
    if gamma_del is not None:
        diagnostics["gamma_del"] = gamma_del
    if N_star_live is not None:
        diagnostics["N_star_live"] = N_star_live
    if m_theory_live is not None:
        diagnostics["m_theory_live"] = m_theory_live
    if blocked_reason is not None:
        diagnostics["blocked_reason"] = blocked_reason
    if mean_l2 is not None:
        diagnostics["mean_l2"] = mean_l2
    if std_l2 is not None:
        diagnostics["std_l2"] = std_l2
    if clip_rate is not None:
        diagnostics["clip_rate"] = clip_rate
    if lambda_est is not None:
        diagnostics["lambda_est"] = lambda_est
    if P_T_true is not None:
        diagnostics["P_T_true"] = P_T_true
        
    # Merge diagnostics into metrics
    metrics.update(diagnostics)
    
    return {
        "x": x,
        "y": y,
        "sample_id": sample_id,
        "event_id": event_id,
        "segment_id": segment_id,
        "metrics": metrics
    }


def parse_event_record(record: Dict[str, Any]) -> Tuple[np.ndarray, Union[float, int], Dict[str, Any]]:
    """
    Parse an event record into (x, y, meta) format.
    
    Args:
        record: Event record dictionary
    
    Returns:
        Tuple of (x, y, metadata) where metadata contains sample_id, event_id, 
        segment_id, and metrics
    """
    x = record["x"]
    y = record["y"]
    
    meta = {
        "sample_id": record["sample_id"],
        "event_id": record["event_id"],
        "segment_id": record["segment_id"],
        "metrics": record["metrics"]
    }
    
    return x, y, meta


def legacy_loader_adapter(event_record: Dict[str, Any]) -> Tuple[np.ndarray, Union[float, int]]:
    """
    Backward compatibility adapter for legacy code expecting (x, y) tuples.
    
    Args:
        event_record: Modern event record dictionary
    
    Returns:
        Legacy (x, y) tuple
    """
    return event_record["x"], event_record["y"]


def validate_event_record(record: Dict[str, Any]) -> bool:
    """
    Validate that a record conforms to the expected event schema.
    
    Args:
        record: Event record to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = {"x", "y", "sample_id", "event_id", "segment_id", "metrics"}
    
    if not isinstance(record, dict):
        return False
    
    if not all(key in record for key in required_keys):
        return False
    
    # Check types
    if not isinstance(record["sample_id"], str):
        return False
    
    if not isinstance(record["event_id"], int):
        return False
    
    if not isinstance(record["segment_id"], int):
        return False
    
    if not isinstance(record["metrics"], dict):
        return False
    
    return True