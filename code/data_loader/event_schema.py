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