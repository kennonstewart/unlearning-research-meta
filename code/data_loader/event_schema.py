"""
Shared event schema for unified data loading and processing.

This module defines the canonical event format that all loaders should emit
and provides utilities for parsing and working with event records.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np


def create_event_record(
    x: np.ndarray,
    y: Union[float, int],
    sample_id: str,
    event_id: int,
    segment_id: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    privacy_info: Optional[Dict[str, Any]] = None
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
        privacy_info: Optional privacy accounting information for deletions
    
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
    
    record = {
        "x": x,
        "y": y,
        "sample_id": sample_id,
        "event_id": event_id,
        "segment_id": segment_id,
        "metrics": metrics
    }
    
    # Add privacy information if provided (for deletion events)
    if privacy_info is not None:
        record["privacy_info"] = privacy_info
    
    return record


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
    
    # Privacy info is optional but must be dict if present
    if "privacy_info" in record and not isinstance(record["privacy_info"], dict):
        return False
    
    return True


def create_deletion_privacy_info(
    accountant_type: str,
    sensitivity: float,
    noise_scale: float,
    budget_spent: Dict[str, float],
    remaining_capacity: int,
    gradient_magnitude: Optional[float] = None,
    pathwise_drift: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create privacy information record for deletion events.
    
    This function creates a standardized privacy information record that
    captures all relevant details about the privacy accounting for a deletion.
    
    Args:
        accountant_type: Type of privacy accountant used ('zCDP', 'eps_delta', 'relaxed')
        sensitivity: L2 sensitivity of the deletion operation
        noise_scale: Standard deviation of Gaussian noise applied
        budget_spent: Dictionary of budget spent (eps_spent, rho_spent, etc.)
        remaining_capacity: Number of deletions remaining after this operation
        gradient_magnitude: Magnitude of gradient for this deletion (optional)
        pathwise_drift: Pathwise drift statistic from comparator (optional)
    
    Returns:
        Privacy information dictionary for inclusion in event records
    """
    privacy_info = {
        "accountant_type": accountant_type,
        "sensitivity": sensitivity,
        "noise_scale": noise_scale,
        "budget_spent": budget_spent,
        "remaining_capacity": remaining_capacity,
        "deletion_timestamp": None  # Can be filled by caller
    }
    
    # Add optional metrics if provided
    if gradient_magnitude is not None:
        privacy_info["gradient_magnitude"] = gradient_magnitude
    
    if pathwise_drift is not None:
        privacy_info["pathwise_drift"] = pathwise_drift
    
    return privacy_info


def extract_privacy_metrics(event_records: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Extract privacy metrics from a list of event records for analysis.
    
    Args:
        event_records: List of event records potentially containing privacy info
    
    Returns:
        Dictionary with lists of privacy metrics suitable for plotting/analysis
    """
    metrics = {
        "accountant_types": [],
        "sensitivities": [],
        "noise_scales": [],
        "budget_spent_eps": [],
        "budget_spent_rho": [],
        "remaining_capacities": [],
        "gradient_magnitudes": [],
        "pathwise_drifts": []
    }
    
    for record in event_records:
        privacy_info = record.get("privacy_info")
        if privacy_info is None:
            continue
        
        metrics["accountant_types"].append(privacy_info.get("accountant_type", "unknown"))
        metrics["sensitivities"].append(privacy_info.get("sensitivity", 0.0))
        metrics["noise_scales"].append(privacy_info.get("noise_scale", 0.0))
        
        budget_spent = privacy_info.get("budget_spent", {})
        metrics["budget_spent_eps"].append(budget_spent.get("eps_spent", 0.0))
        metrics["budget_spent_rho"].append(budget_spent.get("rho_spent", 0.0))
        
        metrics["remaining_capacities"].append(privacy_info.get("remaining_capacity", 0))
        metrics["gradient_magnitudes"].append(privacy_info.get("gradient_magnitude"))
        metrics["pathwise_drifts"].append(privacy_info.get("pathwise_drift"))
    
    return metrics