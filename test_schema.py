"""
Tests for shared event schema functionality.
"""

import os
import sys
import numpy as np

# Add path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

from data_loader.event_schema import (
    create_event_record,
    parse_event_record,
    legacy_loader_adapter,
    validate_event_record
)


def test_event_record_creation():
    """Test creating event records."""
    print("Testing event record creation...")
    
    x = np.array([1.0, 2.0, 3.0])
    y = 0.5
    sample_id = "test_001"
    event_id = 42
    
    record = create_event_record(x, y, sample_id, event_id)
    
    # Check required keys
    assert "x" in record
    assert "y" in record
    assert "sample_id" in record
    assert "event_id" in record
    assert "segment_id" in record
    assert "metrics" in record
    
    # Check values
    assert np.array_equal(record["x"], x)
    assert record["y"] == y
    assert record["sample_id"] == sample_id
    assert record["event_id"] == event_id
    assert record["segment_id"] == 0  # default
    
    # Check x_norm is computed
    assert "x_norm" in record["metrics"]
    assert abs(record["metrics"]["x_norm"] - np.linalg.norm(x)) < 1e-10
    
    print("✓ Event record creation works")


def test_parse_event_record():
    """Test parsing event records."""
    print("Testing event record parsing...")
    
    x = np.array([1.0, 2.0, 3.0])
    y = 0.5
    sample_id = "test_001"
    event_id = 42
    
    record = create_event_record(x, y, sample_id, event_id)
    
    parsed_x, parsed_y, meta = parse_event_record(record)
    
    assert np.array_equal(parsed_x, x)
    assert parsed_y == y
    assert meta["sample_id"] == sample_id
    assert meta["event_id"] == event_id
    assert meta["segment_id"] == 0
    assert "metrics" in meta
    
    print("✓ Event record parsing works")


def test_legacy_adapter():
    """Test legacy compatibility adapter."""
    print("Testing legacy adapter...")
    
    x = np.array([1.0, 2.0, 3.0])
    y = 0.5
    sample_id = "test_001"
    event_id = 42
    
    record = create_event_record(x, y, sample_id, event_id)
    
    legacy_x, legacy_y = legacy_loader_adapter(record)
    
    assert np.array_equal(legacy_x, x)
    assert legacy_y == y
    
    print("✓ Legacy adapter works")


def test_validation():
    """Test record validation."""
    print("Testing record validation...")
    
    # Valid record
    valid_record = {
        "x": np.array([1.0, 2.0]),
        "y": 0.5,
        "sample_id": "test",
        "event_id": 1,
        "segment_id": 0,
        "metrics": {"x_norm": 2.236}
    }
    assert validate_event_record(valid_record)
    
    # Invalid - missing key
    invalid_record = {
        "x": np.array([1.0, 2.0]),
        "y": 0.5,
        "sample_id": "test",
        # missing event_id
        "segment_id": 0,
        "metrics": {}
    }
    assert not validate_event_record(invalid_record)
    
    # Invalid - wrong type
    invalid_record2 = {
        "x": np.array([1.0, 2.0]),
        "y": 0.5,
        "sample_id": 123,  # should be string
        "event_id": 1,
        "segment_id": 0,
        "metrics": {}
    }
    assert not validate_event_record(invalid_record2)
    
    print("✓ Record validation works")


def test_loader_schema_integration():
    """Test that loaders return valid schemas when enabled."""
    print("Testing loader schema integration...")
    
    from data_loader import get_synthetic_linear_stream
    
    # Test with schema enabled
    stream = get_synthetic_linear_stream(dim=5, seed=42, use_event_schema=True)
    
    # Get first few events
    events = []
    for i, record in enumerate(stream):
        if i >= 3:
            break
        events.append(record)
    
    # Validate all records
    for record in events:
        assert validate_event_record(record), f"Invalid record: {record}"
        
        # Check x, y are reasonable
        assert hasattr(record["x"], "shape")
        assert record["x"].shape[0] == 5  # dimension
        assert isinstance(record["y"], (int, float))
        
        # Check metadata
        assert isinstance(record["sample_id"], str)
        assert isinstance(record["event_id"], int)
        assert record["event_id"] >= 0
        
        # Check metrics
        assert "x_norm" in record["metrics"]
        assert record["metrics"]["x_norm"] is not None
    
    # Test with schema disabled (legacy mode)
    legacy_stream = get_synthetic_linear_stream(dim=5, seed=42, use_event_schema=False)
    
    # Should return (x, y) tuples
    x, y = next(legacy_stream)
    assert hasattr(x, "shape")
    assert x.shape[0] == 5
    assert isinstance(y, (int, float))
    
    print("✓ Loader schema integration works")


if __name__ == "__main__":
    print("Running schema tests...")
    test_event_record_creation()
    test_parse_event_record()
    test_legacy_adapter()
    test_validation()
    test_loader_schema_integration()
    print("✓ All schema tests passed!")