"""
Test that feature flags default to no-op behavior.
Verifies that with all flags False, existing functionality works identically.
"""

import os
import sys
import tempfile
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity"))

from data_loader import get_synthetic_linear_stream, parse_event_record, legacy_loader_adapter
from config import Config


def test_flags_default_false():
    """Test that all feature flags default to False."""
    print("Testing feature flags default to False...")
    
    cfg = Config()
    
    # Check all new feature flags are False by default
    assert cfg.adaptive_geometry == False
    assert cfg.dynamic_comparator == False
    assert cfg.strong_convexity == False
    assert cfg.adaptive_privacy == False
    assert cfg.drift_mode == False
    assert cfg.window_erm == False
    assert cfg.online_standardize == False
    
    print("✓ All feature flags default to False")


def test_legacy_behavior_preserved():
    """Test that legacy loaders still work for backward compatibility."""
    print("Testing legacy behavior preserved...")
    
    # Get legacy stream (old behavior)
    legacy_stream = get_synthetic_linear_stream(dim=5, seed=42, use_event_schema=False)
    
    # Should return (x, y) tuples
    x1, y1 = next(legacy_stream)
    x2, y2 = next(legacy_stream)
    
    assert hasattr(x1, "shape")
    assert x1.shape[0] == 5
    assert isinstance(y1, (int, float))
    assert hasattr(x2, "shape")
    assert x2.shape[0] == 5
    assert isinstance(y2, (int, float))
    
    # Data should be different between calls
    assert not np.array_equal(x1, x2)
    
    print("✓ Legacy behavior preserved")


def test_event_schema_roundtrip():
    """Test that event schema preserves data integrity."""
    print("Testing event schema roundtrip...")
    
    # Get new stream (event records)
    event_stream = get_synthetic_linear_stream(dim=5, seed=42, use_event_schema=True)
    
    # Get legacy stream (for comparison)
    legacy_stream = get_synthetic_linear_stream(dim=5, seed=42, use_event_schema=False)
    
    # Compare several events
    for i in range(5):
        record = next(event_stream)
        legacy_x, legacy_y = next(legacy_stream)
        
        # Extract from event record
        event_x, event_y, meta = parse_event_record(record)
        
        # Should be identical data
        assert np.array_equal(event_x, legacy_x), f"Event {i}: x mismatch"
        assert event_y == legacy_y, f"Event {i}: y mismatch"
        
        # Check metadata
        assert isinstance(meta["sample_id"], str)
        assert isinstance(meta["event_id"], int)
        assert meta["event_id"] == i
        assert meta["segment_id"] == 0
        assert "x_norm" in meta["metrics"]
        
        # Test legacy adapter
        adapter_x, adapter_y = legacy_loader_adapter(record)
        assert np.array_equal(adapter_x, legacy_x)
        assert adapter_y == legacy_y
    
    print("✓ Event schema roundtrip works correctly")


def test_config_creation_with_flags():
    """Test config creation with flags explicitly set."""
    print("Testing config creation with flags...")
    
    # Test creating config with flags off (should be no-op)
    cfg_off = Config.from_cli_args(
        adaptive_geometry=False,
        dynamic_comparator=False,
        strong_convexity=False,
        adaptive_privacy=False,
        drift_mode=False,
        window_erm=False,
        online_standardize=False
    )
    
    # Should match defaults
    assert cfg_off.adaptive_geometry == False
    assert cfg_off.dynamic_comparator == False
    assert cfg_off.strong_convexity == False
    assert cfg_off.adaptive_privacy == False
    assert cfg_off.drift_mode == False
    assert cfg_off.window_erm == False
    assert cfg_off.online_standardize == False
    
    # Test creating config with flags on (should store True)
    cfg_on = Config.from_cli_args(
        adaptive_geometry=True,
        dynamic_comparator=True
    )
    
    assert cfg_on.adaptive_geometry == True
    assert cfg_on.dynamic_comparator == True
    # Others should still be False (defaults)
    assert cfg_on.strong_convexity == False
    
    print("✓ Config creation with flags works")


def test_stream_consistency():
    """Test that stream behavior is consistent between runs."""
    print("Testing stream consistency...")
    
    # Same seed should produce same data
    stream1 = get_synthetic_linear_stream(dim=3, seed=123, use_event_schema=True)
    stream2 = get_synthetic_linear_stream(dim=3, seed=123, use_event_schema=True)
    
    for i in range(3):
        record1 = next(stream1)
        record2 = next(stream2)
        
        x1, y1, meta1 = parse_event_record(record1)
        x2, y2, meta2 = parse_event_record(record2)
        
        assert np.array_equal(x1, x2), f"Event {i}: x not consistent"
        assert y1 == y2, f"Event {i}: y not consistent"
        assert meta1["sample_id"] == meta2["sample_id"], f"Event {i}: sample_id not consistent"
    
    print("✓ Stream consistency verified")


def test_minimal_pipeline():
    """Test a minimal version of the full pipeline."""
    print("Testing minimal pipeline...")
    
    # This tests that the basic components work together
    cfg = Config()
    cfg.dataset = "synthetic"
    cfg.bootstrap_iters = 5
    cfg.max_events = 10
    
    # Import minimal components we need
    from data_loader import get_synthetic_linear_stream, parse_event_record
    
    # Get stream
    stream = get_synthetic_linear_stream(dim=3, seed=42, use_event_schema=True)
    
    # Process a few events
    events_processed = 0
    for record in stream:
        if events_processed >= cfg.max_events:
            break
            
        x, y, meta = parse_event_record(record)
        
        # Basic checks
        assert x.shape[0] == 3
        assert isinstance(y, (int, float))
        assert "sample_id" in meta
        assert "event_id" in meta
        assert "metrics" in meta
        
        events_processed += 1
    
    assert events_processed == cfg.max_events
    
    print("✓ Minimal pipeline works")


if __name__ == "__main__":
    print("Running flag default no-op tests...")
    test_flags_default_false()
    test_legacy_behavior_preserved()
    test_event_schema_roundtrip()
    test_config_creation_with_flags()
    test_stream_consistency()
    test_minimal_pipeline()
    print("✓ All flag default no-op tests passed!")