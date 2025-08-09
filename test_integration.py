"""
Integration test to verify the full pipeline works with event schema and extended logging.
Tests that existing functionality remains unchanged when flags are off.
"""

import os
import sys
import tempfile
import numpy as np
import csv

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity"))

from data_loader import get_synthetic_linear_stream, parse_event_record
from io_utils import EventLogger
from config import Config


def test_event_logger_with_extended_columns():
    """Test that EventLogger can handle extended columns."""
    print("Testing EventLogger with extended columns...")
    
    logger = EventLogger()
    
    # Log some events with extended columns
    logger.log("insert", 
               event=1,
               op="insert", 
               regret=0.5,
               acc=0.8,
               sample_id="test_001",
               event_id=1,
               segment_id=0,
               x_norm=2.3,
               S_scalar=None,
               eta_t=None,
               lambda_est=None,
               rho_step=None,
               sigma_step=None,
               sens_delete=None,
               P_T_est=None)
    
    logger.log("delete",
               event=2,
               op="delete",
               regret=0.6,
               acc=np.nan,
               sample_id="test_002", 
               event_id=2,
               segment_id=0,
               x_norm=1.8,
               S_scalar=None,
               eta_t=None,
               lambda_est=None,
               rho_step=None,
               sigma_step=None,
               sens_delete=None,
               P_T_est=None)
    
    # Test CSV export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        logger.to_csv(csv_path)
        
        # Verify CSV contents
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        
        # Check first row (insert)
        row1 = rows[0]
        assert row1['event'] == '1'
        assert row1['op'] == 'insert'
        assert row1['sample_id'] == 'test_001'
        assert row1['event_id'] == '1'
        assert row1['segment_id'] == '0'
        assert row1['x_norm'] == '2.3'
        
        # Check extended columns are present (even if None)
        extended_cols = ['S_scalar', 'eta_t', 'lambda_est', 'rho_step', 'sigma_step', 'sens_delete', 'P_T_est']
        for col in extended_cols:
            assert col in row1, f"Missing extended column: {col}"
        
        # Check second row (delete)
        row2 = rows[1]
        assert row2['event'] == '2'
        assert row2['op'] == 'delete'
        assert row2['sample_id'] == 'test_002'
        
        print("✓ EventLogger handles extended columns correctly")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_config_serialization():
    """Test that config with new flags can be serialized."""
    print("Testing config serialization...")
    
    cfg = Config()
    cfg.adaptive_geometry = True
    cfg.dataset = "synthetic"
    cfg.bootstrap_iters = 10
    
    # Convert to dict
    cfg_dict = cfg.to_dict()
    
    # Check new flags are present
    assert 'adaptive_geometry' in cfg_dict
    assert cfg_dict['adaptive_geometry'] == True
    assert 'dynamic_comparator' in cfg_dict
    assert cfg_dict['dynamic_comparator'] == False
    
    print("✓ Config serialization works with new flags")


def test_stream_generator_metadata():
    """Test that stream generators properly attach metadata."""
    print("Testing stream generator metadata...")
    
    stream = get_synthetic_linear_stream(dim=4, seed=100, use_event_schema=True)
    
    # Check first several events
    for i in range(5):
        record = next(stream)
        x, y, meta = parse_event_record(record)
        
        # Check dimensions
        assert x.shape[0] == 4, f"Expected dim 4, got {x.shape[0]}"
        assert isinstance(y, (int, float))
        
        # Check metadata progression
        assert meta["event_id"] == i, f"Expected event_id {i}, got {meta['event_id']}"
        assert isinstance(meta["sample_id"], str)
        assert meta["segment_id"] == 0
        
        # Check computed metrics
        assert "x_norm" in meta["metrics"]
        expected_norm = np.linalg.norm(x)
        actual_norm = meta["metrics"]["x_norm"]
        assert abs(actual_norm - expected_norm) < 1e-10, f"x_norm mismatch: {actual_norm} vs {expected_norm}"
    
    print("✓ Stream generator metadata works correctly")


def test_phases_helper_functions():
    """Test the phase helper functions work correctly."""
    print("Testing phases helper functions...")
    
    # Import the phases module helpers
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity"))
    from phases import PhaseState, _create_extended_log_entry
    
    # Create test state
    state = PhaseState()
    state.event = 42
    state.current_record = {
        "x": np.array([1.0, 2.0]),
        "y": 0.5,
        "sample_id": "test_sample",
        "event_id": 42,
        "segment_id": 1,
        "metrics": {"x_norm": 2.236}
    }
    
    # Create test config
    cfg = Config()
    
    # Create test base entry
    base_entry = {
        "event": 42,
        "op": "insert",
        "regret": 0.1,
        "acc": 0.9
    }
    
    # Test extended log entry creation
    extended_entry = _create_extended_log_entry(base_entry, state, None, cfg)
    
    # Check base fields are preserved
    assert extended_entry["event"] == 42
    assert extended_entry["op"] == "insert"
    assert extended_entry["regret"] == 0.1
    assert extended_entry["acc"] == 0.9
    
    # Check metadata fields added
    assert extended_entry["sample_id"] == "test_sample"
    assert extended_entry["event_id"] == 42
    assert extended_entry["segment_id"] == 1
    assert extended_entry["x_norm"] == 2.236
    
    # Check extended columns added (should be None)
    extended_cols = ['S_scalar', 'eta_t', 'lambda_est', 'rho_step', 'sigma_step', 'sens_delete', 'P_T_est']
    for col in extended_cols:
        assert col in extended_entry, f"Missing extended column: {col}"
        assert extended_entry[col] is None, f"Extended column {col} should be None"
    
    print("✓ Phases helper functions work correctly")


def test_backward_compatibility():
    """Test that legacy systems still work."""
    print("Testing backward compatibility...")
    
    # Test that we can still get (x, y) tuples when needed
    from data_loader import legacy_loader_adapter
    
    # Get an event record
    stream = get_synthetic_linear_stream(dim=3, seed=200, use_event_schema=True)
    record = next(stream)
    
    # Use legacy adapter
    x_legacy, y_legacy = legacy_loader_adapter(record)
    
    # Compare with direct parsing
    x_direct, y_direct, meta = parse_event_record(record)
    
    assert np.array_equal(x_legacy, x_direct)
    assert y_legacy == y_direct
    
    # Test legacy stream still works
    legacy_stream = get_synthetic_linear_stream(dim=3, seed=200, use_event_schema=False)
    x_old, y_old = next(legacy_stream)
    
    assert np.array_equal(x_old, x_direct)
    assert y_old == y_direct
    
    print("✓ Backward compatibility maintained")


def test_end_to_end_logging():
    """Test end-to-end logging with the new schema."""
    print("Testing end-to-end logging...")
    
    # Create a minimal experiment simulation
    cfg = Config()
    cfg.dataset = "synthetic"
    cfg.bootstrap_iters = 3
    cfg.max_events = 5
    
    # Get stream with new schema
    stream = get_synthetic_linear_stream(dim=2, seed=300, use_event_schema=True)
    
    # Create logger
    logger = EventLogger()
    
    # Simulate processing events
    for i in range(cfg.max_events):
        record = next(stream)
        x, y, meta = parse_event_record(record)
        
        # Create a log entry (simulating what phases would do)
        base_entry = {
            "event": i,
            "op": "insert" if i < 3 else "delete",
            "regret": i * 0.1,
            "acc": 0.9 - i * 0.1,
            # Privacy metrics (simulated)
            "eps_spent": i * 0.01,
            "capacity_remaining": 1.0 - i * 0.01,
        }
        
        # Add metadata from record
        base_entry.update({
            "sample_id": meta["sample_id"],
            "event_id": meta["event_id"],
            "segment_id": meta["segment_id"],
            "x_norm": meta["metrics"]["x_norm"],
        })
        
        # Add extended columns (all None for now)
        base_entry.update({
            "S_scalar": None,
            "eta_t": None,
            "lambda_est": None,
            "rho_step": None,
            "sigma_step": None,
            "sens_delete": None,
            "P_T_est": None,
        })
        
        logger.log(base_entry["op"], **base_entry)
    
    # Export to CSV and verify
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        logger.to_csv(csv_path)
        
        # Read and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == cfg.max_events
        
        # Check structure of first row
        row = rows[0]
        required_cols = [
            'event', 'op', 'regret', 'acc', 'eps_spent', 'capacity_remaining',
            'sample_id', 'event_id', 'segment_id', 'x_norm',
            'S_scalar', 'eta_t', 'lambda_est', 'rho_step', 'sigma_step', 'sens_delete', 'P_T_est'
        ]
        
        for col in required_cols:
            assert col in row, f"Missing required column: {col}"
        
        # Check that operations are as expected
        assert rows[0]['op'] == 'insert'
        assert rows[1]['op'] == 'insert'
        assert rows[2]['op'] == 'insert'
        assert rows[3]['op'] == 'delete'
        assert rows[4]['op'] == 'delete'
        
        print("✓ End-to-end logging works correctly")
        
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


if __name__ == "__main__":
    print("Running integration tests...")
    test_event_logger_with_extended_columns()
    test_config_serialization()
    test_stream_generator_metadata()
    test_phases_helper_functions()
    test_backward_compatibility()
    test_end_to_end_logging()
    print("✓ All integration tests passed!")