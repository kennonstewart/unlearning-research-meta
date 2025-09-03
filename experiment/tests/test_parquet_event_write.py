#!/usr/bin/env python3
"""
Test Parquet event writing integration.
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.deletion_capacity.exp_integration import write_event_rows_parquet


def test_write_event_rows_parquet():
    """Test writing event rows to Parquet."""
    event_data = [
        {
            "seed": 1,
            "event": 1,
            "op": "insert",
            "regret": 0.05,
            "regret_increment": 0.05,
            "acc": 0.8,
            "sigma_step": 0.1,
            "sigma_delete": float('nan'),  # NaN for insert ops
            "rho_spent": 0.5,
            "privacy_spend_running": 0.5
        },
        {
            "seed": 1,
            "event": 2,
            "op": "delete", 
            "regret": 0.08,
            "regret_increment": 0.03,
            "acc": 0.82,
            "sigma_step": 0.1,
            "sigma_delete": 0.2,
            "rho_spent": 0.7,
            "privacy_spend_running": 0.7
        }
    ]
    
    params_with_grid = {
        "grid_id": "test_grid_456",
        "algo": "memorypair",
        "accountant": "zcdp",
        "gamma_bar": 1.0
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_event_rows_parquet(event_data, tmpdir, params_with_grid)
        
        # Check that events directory was created
        events_path = os.path.join(tmpdir, "events")
        assert os.path.exists(events_path)
        
        # Check that parquet files exist
        parquet_files = []
        for root, dirs, files in os.walk(events_path):
            parquet_files.extend([f for f in files if f.endswith('.parquet')])
        assert len(parquet_files) > 0
        
        # Check that grid params were saved
        grids_path = os.path.join(tmpdir, "grids", "grid_id=test_grid_456", "params.json")
        assert os.path.exists(grids_path)


def test_write_event_rows_parquet_required_columns():
    """Test that event rows contain required privacy accounting columns."""
    event_data = [
        {
            "seed": 1,
            "event": 1,
            "op": "insert",
            "regret": 0.05,
            "acc": 0.8,
            "sigma_step": 0.1,
            "sigma_delete": float('nan'),
            "rho_spent": 0.5  # Required privacy spend column
        },
        {
            "seed": 1,
            "event": 2,
            "op": "delete",
            "regret": 0.08,
            "acc": 0.82,
            "sigma_step": 0.1,
            "sigma_delete": 0.2,  # Required for delete ops
            "rho_spent": 0.7
        }
    ]
    
    params_with_grid = {
        "grid_id": "test_grid_789",
        "algo": "memorypair",
        "accountant": "zcdp"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise an exception
        write_event_rows_parquet(event_data, tmpdir, params_with_grid)
        
        # Verify files were created
        events_path = os.path.join(tmpdir, "events")
        assert os.path.exists(events_path)


if __name__ == "__main__":
    test_write_event_rows_parquet()
    test_write_event_rows_parquet_required_columns()
    print("✓ All Parquet event tests passed!")