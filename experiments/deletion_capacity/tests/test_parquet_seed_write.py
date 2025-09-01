#!/usr/bin/env python3
"""
Test Parquet seed writing integration.
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.deletion_capacity.exp_integration import (
    build_params_from_config, 
    write_seed_summary_parquet
)


def test_build_params_from_config():
    """Test parameter building and grid_id attachment."""
    cfg = {
        "algo": "memorypair",
        "accountant": "zcdp", 
        "gamma_bar": 1.0,
        "gamma_split": 0.5,
        "rho_total": 2.0,
        "seed": 42,  # volatile - should be removed
        "base_out": "/tmp/results",  # volatile - should be removed
        "timestamp": "2025-01-01"  # volatile - should be removed
    }
    
    params = build_params_from_config(cfg)
    
    # Should have grid_id
    assert "grid_id" in params
    assert len(params["grid_id"]) == 12  # SHA-256 12-char prefix
    
    # Should not have volatile keys
    assert "seed" not in params
    assert "base_out" not in params
    assert "timestamp" not in params
    
    # Should have stable parameters
    assert params["algo"] == "memorypair"
    assert params["accountant"] == "zcdp"
    assert params["gamma_bar"] == 1.0
    assert params["rho_total"] == 2.0


def test_write_seed_summary_parquet():
    """Test writing seed summaries to Parquet."""
    seed_data = [
        {
            "seed": 1,
            "avg_regret_empirical": 0.1,
            "N_star_emp": 100,
            "m_emp": 10,
            "final_acc": 0.95,
            "total_events": 1000,
            "gamma_pass_overall": True,
            "rho_spent_final": 1.5,
            "rho_util": 0.75
        },
        {
            "seed": 2,
            "avg_regret_empirical": 0.15,
            "N_star_emp": 105,
            "m_emp": 12,
            "final_acc": 0.92,
            "total_events": 1100,
            "gamma_pass_overall": True,
            "rho_spent_final": 1.6,
            "rho_util": 0.80
        }
    ]
    
    params_with_grid = {
        "grid_id": "test_grid_123",
        "algo": "memorypair",
        "accountant": "zcdp",
        "gamma_bar": 1.0
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_seed_summary_parquet(seed_data, tmpdir, params_with_grid)
        
        # Check that seeds directory was created
        seeds_path = os.path.join(tmpdir, "seeds")
        assert os.path.exists(seeds_path)
        
        # Check that parquet files exist
        parquet_files = []
        for root, dirs, files in os.walk(seeds_path):
            parquet_files.extend([f for f in files if f.endswith('.parquet')])
        assert len(parquet_files) > 0
        
        # Check that grid params were saved
        grids_path = os.path.join(tmpdir, "grids", "grid_id=test_grid_123", "params.json")
        assert os.path.exists(grids_path)


if __name__ == "__main__":
    test_build_params_from_config()
    test_write_seed_summary_parquet()
    print("✓ All Parquet seed tests passed!")