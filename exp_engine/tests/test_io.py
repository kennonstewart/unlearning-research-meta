import pytest
import tempfile
import os
import pandas as pd
from exp_engine.engine.io import write_seed_rows, write_event_rows


def test_write_seed_rows_creates_parquet():
    """Test that write_seed_rows creates partitioned Parquet files."""
    seed_data = [
        {
            "seed": 1,
            "algo": "memorypair", 
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
            "avg_regret_empirical": 0.1,
            "N_star_emp": 100,
            "m_emp": 10
        },
        {
            "seed": 2,
            "algo": "memorypair",
            "accountant": "zcdp", 
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
            "avg_regret_empirical": 0.15,
            "N_star_emp": 105,
            "m_emp": 12
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = write_seed_rows(seed_data, tmpdir)
        
        # Should create seeds directory
        assert os.path.exists(seeds_path)
        assert "seeds" in seeds_path
        
        # Should contain parquet files
        parquet_files = []
        for root, dirs, files in os.walk(seeds_path):
            parquet_files.extend([f for f in files if f.endswith('.parquet')])
        assert len(parquet_files) > 0


def test_write_seed_rows_with_grid_id():
    """Test that write_seed_rows can attach grid_id from params."""
    seed_data = [
        {
            "seed": 1,
            "algo": "memorypair",
            "avg_regret_empirical": 0.1
        }
    ]
    
    params = {
        "algo": "memorypair",
        "gamma_bar": 1.0,
        "accountant": "zcdp"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = write_seed_rows(seed_data, tmpdir, params)
        
        # Should create grid directory with params
        grid_dirs = [d for d in os.listdir(os.path.join(tmpdir, "grids")) if d.startswith("grid_id=")]
        assert len(grid_dirs) == 1
        
        # Should contain params.json
        grid_dir = os.path.join(tmpdir, "grids", grid_dirs[0])
        params_file = os.path.join(grid_dir, "params.json")
        assert os.path.exists(params_file)


def test_write_event_rows_creates_parquet():
    """Test that write_event_rows creates partitioned Parquet files."""
    event_data = [
        {
            "seed": 1,
            "algo": "memorypair",
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "event_type": "insert",
            "op": "insert", 
            "regret": 0.05,
            "acc": 0.8
        },
        {
            "seed": 1,
            "algo": "memorypair", 
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "event_type": "delete",
            "op": "delete",
            "regret": 0.03,
            "acc": 0.82
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = write_event_rows(event_data, tmpdir)
        
        # Should create events directory
        assert os.path.exists(events_path)
        assert "events" in events_path
        
        # Should contain parquet files
        parquet_files = []
        for root, dirs, files in os.walk(events_path):
            parquet_files.extend([f for f in files if f.endswith('.parquet')])
        assert len(parquet_files) > 0


def test_write_empty_data():
    """Test handling of empty data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty seed data
        seeds_path = write_seed_rows([], tmpdir, {})
        assert seeds_path == ""
        
        # Empty event data
        events_path = write_event_rows([], tmpdir, {})
        assert events_path == ""


def test_missing_partition_columns_handled():
    """Test that missing partition columns are handled gracefully."""
    seed_data = [
        {
            "seed": 1,
            "avg_regret_empirical": 0.1,
            # Missing many expected partition columns
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not crash despite missing columns
        params = {
            "algo": "memorypair",
            "gamma_bar": 1.0,
            "accountant": "zcdp"
        }
        seeds_path = write_seed_rows(seed_data, tmpdir, params)
        assert os.path.exists(seeds_path)