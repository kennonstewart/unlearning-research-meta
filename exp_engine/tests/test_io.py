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


def test_normalize_regret_missing_regret_has_cum_regret():
    """Test normalization when regret is missing but cum_regret exists."""
    event_data = [
        {
            "seed": 1,
            "event": 1,
            "cum_regret": 0.1,
            "op": "insert"
        },
        {
            "seed": 1,
            "event": 2,
            "cum_regret": 0.15,
            "op": "insert"
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = write_event_rows(event_data, tmpdir)
        
        # Read back the data to verify normalization
        parquet_files = []
        for root, dirs, files in os.walk(events_path):
            parquet_files.extend([os.path.join(root, f) for f in files if f.endswith('.parquet')])
        
        # Read all parquet files and combine
        dfs = [pd.read_parquet(f) for f in parquet_files]
        result_df = pd.concat(dfs, ignore_index=True).sort_values('event')
        
        # Should have both regret and cum_regret columns now
        assert 'regret' in result_df.columns
        assert 'cum_regret' in result_df.columns
        # regret should be set from cum_regret
        assert result_df.iloc[0]['regret'] == 0.1
        assert result_df.iloc[1]['regret'] == 0.15


def test_normalize_regret_missing_cum_regret_has_regret():
    """Test normalization when cum_regret is missing but regret exists."""
    event_data = [
        {
            "seed": 1,
            "event": 1,
            "regret": 0.1,
            "op": "insert"
        },
        {
            "seed": 1,
            "event": 2,
            "regret": 0.15,
            "op": "insert"
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = write_event_rows(event_data, tmpdir)
        
        # Read back the data to verify normalization
        parquet_files = []
        for root, dirs, files in os.walk(events_path):
            parquet_files.extend([os.path.join(root, f) for f in files if f.endswith('.parquet')])
        
        # Read all parquet files and combine
        dfs = [pd.read_parquet(f) for f in parquet_files]
        result_df = pd.concat(dfs, ignore_index=True).sort_values('event')
        
        # Should have both regret and cum_regret columns now
        assert 'regret' in result_df.columns
        assert 'cum_regret' in result_df.columns
        # cum_regret should be mirrored from regret
        assert result_df.iloc[0]['cum_regret'] == 0.1
        assert result_df.iloc[1]['cum_regret'] == 0.15


def test_normalize_regret_fallback_to_regret_increment():
    """Test normalization when only regret_increment exists."""
    event_data = [
        {
            "seed": 1,
            "event": 1,
            "regret_increment": 0.05,
            "op": "insert"
        },
        {
            "seed": 1,
            "event": 2,
            "regret_increment": 0.03,
            "op": "delete"
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = write_event_rows(event_data, tmpdir)
        
        # Read back the data to verify normalization
        parquet_files = []
        for root, dirs, files in os.walk(events_path):
            parquet_files.extend([os.path.join(root, f) for f in files if f.endswith('.parquet')])
        
        # Read all parquet files and combine
        dfs = [pd.read_parquet(f) for f in parquet_files]
        result_df = pd.concat(dfs, ignore_index=True).sort_values('event')
        
        # Should have regret set from regret_increment
        assert 'regret' in result_df.columns
        assert result_df.iloc[0]['regret'] == 0.05
        assert result_df.iloc[1]['regret'] == 0.03


def test_normalize_regret_preserves_existing_values():
    """Test that normalization doesn't overwrite existing values."""
    event_data = [
        {
            "seed": 1,
            "event": 1,
            "regret": 0.1,
            "cum_regret": 0.15,  # Different value
            "regret_increment": 0.05,
            "op": "insert"
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = write_event_rows(event_data, tmpdir)
        
        # Read back the data to verify normalization
        parquet_files = []
        for root, dirs, files in os.walk(events_path):
            parquet_files.extend([os.path.join(root, f) for f in files if f.endswith('.parquet')])
        
        # Read all parquet files and combine
        dfs = [pd.read_parquet(f) for f in parquet_files]
        result_df = pd.concat(dfs, ignore_index=True)
        
        # Original values should be preserved (not overwritten)
        assert result_df.iloc[0]['regret'] == 0.1
        assert result_df.iloc[0]['cum_regret'] == 0.15