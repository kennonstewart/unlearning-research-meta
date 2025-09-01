import pytest
import tempfile
import os
import pandas as pd
from exp_engine.engine.duck import create_connection_and_views
from exp_engine.engine.io import write_seed_rows, write_event_rows

# Skip DuckDB tests if not available
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not available")


def test_create_connection_and_views_empty_dir():
    """Test creating connection with empty data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = create_connection_and_views(tmpdir)
        
        # Should create connection even with no data
        assert conn is not None
        
        # Views should not exist yet
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert "seeds" not in table_names
        assert "events" not in table_names


def test_create_connection_and_views_with_data():
    """Test creating connection and views with actual data."""
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
        }
    ]
    
    event_data = [
        {
            "seed": 1,
            "algo": "memorypair",
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
            "event_type": "insert",
            "op": "insert",
            "regret": 0.05,
            "acc": 0.8
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some data first
        write_seed_rows(seed_data, tmpdir)
        write_event_rows(event_data, tmpdir)
        
        # Create connection and views
        conn = create_connection_and_views(tmpdir)
        
        # Views should exist now
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert "seeds" in table_names
        assert "events" in table_names
        assert "seeds_summary" in table_names
        assert "events_summary" in table_names


def test_query_seeds():
    """Test querying seeds view."""
    seed_data = [
        {
            "seed": 1,
            "algo": "memorypair",
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "avg_regret_empirical": 0.1
        },
        {
            "seed": 2,
            "algo": "memorypair", 
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "avg_regret_empirical": 0.2
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_seed_rows(seed_data, tmpdir)
        conn = create_connection_and_views(tmpdir)
        
        # Query all seeds
        df = query_seeds(conn)
        assert len(df) == 2
        assert "seed" in df.columns
        assert "avg_regret_empirical" in df.columns
        
        # Query with filter
        df_filtered = query_seeds(conn, "seed = 1")
        assert len(df_filtered) == 1
        assert df_filtered.iloc[0]["seed"] == 1
        
        # Query with limit
        df_limited = query_seeds(conn, limit=1)
        assert len(df_limited) == 1


def test_query_events():
    """Test querying events view."""
    event_data = [
        {
            "seed": 1,
            "algo": "memorypair",
            "event_type": "insert",
            "op": "insert",
            "regret": 0.05
        },
        {
            "seed": 1,
            "algo": "memorypair",
            "event_type": "delete", 
            "op": "delete",
            "regret": 0.03
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_event_rows(event_data, tmpdir)
        conn = create_connection_and_views(tmpdir)
        
        # Query all events
        df = query_events(conn)
        assert len(df) == 2
        assert "event_type" in df.columns
        assert "regret" in df.columns
        
        # Query with filter
        df_filtered = query_events(conn, "op = 'insert'")
        assert len(df_filtered) == 1
        assert df_filtered.iloc[0]["op"] == "insert"


def test_reuse_existing_connection():
    """Test reusing an existing DuckDB connection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial connection
        existing_conn = duckdb.connect()
        
        # Should reuse the connection
        conn = create_connection_and_views(tmpdir, existing_conn)
        assert conn is existing_conn