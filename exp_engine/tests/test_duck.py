import pytest
import tempfile
import os
import pandas as pd
from exp_engine.engine.duck import create_connection_and_views, query_events
from exp_engine.engine.io import write_event_rows

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
        assert "events" not in table_names


def test_create_connection_and_views_with_data():
    """Test creating connection and views with actual data."""
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
        write_event_rows(event_data, tmpdir)
        
        # Create connection and views
        conn = create_connection_and_views(tmpdir)
        
        # Views should exist now
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        assert "events" in table_names


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