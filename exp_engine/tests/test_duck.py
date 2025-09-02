import pytest
import tempfile
import os
import pandas as pd
from exp_engine.engine.duck import create_connection_and_views, query_seeds, query_events
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


def test_events_summary_semantics():
    """Test that events_summary computes metrics matching CSV semantics."""
    # Create synthetic event data that tests the key semantics
    event_data = [
        # Seed 1: cumulative regret that increases then decreases (tests ARG_MAX vs MAX)
        {"grid_id": "test1", "seed": 1, "event": 1, "regret": 0.1, "cum_regret": 0.1, "op": "insert", "acc": 0.8},
        {"grid_id": "test1", "seed": 1, "event": 2, "regret": 0.2, "cum_regret": 0.2, "op": "insert", "acc": 0.85},
        {"grid_id": "test1", "seed": 1, "event": 3, "regret": 0.15, "cum_regret": 0.15, "op": "delete", "acc": 0.9},  # Decreases!
        
        # Seed 2: simple increasing case
        {"grid_id": "test1", "seed": 2, "event": 1, "regret": 0.05, "cum_regret": 0.05, "op": "insert", "acc": 0.7},
        {"grid_id": "test1", "seed": 2, "event": 2, "regret": 0.08, "cum_regret": 0.08, "op": "insert", "acc": 0.75},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_event_rows(event_data, tmpdir)
        conn = create_connection_and_views(tmpdir)
        
        # Query events_summary
        summary_df = conn.execute("SELECT * FROM events_summary ORDER BY seed").df()
        
        # Verify we have the right seeds
        assert len(summary_df) == 2
        assert list(summary_df['seed']) == [1, 2]
        
        # Test avg_regret_empirical equals mean of cumulative curve (regret column)
        # Seed 1: mean of [0.1, 0.2, 0.15] = 0.15
        seed1_row = summary_df[summary_df['seed'] == 1].iloc[0]
        expected_avg_regret_1 = (0.1 + 0.2 + 0.15) / 3
        assert abs(seed1_row['avg_regret_empirical'] - expected_avg_regret_1) < 1e-10
        
        # Seed 2: mean of [0.05, 0.08] = 0.065
        seed2_row = summary_df[summary_df['seed'] == 2].iloc[0]
        expected_avg_regret_2 = (0.05 + 0.08) / 2
        assert abs(seed2_row['avg_regret_empirical'] - expected_avg_regret_2) < 1e-10
        
        # Test final_cum_regret uses last-row semantics (ARG_MAX), not MAX
        # Seed 1: last event (event=3) has cum_regret=0.15, but MAX would be 0.2
        assert seed1_row['final_cum_regret'] == 0.15  # Last row, not max
        
        # Seed 2: last event (event=2) has cum_regret=0.08
        assert seed2_row['final_cum_regret'] == 0.08
        
        # Test operation counts
        assert seed1_row['inserts'] == 2
        assert seed1_row['deletions'] == 1
        assert seed1_row['total_events'] == 3
        
        assert seed2_row['inserts'] == 2
        assert seed2_row['deletions'] == 0
        assert seed2_row['total_events'] == 2


def test_seeds_from_events_view():
    """Test that seeds_from_events view reproduces CSV per-seed rollup from events."""
    # Same test data as above
    event_data = [
        {"grid_id": "test1", "seed": 1, "event": 1, "regret": 0.1, "cum_regret": 0.1, "op": "insert", "acc": 0.8},
        {"grid_id": "test1", "seed": 1, "event": 2, "regret": 0.2, "cum_regret": 0.2, "op": "insert", "acc": 0.85},
        {"grid_id": "test1", "seed": 1, "event": 3, "regret": 0.15, "cum_regret": 0.15, "op": "delete", "acc": 0.9},
        
        {"grid_id": "test1", "seed": 2, "event": 1, "regret": 0.05, "cum_regret": 0.05, "op": "insert", "acc": 0.7},
        {"grid_id": "test1", "seed": 2, "event": 2, "regret": 0.08, "cum_regret": 0.08, "op": "insert", "acc": 0.75},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_event_rows(event_data, tmpdir)
        conn = create_connection_and_views(tmpdir)
        
        # Query seeds_from_events
        seeds_df = conn.execute("SELECT * FROM seeds_from_events ORDER BY seed").df()
        
        # Verify we have the right structure
        assert len(seeds_df) == 2
        assert 'avg_regret_empirical' in seeds_df.columns
        assert 'N_star_emp' in seeds_df.columns
        assert 'm_emp' in seeds_df.columns
        assert 'total_events' in seeds_df.columns
        assert 'final_acc' in seeds_df.columns
        
        # Test the values match what CSV pipeline would compute
        seed1_row = seeds_df[seeds_df['seed'] == 1].iloc[0]
        seed2_row = seeds_df[seeds_df['seed'] == 2].iloc[0]
        
        # Same avg_regret_empirical as events_summary
        expected_avg_regret_1 = (0.1 + 0.2 + 0.15) / 3
        expected_avg_regret_2 = (0.05 + 0.08) / 2
        assert abs(seed1_row['avg_regret_empirical'] - expected_avg_regret_1) < 1e-10
        assert abs(seed2_row['avg_regret_empirical'] - expected_avg_regret_2) < 1e-10
        
        # Operation counts
        assert seed1_row['N_star_emp'] == 2  # 2 inserts
        assert seed1_row['m_emp'] == 1       # 1 delete
        assert seed1_row['total_events'] == 3
        
        assert seed2_row['N_star_emp'] == 2  # 2 inserts  
        assert seed2_row['m_emp'] == 0       # 0 deletes
        assert seed2_row['total_events'] == 2
        
        # final_acc should use ARG_MAX (last by event)
        assert seed1_row['final_acc'] == 0.9   # Last event (event=3)
        assert seed2_row['final_acc'] == 0.75  # Last event (event=2)


def test_regret_normalization_affects_duckdb_queries():
    """Test that the regret normalization in write_event_rows properly affects DuckDB queries."""
    # Test data with only cum_regret (no regret column)
    event_data = [
        {"grid_id": "test1", "seed": 1, "event": 1, "cum_regret": 0.1, "op": "insert"},
        {"grid_id": "test1", "seed": 1, "event": 2, "cum_regret": 0.15, "op": "insert"},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # The normalization should fill in regret from cum_regret
        write_event_rows(event_data, tmpdir)
        conn = create_connection_and_views(tmpdir)
        
        # Verify that events now has both regret and cum_regret
        events_df = conn.execute("SELECT * FROM events ORDER BY event").df()
        assert 'regret' in events_df.columns
        assert 'cum_regret' in events_df.columns
        
        # Both should have the same values since regret was filled from cum_regret
        assert events_df.iloc[0]['regret'] == 0.1
        assert events_df.iloc[0]['cum_regret'] == 0.1
        assert events_df.iloc[1]['regret'] == 0.15
        assert events_df.iloc[1]['cum_regret'] == 0.15
        
        # events_summary should compute avg_regret_empirical from regret column
        summary_df = conn.execute("SELECT * FROM events_summary").df()
        expected_avg = (0.1 + 0.15) / 2
        assert abs(summary_df.iloc[0]['avg_regret_empirical'] - expected_avg) < 1e-10


def test_fallback_without_event_column():
    """Test that views handle missing event column gracefully."""
    # Test data without event column
    event_data = [
        {"grid_id": "test1", "seed": 1, "regret": 0.1, "cum_regret": 0.1, "op": "insert", "acc": 0.8},
        {"grid_id": "test1", "seed": 1, "regret": 0.2, "cum_regret": 0.05, "op": "insert", "acc": 0.85},  # cum_regret lower!
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        write_event_rows(event_data, tmpdir)
        conn = create_connection_and_views(tmpdir)
        
        # Without event column, should fall back to MAX aggregations
        summary_df = conn.execute("SELECT * FROM events_summary").df()
        
        # final_cum_regret should be MAX(cum_regret) = 0.1 (not last row)
        assert summary_df.iloc[0]['final_cum_regret'] == 0.1
        
        # final_acc should be MAX(acc) = 0.85 (not last row)
        seeds_df = conn.execute("SELECT * FROM seeds_from_events").df()
        assert seeds_df.iloc[0]['final_acc'] == 0.85