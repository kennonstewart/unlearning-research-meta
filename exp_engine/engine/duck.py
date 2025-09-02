from __future__ import annotations
import os
from typing import Optional

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def create_connection_and_views(base_out: str, connection: Optional = None):
    """Create DuckDB connection with views over event-level Parquet datasets.
    
    Args:
        base_out: Base output directory containing events/ dataset
        connection: Optional existing DuckDB connection to reuse
        
    Returns:
        DuckDB connection with views configured
        
    Raises:
        ImportError: If DuckDB is not available
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB is not available. Install with: pip install duckdb")
    
    # Create or reuse connection
    if connection is None:
        connection = duckdb.connect()
    
    events_path = os.path.join(base_out, "events")
    
    # Create events view if dataset exists  
    if os.path.exists(events_path):
        # Use glob pattern to read all parquet files recursively
        events_pattern = os.path.join(events_path, "**/*.parquet")
        connection.execute(f"""
            CREATE OR REPLACE VIEW events AS
            SELECT * FROM read_parquet('{events_pattern}')
        """)
    
        # Create some useful aggregation views
        try:
            sample_query = connection.execute(f"SELECT * FROM read_parquet('{events_pattern}') LIMIT 1")
            events_columns = [desc[0] for desc in sample_query.description]
            
            # Build events_summary with available columns
            agg_cols = ["COUNT(*) as total_events"]
            group_cols = ["grid_id", "seed"]
            
            if "op" in events_columns:
                agg_cols.append("SUM(CASE WHEN op = 'insert' THEN 1 ELSE 0 END) as inserts")
                agg_cols.append("SUM(CASE WHEN op = 'delete' THEN 1 ELSE 0 END) as deletions")
            if "regret" in events_columns:
                agg_cols.append("AVG(regret) as avg_regret_per_event")
                agg_cols.append("SUM(CASE WHEN regret < 0 THEN 1 ELSE 0 END) as negative_regret_count")
            
            # Only create view if we have required group columns
            if all(col in events_columns for col in group_cols):
                connection.execute(f"""
                    CREATE OR REPLACE VIEW events_summary AS  
                    SELECT
                        {', '.join(group_cols + agg_cols)}
                    FROM events
                    GROUP BY {', '.join(group_cols)}
                """)
        except Exception:
            # Skip creating events_summary if there's an issue
            pass
    
    return connection


def query_events(connection, where_clause: str = "1=1", limit: Optional[int] = None) -> "pd.DataFrame":
    """Query events view with optional filtering.
    
    Args:
        connection: DuckDB connection with views configured
        where_clause: SQL WHERE clause (default: no filtering)
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with query results
    """
    query = f"SELECT * FROM events WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    return connection.execute(query).df()