from __future__ import annotations
import os
from typing import Optional

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def create_connection_and_views(base_out: str, connection: Optional = None):
    """Create DuckDB connection with views over Parquet datasets.
    
    Args:
        base_out: Base output directory containing seeds/ and events/ datasets
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
    
    seeds_path = os.path.join(base_out, "seeds")
    events_path = os.path.join(base_out, "events")
    
    # Create seeds view if dataset exists
    if os.path.exists(seeds_path):
        # Use glob pattern to read all parquet files recursively
        seeds_pattern = os.path.join(seeds_path, "**/*.parquet")
        connection.execute(f"""
            CREATE OR REPLACE VIEW seeds AS 
            SELECT * FROM read_parquet('{seeds_pattern}')
        """)
    
    # Create events view if dataset exists  
    if os.path.exists(events_path):
        # Use glob pattern to read all parquet files recursively
        events_pattern = os.path.join(events_path, "**/*.parquet")
        connection.execute(f"""
            CREATE OR REPLACE VIEW events AS
            SELECT * FROM read_parquet('{events_pattern}')
        """)
    
    # Create some useful aggregation views
    if os.path.exists(seeds_path):
        # Get column names first to avoid referencing non-existent columns
        try:
            sample_query = connection.execute(f"SELECT * FROM read_parquet('{os.path.join(seeds_path, '**/*.parquet')}') LIMIT 1")
            seeds_columns = [desc[0] for desc in sample_query.description]
            
            # Build seeds_summary with available columns
            agg_cols = []
            group_cols = ["grid_id", "algo", "accountant", "gamma_bar", "gamma_split"]
            
            agg_cols.append("COUNT(*) as num_seeds")
            if "avg_regret_empirical" in seeds_columns:
                agg_cols.append("AVG(avg_regret_empirical) as mean_regret")
                agg_cols.append("STDDEV(avg_regret_empirical) as std_regret")
            if "N_star_emp" in seeds_columns:
                agg_cols.append("AVG(N_star_emp) as mean_n_star")
            if "m_emp" in seeds_columns:
                agg_cols.append("AVG(m_emp) as mean_deletions")
            
            # Only create view if we have required group columns
            if all(col in seeds_columns for col in group_cols):
                connection.execute(f"""
                    CREATE OR REPLACE VIEW seeds_summary AS
                    SELECT 
                        {', '.join(group_cols + agg_cols)}
                    FROM seeds
                    GROUP BY {', '.join(group_cols)}
                """)
        except Exception:
            # Skip creating seeds_summary if there's an issue
            pass
    
    if os.path.exists(events_path):
        # Get column names first to avoid referencing non-existent columns
        try:
            sample_query = connection.execute(f"SELECT * FROM read_parquet('{os.path.join(events_path, '**/*.parquet')}') LIMIT 1")
            events_columns = [desc[0] for desc in sample_query.description]
            
            # Build events_summary with available columns
            agg_cols = ["COUNT(*) as total_events"]
            group_cols = ["grid_id", "seed"]
            
            if "op" in events_columns:
                agg_cols.append("SUM(CASE WHEN op = 'insert' THEN 1 ELSE 0 END) as inserts")
                agg_cols.append("SUM(CASE WHEN op = 'delete' THEN 1 ELSE 0 END) as deletions")
            if "regret" in events_columns:
                agg_cols.append("AVG(regret) as avg_regret_per_event")
                # Add finalized regret analysis
                agg_cols.append("AVG(CASE WHEN op = 'insert' AND regret >= 0 THEN regret END) as avg_nonneg_regret_insert")
                agg_cols.append("SUM(CASE WHEN regret < 0 THEN 1 ELSE 0 END) as negative_regret_count")
            if "cum_regret" in events_columns:
                agg_cols.append("MAX(cum_regret) as final_cum_regret")
            
            # Add regret decomposition columns (finalized spec)
            regret_decomp_cols = ["regret_dynamic", "regret_static_term", "regret_path_term"]
            for col in regret_decomp_cols:
                if col in events_columns:
                    agg_cols.extend([
                        f"AVG({col}) as avg_{col}",
                        f"MAX({col}) as max_{col}"
                    ])
            
            # Only create view if we have required group columns
            if all(col in events_columns for col in group_cols):
                connection.execute(f"""
                    CREATE OR REPLACE VIEW events_summary AS  
                    SELECT
                        {', '.join(group_cols + agg_cols)}
                    FROM events
                    GROUP BY {', '.join(group_cols)}
                """)
                
                # Create regret-specific view for detailed analysis (finalized spec)
                if "regret" in events_columns and "op" in events_columns:
                    connection.execute(f"""
                        CREATE OR REPLACE VIEW regret_analysis AS
                        SELECT
                            grid_id, seed, op,
                            COUNT(*) as event_count,
                            AVG(regret) as avg_regret,
                            STDDEV(regret) as std_regret,
                            MIN(regret) as min_regret,
                            MAX(regret) as max_regret,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY regret) as median_regret,
                            SUM(CASE WHEN regret < 0 THEN 1 ELSE 0 END) as negative_count,
                            AVG(CASE WHEN regret >= 0 THEN regret END) as avg_nonneg_regret
                        FROM events
                        WHERE regret IS NOT NULL
                        GROUP BY grid_id, seed, op
                    """)
        except Exception:
            # Skip creating events_summary if there's an issue
            pass
    
    return connection


def query_seeds(connection, where_clause: str = "1=1", limit: Optional[int] = None) -> "pd.DataFrame":
    """Query seeds view with optional filtering.
    
    Args:
        connection: DuckDB connection with views configured
        where_clause: SQL WHERE clause (default: no filtering)
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with query results
    """
    query = f"SELECT * FROM seeds WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    return connection.execute(query).df()


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


def query_regret_analysis(connection, where_clause: str = "1=1", limit: Optional[int] = None) -> "pd.DataFrame":
    """Query regret analysis view for detailed regret statistics.
    
    Args:
        connection: DuckDB connection with views configured
        where_clause: SQL WHERE clause (default: no filtering)
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with regret analysis results
    """
    try:
        query = f"SELECT * FROM regret_analysis WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"
        
        return connection.execute(query).df()
    except Exception:
        # Fallback if regret_analysis view doesn't exist
        import pandas as pd
        return pd.DataFrame()


def get_negative_regret_summary(connection) -> "pd.DataFrame":
    """Get summary of negative regret occurrences across experiments.
    
    Args:
        connection: DuckDB connection with views configured
        
    Returns:
        Pandas DataFrame with negative regret summary
    """
    try:
        query = """
            SELECT 
                grid_id, 
                COUNT(DISTINCT seed) as seeds_with_neg_regret,
                SUM(negative_count) as total_negative_events,
                AVG(negative_count / NULLIF(event_count, 0)) as avg_negative_fraction
            FROM regret_analysis 
            WHERE negative_count > 0
            GROUP BY grid_id
            ORDER BY total_negative_events DESC
        """
        return connection.execute(query).df()
    except Exception:
        import pandas as pd
        return pd.DataFrame()