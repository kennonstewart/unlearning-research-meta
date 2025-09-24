from __future__ import annotations
import os
import glob
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
        # Try different patterns to find parquet files
        events_patterns = [
            os.path.join(events_path, "**/*.parquet"),
            os.path.join(events_path, "*/*/*.parquet"),
            os.path.join(events_path, "*/*.parquet"),
            os.path.join(events_path, "*.parquet")
        ]
        
        events_pattern = None
        for pattern in events_patterns:
            if glob.glob(pattern):
                events_pattern = pattern
                break
        
        if not events_pattern:
            print(f"Warning: No parquet files found in {events_path}")
            return connection
        
        # First discover all columns to ensure complete schema
        try:
            
            # Get all matching files
            all_files = glob.glob(events_pattern)
            if not all_files:
                print(f"Warning: No files found matching pattern: {events_pattern}")
                connection.execute(f"""
                    CREATE OR REPLACE VIEW events AS
                    SELECT * FROM read_parquet('{events_pattern}')
                """)
            else:
                print(f"Found {len(all_files)} files matching pattern: {events_pattern}")
                
                # Discover columns from each file individually
                all_columns = set()
                for file_path in all_files:
                    try:
                        # Get schema from individual file
                        schema_query = connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}') LIMIT 0").df()
                        if schema_query is not None and not schema_query.empty:
                            file_columns = set(schema_query['column_name'].tolist())
                            all_columns.update(file_columns)
                            print(f"  {os.path.basename(file_path)}: {len(file_columns)} columns")
                    except Exception as e:
                        print(f"  {os.path.basename(file_path)}: Error - {e}")
                
                # Convert to sorted list for consistent ordering
                columns_list = sorted(list(all_columns))
                print(f"Total unique columns discovered: {len(columns_list)}")
                
                if columns_list:
                    if len(all_files) == 1:
                        # Single file - use direct approach
                        column_list = ', '.join(columns_list)
                        connection.execute(f"""
                            CREATE OR REPLACE VIEW events AS
                            SELECT {column_list} FROM read_parquet('{events_pattern}')
                        """)
                    else:
                        # Multiple files - create UNION ALL with all columns
                        # Use a more robust approach that handles missing columns
                        union_parts = []
                        for file_path in all_files:
                            # Get columns available in this specific file
                            try:
                                file_schema = connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}') LIMIT 0").df()
                                file_columns = set(file_schema['column_name'].tolist())
                                
                                # Create SELECT statement with all discovered columns
                                # Use CASE statements to handle missing columns
                                select_parts = []
                                for col in columns_list:
                                    if col in file_columns:
                                        select_parts.append(col)
                                    else:
                                        select_parts.append(f"NULL as {col}")
                                
                                column_list = ', '.join(select_parts)
                                union_parts.append(f"SELECT {column_list} FROM read_parquet('{file_path}')")
                                
                            except Exception as e:
                                print(f"  Error processing {os.path.basename(file_path)}: {e}")
                                continue
                        
                        if union_parts:
                            union_query = " UNION ALL ".join(union_parts)
                            connection.execute(f"""
                                CREATE OR REPLACE VIEW events AS
                                {union_query}
                            """)
                        else:
                            # Fallback if no files could be processed
                            connection.execute(f"""
                                CREATE OR REPLACE VIEW events AS
                                SELECT * FROM read_parquet('{events_pattern}')
                            """)
                    
                    print(f"Created events view with {len(columns_list)} columns: {columns_list}")
                else:
                    # Fallback to original approach
                    connection.execute(f"""
                        CREATE OR REPLACE VIEW events AS
                        SELECT * FROM read_parquet('{events_pattern}')
                    """)
                    
        except Exception as e:
            print(f"Warning: Error creating events view with full schema: {e}")
            # Fallback to original approach
            connection.execute(f"""
                CREATE OR REPLACE VIEW events AS
                SELECT * FROM read_parquet('{events_pattern}')
            """)
    
        # Create some useful aggregation views
        try:
            # Get columns from the events view we just created
            events_columns = connection.execute("DESCRIBE events").df()['column_name'].tolist()
            
            # Build events_summary with available columns
            agg_cols = ["COUNT(*) as total_events"]
            group_cols = ["grid_id", "seed"]
            
            if "op" in events_columns:
                agg_cols.append("SUM(CASE WHEN op = 'insert' THEN 1 ELSE 0 END) as inserts")
                agg_cols.append("SUM(CASE WHEN op = 'delete' THEN 1 ELSE 0 END) as deletions")
            
            # Add aggregations for numeric columns dynamically
            numeric_columns = ["regret", "loss", "acc", "step", "timestamp", "sample_id", "segment_id", 
                              "noise", "rho_step", "rho_spent", "sigma_step", "sigma_delete"]
            boolean_columns = ["sc_stable"]
            
            for col in numeric_columns:
                if col in events_columns:
                    if col == "regret":
                        agg_cols.append("AVG(regret) as avg_regret_per_event")
                        agg_cols.append("SUM(CASE WHEN regret < 0 THEN 1 ELSE 0 END) as negative_regret_count")
                    elif col == "timestamp":
                        agg_cols.extend([
                            f"MIN({col}) as start_{col}",
                            f"MAX({col}) as end_{col}"
                        ])
                    else:
                        agg_cols.extend([
                            f"AVG({col}) as avg_{col}",
                            f"MIN({col}) as min_{col}", 
                            f"MAX({col}) as max_{col}"
                        ])
            
            # Handle boolean columns separately (no AVG aggregation)
            for col in boolean_columns:
                if col in events_columns:
                    agg_cols.extend([
                        f"MIN({col}) as min_{col}", 
                        f"MAX({col}) as max_{col}",
                        f"SUM(CASE WHEN {col} THEN 1 ELSE 0 END) as count_true_{col}"
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
                print(f"Created events_summary view with {len(agg_cols)} aggregations")
        except Exception as e:
            print(f"Warning: Could not create events_summary view: {e}")
            # Skip creating events_summary if there's an issue
            pass
    
    # Create workload-only events view for theoretical regret analysis
    try:
        _create_workload_events_view(connection)
    except Exception as e:
        print(f"Warning: Could not create workload events view: {e}")
    
    return connection


def _create_workload_events_view(connection) -> None:
    """
    Create workload-only events view with regret metrics computed only from workload phase onwards.
    
    The workload phase begins at the first 'delete' event per run, or at the theoretical 
    sample complexity N_star if no deletes exist. This view provides:
    - Running workload-only cumulative regret
    - Running workload-only event count  
    - Running workload-only average regret
    
    Args:
        connection: DuckDB connection
    """
    connection.execute("""
        CREATE OR REPLACE VIEW events_workload AS
        WITH workload_boundaries AS (
            -- Find workload phase start for each run (grid_id, seed)
            SELECT 
                grid_id,
                seed,
                COALESCE(
                    MIN(CASE WHEN op = 'delete' THEN event_id END),
                    -- Fallback: Use N_star from calibration if available, else use event where phase changes
                    MIN(CASE WHEN N_star IS NOT NULL AND event_id >= N_star THEN event_id END),
                    -- Final fallback: Start of data (no workload boundary found)
                    MIN(event_id)
                ) AS workload_start_event_id
            FROM events
            GROUP BY grid_id, seed
        ),
        workload_baseline AS (
            -- Get cumulative regret at workload phase start (baseline to subtract)
            SELECT 
                wb.grid_id,
                wb.seed,
                wb.workload_start_event_id,
                COALESCE(e.cum_regret, 0.0) AS workload_baseline_cum_regret,
                COALESCE(e.noise_regret_cum, 0.0) AS workload_baseline_noise_regret,
                COALESCE(e.cum_regret_with_noise, 0.0) AS workload_baseline_cum_regret_with_noise
            FROM workload_boundaries wb
            LEFT JOIN events e ON 
                wb.grid_id = e.grid_id 
                AND wb.seed = e.seed 
                AND wb.workload_start_event_id = e.event_id
        )
        SELECT 
            e.*,
            wb.workload_start_event_id,
            wb.workload_baseline_cum_regret,
            wb.workload_baseline_noise_regret,
            wb.workload_baseline_cum_regret_with_noise,
            -- Workload-only cumulative regret (subtract baseline)
            GREATEST(0.0, COALESCE(e.cum_regret, 0.0) - wb.workload_baseline_cum_regret) AS workload_cum_regret,
            GREATEST(0.0, COALESCE(e.noise_regret_cum, 0.0) - wb.workload_baseline_noise_regret) AS workload_noise_regret_cum,
            GREATEST(0.0, COALESCE(e.cum_regret_with_noise, 0.0) - wb.workload_baseline_cum_regret_with_noise) AS workload_cum_regret_with_noise,
            -- Workload-only event count (events since workload start)
            GREATEST(1, e.event_id - wb.workload_start_event_id + 1) AS workload_events_seen,
            -- Workload-only average regret
            CASE 
                WHEN e.event_id >= wb.workload_start_event_id THEN
                    GREATEST(0.0, COALESCE(e.cum_regret, 0.0) - wb.workload_baseline_cum_regret) / 
                    GREATEST(1, e.event_id - wb.workload_start_event_id + 1)
                ELSE NULL  -- Pre-workload events don't have workload avg regret
            END AS workload_avg_regret,
            CASE 
                WHEN e.event_id >= wb.workload_start_event_id THEN
                    GREATEST(0.0, COALESCE(e.cum_regret_with_noise, 0.0) - wb.workload_baseline_cum_regret_with_noise) / 
                    GREATEST(1, e.event_id - wb.workload_start_event_id + 1)
                ELSE NULL  -- Pre-workload events don't have workload avg regret
            END AS workload_avg_regret_with_noise,
            -- Flag indicating if this event is in workload phase
            CASE WHEN e.event_id >= wb.workload_start_event_id THEN TRUE ELSE FALSE END AS is_workload_phase
        FROM events e
        INNER JOIN workload_baseline wb ON e.grid_id = wb.grid_id AND e.seed = wb.seed
        ORDER BY e.grid_id, e.seed, e.event_id
    """)


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


def query_workload_events(connection, where_clause: str = "1=1", limit: Optional[int] = None) -> "pd.DataFrame":
    """Query workload-only events view with optional filtering.
    
    Args:
        connection: DuckDB connection with views configured
        where_clause: SQL WHERE clause (default: no filtering)
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with workload-only regret metrics
    """
    query = f"SELECT * FROM events_workload WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    return connection.execute(query).df()


def query_workload_regret_analysis(connection) -> "pd.DataFrame":
    """Query workload regret analysis with summary statistics per run.
    
    Args:
        connection: DuckDB connection with views configured
        
    Returns:
        Pandas DataFrame with workload regret analysis per run
    """
    return connection.execute("""
        SELECT 
            grid_id,
            seed,
            workload_start_event_id,
            COUNT(*) FILTER (WHERE is_workload_phase) as workload_events,
            COUNT(*) FILTER (WHERE NOT is_workload_phase) as pre_workload_events,
            MAX(workload_cum_regret) as final_workload_cum_regret,
            MAX(workload_avg_regret) as final_workload_avg_regret,
            MAX(workload_cum_regret_with_noise) as final_workload_cum_regret_with_noise,
            MAX(workload_avg_regret_with_noise) as final_workload_avg_regret_with_noise,
            -- Compare workload vs total average regret
            MAX(avg_regret) as final_total_avg_regret,
            MAX(workload_avg_regret) - MAX(avg_regret) as workload_vs_total_avg_regret_diff,
            SUM(CASE WHEN is_workload_phase AND workload_avg_regret < 0 THEN 1 ELSE 0 END) as negative_workload_regret_count
        FROM events_workload
        GROUP BY grid_id, seed, workload_start_event_id
        ORDER BY grid_id, seed
    """).df()


def query_regret_analysis(connection) -> "pd.DataFrame":
    """Query regret analysis with summary statistics per run.
    
    Args:
        connection: DuckDB connection with views configured
        
    Returns:
        Pandas DataFrame with regret analysis per run
    """
    return connection.execute("""
        SELECT 
            grid_id,
            seed,
            COUNT(*) as total_events,
            MAX(cum_regret) as final_cum_regret,
            MAX(avg_regret) as final_avg_regret,
            MAX(cum_regret_with_noise) as final_cum_regret_with_noise,
            MAX(avg_regret_with_noise) as final_avg_regret_with_noise,
            SUM(CASE WHEN regret < 0 THEN 1 ELSE 0 END) as negative_count
        FROM events
        GROUP BY grid_id, seed
        ORDER BY grid_id, seed
    """).df()


def get_negative_regret_summary(connection) -> "pd.DataFrame":
    """Get summary of negative regret events.
    
    Args:
        connection: DuckDB connection with views configured
        
    Returns:
        Pandas DataFrame with negative regret summary
    """
    return connection.execute("""
        SELECT 
            grid_id,
            seed,
            COUNT(*) FILTER (WHERE regret < 0) as negative_regret_events,
            COUNT(*) as total_events,
            COUNT(*) FILTER (WHERE regret < 0) * 100.0 / COUNT(*) as negative_regret_pct,
            MIN(regret) as min_regret,
            AVG(regret) FILTER (WHERE regret < 0) as avg_negative_regret
        FROM events
        GROUP BY grid_id, seed
        HAVING COUNT(*) FILTER (WHERE regret < 0) > 0
        ORDER BY negative_regret_pct DESC
    """).df()