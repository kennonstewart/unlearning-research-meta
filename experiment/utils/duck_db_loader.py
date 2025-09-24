"""
DuckDB loader with star schema support for experiment data.

This module provides functionality to load experiment data into DuckDB
with a star schema design, including parameter data integration.

Example usage with memory limit to prevent kernel crashes:
    conn = load_star_schema(
        input_path="results_parquet/events/grid_id=*/seed=*/*.parquet",
        memory_limit="1GB"  # Optional: limit memory usage (MB or GB only)
    )
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def load_star_schema(
    input_path: str,
    db_path: Optional[str] = None,
    staging_table: str = "staging_events",
    run_ddl: bool = True,
    create_events_view: bool = True,
    include_parameters: bool = True,
    parameters_path: Optional[str] = None,
    memory_limit: Optional[str] = None
) -> duckdb.DuckDBPyConnection:
    """
    Load experiment data into DuckDB with star schema design.
    
    Args:
        input_path: Path to the events parquet data (glob pattern supported)
        db_path: Optional path to DuckDB database file (if None, uses in-memory)
        staging_table: Name for the staging table
        run_ddl: Whether to create the star schema DDL
        create_events_view: Whether to create the events view
        include_parameters: Whether to include parameter data from grids directory
        parameters_path: Path to parameters directory (defaults to grids/ in same dir as events)
        memory_limit: Optional memory limit for DuckDB in MB or GB (e.g., '1GB', '512MB'). Helps prevent kernel crashes.
        
    Returns:
        DuckDB connection with star schema loaded
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB is not available. Install with: pip install duckdb")
    
    # Validate memory_limit format if provided
    if memory_limit is not None:
        if not isinstance(memory_limit, str):
            raise ValueError("memory_limit must be a string (e.g., '1GB', '512MB')")
        # Validate memory limit format - only accept MB or GB
        import re
        if not re.match(r'^\d+(MB|GB)$', memory_limit.upper()):
            raise ValueError("memory_limit must be in MB or GB format (e.g., '1GB', '512MB')")
    
    # Create connection
    if db_path:
        conn = duckdb.connect(db_path)
    else:
        conn = duckdb.connect()
    
    # Configure memory limit if specified
    if memory_limit is not None:
        try:
            conn.execute(f"SET memory_limit='{memory_limit}'")
            print(f"DuckDB memory limit set to: {memory_limit}")
        except Exception as e:
            print(f"Warning: Could not set memory limit '{memory_limit}': {e}")
    
    # Create staging schema if needed
    if '.' in staging_table:
        schema_name = staging_table.split('.')[0]
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
    
    # Load events data with improved schema inference
    # First, discover all columns across all parquet files
    all_columns = _discover_all_columns(conn, input_path)
    
    # Create table with explicit column selection to ensure all columns are included
    if all_columns:
        # Create a UNION ALL query that includes all columns, using COALESCE for missing columns
        import glob
        all_files = glob.glob(input_path)
        
        if len(all_files) == 1:
            # Single file - use direct approach
            column_list = ', '.join(all_columns)
            conn.execute(f"""
                CREATE OR REPLACE TABLE {staging_table} AS
                SELECT {column_list} FROM read_parquet('{input_path}')
                WHERE seed != 3
            """)
        else:
            # Multiple files - create UNION ALL with all columns
            # Use a more robust approach that handles missing columns
            union_parts = []
            for file_path in all_files:
                # Get columns available in this specific file
                try:
                    file_schema = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}') LIMIT 0").df()
                    file_columns = set(file_schema['column_name'].tolist())
                    
                    # Create SELECT statement with all discovered columns
                    # Use CASE statements to handle missing columns
                    select_parts = []
                    for col in all_columns:
                        if col in file_columns:
                            select_parts.append(col)
                        else:
                            select_parts.append(f"NULL as {col}")
                    
                    column_list = ', '.join(select_parts)
                    union_parts.append(f"SELECT {column_list} FROM read_parquet('{file_path}') WHERE seed != 3")
                    
                except Exception as e:
                    print(f"  Error processing {os.path.basename(file_path)}: {e}")
                    continue
            
            if union_parts:
                union_query = " UNION ALL ".join(union_parts)
                conn.execute(f"""
                    CREATE OR REPLACE TABLE {staging_table} AS
                    {union_query}
                """)
            else:
                # Fallback if no files could be processed
                conn.execute(f"""
                    CREATE OR REPLACE TABLE {staging_table} AS
                    SELECT * FROM read_parquet('{input_path}')
                    WHERE seed != 3
                """)
    else:
        # Fallback to original approach if column discovery fails
        conn.execute(f"""
            CREATE OR REPLACE TABLE {staging_table} AS
            SELECT * FROM read_parquet('{input_path}')
            WHERE seed != 3
        """)
    
    if include_parameters:
        # Determine parameters path
        if parameters_path is None:
            # Extract base directory from input_path and look for grids/ subdirectory
            # Handle glob patterns by finding the first matching file and getting its parent
            import glob
            
            # Try different glob patterns to find matching files
            patterns_to_try = [
                input_path,  # Original pattern
                input_path.replace('**', '*'),  # Replace ** with * for Python glob
                input_path.replace('**', 'grid_id=*/seed=*'),  # More specific pattern
            ]
            
            matching_files = []
            for pattern in patterns_to_try:
                matching_files = glob.glob(pattern)
                if matching_files:
                    break
            
            if matching_files:
                # Get the parent directory of the first matching file, then go up to find grids/
                first_file = Path(matching_files[0])
                # Go up from events/grid_id=xxx/seed=xxx to results_parquet, then to grids/
                # events/grid_id=xxx/seed=xxx -> events -> results_parquet -> grids
                base_dir = first_file.parent.parent.parent.parent  # events/grid_id=xxx/seed=xxx -> results_parquet
                parameters_path = str(base_dir / "grids")
            else:
                # Fallback: assume input_path is a directory pattern
                base_dir = Path(input_path).parent
                parameters_path = str(base_dir / "grids")
        
        # Load parameter data
        _load_parameter_data(conn, parameters_path)
    
    if run_ddl:
        _create_star_schema_ddl(conn, staging_table)
    
    if create_events_view:
        _create_events_view(conn)
        _create_workload_events_view(conn)
    
    return conn


def _discover_all_columns(conn: duckdb.DuckDBPyConnection, input_path: str) -> List[str]:
    """
    Discover all columns across all parquet files in the input path.
    
    Args:
        conn: DuckDB connection
        input_path: Path to parquet files (glob pattern supported)
        
    Returns:
        List of all unique column names found across all files
    """
    try:
        import glob
        
        # Get all matching files
        all_files = glob.glob(input_path)
        if not all_files:
            print(f"Warning: No files found matching pattern: {input_path}")
            return []
        
        print(f"Found {len(all_files)} files matching pattern: {input_path}")
        
        # Discover columns from each file individually
        all_columns = set()
        for file_path in all_files:
            try:
                # Get schema from individual file
                schema_query = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}') LIMIT 0").df()
                if schema_query is not None and not schema_query.empty:
                    file_columns = set(schema_query['column_name'].tolist())
                    all_columns.update(file_columns)
                    print(f"  {os.path.basename(file_path)}: {len(file_columns)} columns")
                else:
                    print(f"  {os.path.basename(file_path)}: No columns found")
            except Exception as e:
                print(f"  {os.path.basename(file_path)}: Error - {e}")
        
        # Convert to sorted list for consistent ordering
        columns_list = sorted(list(all_columns))
        print(f"Total unique columns discovered: {len(columns_list)}")
        print(f"All columns: {columns_list}")
        
        return columns_list
        
    except Exception as e:
        print(f"Error in column discovery: {e}")
        return []


def _load_parameter_data(conn: duckdb.DuckDBPyConnection, parameters_path: str) -> None:
    """
    Load parameter data from JSON files into DuckDB.
    
    Args:
        conn: DuckDB connection
        parameters_path: Path to the grids directory containing parameter JSON files
    """
    if not os.path.exists(parameters_path):
        print(f"Warning: Parameters path {parameters_path} does not exist. Skipping parameter loading.")
        return
    
    # Find all parameter JSON files
    param_files = glob.glob(os.path.join(parameters_path, "grid_id=*/params.json"))
    
    if not param_files:
        print(f"Warning: No parameter files found in {parameters_path}")
        return
    
    # Load parameter data into a list of dictionaries
    param_data = []
    for param_file in param_files:
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
            
            # Extract grid_id from the file path
            grid_id = os.path.basename(os.path.dirname(param_file)).replace("grid_id=", "")
            params['grid_id'] = grid_id
            
            param_data.append(params)
        except Exception as e:
            print(f"Warning: Could not load parameters from {param_file}: {e}")
    
    if not param_data:
        print("Warning: No parameter data could be loaded")
        return
    
    # Convert to DataFrame and create table
    param_df = pd.DataFrame(param_data)
    
    # Create parameters table
    conn.execute("DROP TABLE IF EXISTS analytics.dim_parameters")
    conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    
    # Register the DataFrame and create table
    conn.register('param_df', param_df)
    conn.execute("""
        CREATE TABLE analytics.dim_parameters AS
        SELECT * FROM param_df
    """)
    
    print(f"Loaded {len(param_data)} parameter records into analytics.dim_parameters")


def _create_star_schema_ddl(conn: duckdb.DuckDBPyConnection, staging_table: str) -> None:
    """
    Create star schema DDL for analytics.
    
    Args:
        conn: DuckDB connection
        staging_table: Name of the staging table
    """
    # Create analytics schema
    conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    
    # First, get the actual columns and their types from the staging table
    columns_info = conn.execute(f"DESCRIBE {staging_table}").df()
    available_columns = set(columns_info['column_name'].tolist())
    
    # Create a mapping of column names to their types
    column_types = {}
    for _, row in columns_info.iterrows():
        column_types[row['column_name']] = row['column_type']
    
    # Create dimension tables
    conn.execute(f"""
        CREATE OR REPLACE TABLE analytics.dim_run AS
        SELECT DISTINCT 
            grid_id,
            seed,
            grid_id || '_' || seed::VARCHAR as run_id
        FROM {staging_table}
    """)
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE analytics.dim_event_type AS
        SELECT DISTINCT 
            op as event_type,
            op as event_type_description
        FROM {staging_table}
        WHERE op IS NOT NULL
    """)
    
    # Create fact table with all available columns
    fact_columns = ["grid_id", "seed", "event_id", "op as event_type"]
    
    # Add all other columns dynamically (excluding the ones already included)
    excluded_columns = {"grid_id", "seed", "event_id", "op"}
    for column in available_columns:
        if column not in excluded_columns:
            fact_columns.append(column)
    
    print(f"Creating fact table with {len(fact_columns)} columns: {fact_columns}")
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE analytics.fact_event AS
        SELECT 
            {', '.join(fact_columns)}
        FROM {staging_table}
    """)
    
    # Create run summary view with dynamic aggregations
    agg_cols = ["COUNT(fe.event_id) as total_events"]
    agg_cols.append("SUM(CASE WHEN fe.event_type = 'insert' THEN 1 ELSE 0 END) as insert_events")
    agg_cols.append("SUM(CASE WHEN fe.event_type = 'delete' THEN 1 ELSE 0 END) as delete_events")
    
    # Add aggregations for numeric columns dynamically - check types first
    numeric_columns = ["regret", "loss", "acc", "step", "timestamp", "sample_id", "segment_id", 
                      "noise", "rho_step", "rho_spent", "sigma_step", "sigma_delete"]
    boolean_columns = ["sc_stable"]
    
    for col in numeric_columns:
        if col in available_columns:
            col_type = column_types.get(col, '').upper()
            
            # Only apply numeric aggregations to actual numeric types
            if any(numeric_type in col_type for numeric_type in ['INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'HUGEINT', 
                                                               'DECIMAL', 'NUMERIC', 'DOUBLE', 'REAL', 'FLOAT']):
                if col in ["timestamp"]:
                    # For timestamp columns, add min/max
                    agg_cols.extend([
                        f"MIN(fe.{col}) as start_{col}",
                        f"MAX(fe.{col}) as end_{col}"
                    ])
                else:
                    # For other numeric columns, add avg/min/max
                    agg_cols.extend([
                        f"AVG(fe.{col}) as avg_{col}",
                        f"MIN(fe.{col}) as min_{col}", 
                        f"MAX(fe.{col}) as max_{col}"
                    ])
            else:
                print(f"  Skipping numeric aggregations for {col} (type: {col_type})")
    
    # Handle boolean columns separately (no AVG aggregation)
    for col in boolean_columns:
        if col in available_columns:
            col_type = column_types.get(col, '').upper()
            if 'BOOLEAN' in col_type or 'BOOL' in col_type:
                agg_cols.extend([
                    f"MIN(fe.{col}) as min_{col}", 
                    f"MAX(fe.{col}) as max_{col}",
                    f"SUM(CASE WHEN fe.{col} THEN 1 ELSE 0 END) as count_true_{col}"
                ])
            else:
                print(f"  Skipping boolean aggregations for {col} (type: {col_type})")
    
    conn.execute(f"""
        CREATE OR REPLACE VIEW analytics.v_run_summary AS
        SELECT 
            r.grid_id,
            r.seed,
            r.run_id,
            {', '.join(agg_cols)}
        FROM analytics.dim_run r
        LEFT JOIN analytics.fact_event fe ON r.grid_id = fe.grid_id AND r.seed = fe.seed
        GROUP BY r.grid_id, r.seed, r.run_id
    """)
    
    # Create enhanced run summary view with parameters if available
    # Check if dim_parameters table exists
    try:
        conn.execute("SELECT COUNT(*) FROM analytics.dim_parameters LIMIT 1")
        params_exist = True
    except:
        params_exist = False
    
    if params_exist:
        conn.execute("""
            CREATE OR REPLACE VIEW analytics.v_run_summary_with_params AS
            SELECT 
                rs.*,
                p.algo,
                p.dataset,
                p.delete_ratio,
                p.eps_total,
                p.delta_total,
                p.rho_total,
                p.lambda_,
                p.target_lambda,
                p.target_D,
                p.target_G,
                p.target_C,
                p.target_c,
                p.strong_convexity,
                p.enable_oracle,
                p.regret_comparator_mode,
                p.drift_mode,
                p.adaptive_privacy,
                p.adaptive_geometry
            FROM analytics.v_run_summary rs
            LEFT JOIN analytics.dim_parameters p ON rs.grid_id = p.grid_id
        """)
    else:
        # Create a view without parameters
        conn.execute("""
            CREATE OR REPLACE VIEW analytics.v_run_summary_with_params AS
            SELECT 
                rs.*,
                NULL as algo,
                NULL as dataset,
                NULL as delete_ratio,
                NULL as eps_total,
                NULL as delta_total,
                NULL as rho_total,
                NULL as lambda_,
                NULL as target_lambda,
                NULL as target_D,
                NULL as target_G,
                NULL as target_C,
                NULL as target_c,
                NULL as strong_convexity,
                NULL as enable_oracle,
                NULL as regret_comparator_mode,
                NULL as drift_mode,
                NULL as adaptive_privacy,
                NULL as adaptive_geometry
            FROM analytics.v_run_summary rs
        """)


def _create_events_view(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create events view with parameter data joined.
    
    Args:
        conn: DuckDB connection
    """
    # Check if dim_parameters table exists
    try:
        conn.execute("SELECT COUNT(*) FROM analytics.dim_parameters LIMIT 1")
        params_exist = True
    except:
        params_exist = False
    
    if params_exist:
        conn.execute("""
            CREATE OR REPLACE VIEW analytics.v_events_with_params AS
            SELECT 
                fe.*,
                p.algo,
                p.dataset,
                p.delete_ratio,
                p.eps_total,
                p.delta_total,
                p.rho_total,
                p.lambda_,
                p.target_lambda,
                p.target_D,
                p.target_G,
                p.target_C,
                p.target_c,
                p.strong_convexity,
                p.enable_oracle,
                p.regret_comparator_mode,
                p.drift_mode,
                p.adaptive_privacy,
                p.adaptive_geometry
            FROM analytics.fact_event fe
            LEFT JOIN analytics.dim_parameters p ON fe.grid_id = p.grid_id
        """)
    else:
        # Create a view without parameters
        conn.execute("""
            CREATE OR REPLACE VIEW analytics.v_events_with_params AS
            SELECT 
                fe.*,
                NULL as algo,
                NULL as dataset,
                NULL as delete_ratio,
                NULL as eps_total,
                NULL as delta_total,
                NULL as rho_total,
                NULL as lambda_,
                NULL as target_lambda,
                NULL as target_D,
                NULL as target_G,
                NULL as target_C,
                NULL as target_c,
                NULL as strong_convexity,
                NULL as enable_oracle,
                NULL as regret_comparator_mode,
                NULL as drift_mode,
                NULL as adaptive_privacy,
                NULL as adaptive_geometry
            FROM analytics.fact_event fe
        """)


def _create_workload_events_view(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create workload-only events view with regret metrics computed only from workload phase onwards.
    
    The workload phase begins at the first event with phase='workload' per run. This view provides:
    - Running workload-only cumulative regret
    - Running workload-only event count  
    - Running workload-only average regret
    
    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE OR REPLACE VIEW analytics.v_events_workload AS
        WITH workload_boundaries AS (
            -- Find workload phase start for each run (grid_id, seed) using phase column
            SELECT 
                grid_id,
                seed,
                COALESCE(
                    MIN(CASE WHEN phase = 'workload' THEN event_id END),
                    -- Fallback: Use N_gamma from calibration if available
                    MIN(CASE WHEN N_gamma IS NOT NULL AND event_id >= N_gamma THEN event_id END),
                    -- Final fallback: Start of data (no workload boundary found)
                    MIN(event_id)
                ) AS workload_start_event_id
            FROM analytics.fact_event
            GROUP BY grid_id, seed
        ),
        workload_baseline AS (
            -- Get cumulative regret at workload phase start (baseline to subtract)
            SELECT 
                wb.grid_id,
                wb.seed,
                wb.workload_start_event_id,
                COALESCE(fe.cum_regret, 0.0) AS workload_baseline_cum_regret,
                COALESCE(fe.noise_regret_cum, 0.0) AS workload_baseline_noise_regret,
                COALESCE(fe.cum_regret_with_noise, 0.0) AS workload_baseline_cum_regret_with_noise
            FROM workload_boundaries wb
            LEFT JOIN analytics.fact_event fe ON 
                wb.grid_id = fe.grid_id 
                AND wb.seed = fe.seed 
                AND wb.workload_start_event_id = fe.event_id
        )
        SELECT 
            fe.*,
            wb.workload_start_event_id,
            wb.workload_baseline_cum_regret,
            wb.workload_baseline_noise_regret,
            wb.workload_baseline_cum_regret_with_noise,
            -- Workload-only cumulative regret (subtract baseline)
            GREATEST(0.0, COALESCE(fe.cum_regret, 0.0) - wb.workload_baseline_cum_regret) AS workload_cum_regret,
            GREATEST(0.0, COALESCE(fe.noise_regret_cum, 0.0) - wb.workload_baseline_noise_regret) AS workload_noise_regret_cum,
            GREATEST(0.0, COALESCE(fe.cum_regret_with_noise, 0.0) - wb.workload_baseline_cum_regret_with_noise) AS workload_cum_regret_with_noise,
            -- Workload-only event count (events since workload start)
            GREATEST(1, fe.event_id - wb.workload_start_event_id + 1) AS workload_events_seen,
            -- Workload-only average regret
            CASE 
                WHEN fe.event_id >= wb.workload_start_event_id THEN
                    GREATEST(0.0, COALESCE(fe.cum_regret, 0.0) - wb.workload_baseline_cum_regret) / 
                    GREATEST(1, fe.event_id - wb.workload_start_event_id + 1)
                ELSE NULL  -- Pre-workload events don't have workload avg regret
            END AS workload_avg_regret,
            CASE 
                WHEN fe.event_id >= wb.workload_start_event_id THEN
                    GREATEST(0.0, COALESCE(fe.cum_regret_with_noise, 0.0) - wb.workload_baseline_cum_regret_with_noise) / 
                    GREATEST(1, fe.event_id - wb.workload_start_event_id + 1)
                ELSE NULL  -- Pre-workload events don't have workload avg regret
            END AS workload_avg_regret_with_noise,
            -- Flag indicating if this event is in workload phase
            CASE WHEN fe.event_id >= wb.workload_start_event_id THEN TRUE ELSE FALSE END AS is_workload_phase
        FROM analytics.fact_event fe
        INNER JOIN workload_baseline wb ON fe.grid_id = wb.grid_id AND fe.seed = wb.seed
        ORDER BY fe.grid_id, fe.seed, fe.event_id
    """)


def query_events_with_params(
    conn: duckdb.DuckDBPyConnection, 
    where_clause: str = "1=1", 
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Query events with parameter data joined.
    
    Args:
        conn: DuckDB connection
        where_clause: SQL WHERE clause
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with query results
    """
    query = f"SELECT * FROM analytics.v_events_with_params WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    return conn.execute(query).df()


def query_runs_with_params(
    conn: duckdb.DuckDBPyConnection,
    where_clause: str = "1=1",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Query run summaries with parameter data joined.
    
    Args:
        conn: DuckDB connection
        where_clause: SQL WHERE clause
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with query results
    """
    query = f"SELECT * FROM analytics.v_run_summary_with_params WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    return conn.execute(query).df()


def query_workload_events(
    conn: duckdb.DuckDBPyConnection, 
    where_clause: str = "1=1", 
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Query workload-only events with optional filtering.
    
    This function queries events from the workload phase onwards, where the workload phase
    is determined by the explicit 'phase' column in the event logs. Events with phase='workload'
    are considered part of the workload phase.
    
    Args:
        conn: DuckDB connection
        where_clause: SQL WHERE clause
        limit: Optional limit on number of rows
        
    Returns:
        Pandas DataFrame with workload-only regret metrics
    """
    query = f"SELECT * FROM analytics.v_events_workload WHERE {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    return conn.execute(query).df()


def query_workload_regret_analysis(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Query workload regret analysis with summary statistics per run.
    
    This function provides workload-only regret analysis where the workload phase
    is determined by the explicit 'phase' column in the event logs. The analysis
    compares workload-only regret metrics against total regret metrics.
    
    Args:
        conn: DuckDB connection
        
    Returns:
        Pandas DataFrame with workload regret analysis per run
    """
    return conn.execute("""
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
        FROM analytics.v_events_workload
        GROUP BY grid_id, seed, workload_start_event_id
        ORDER BY grid_id, seed
    """).df()
