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