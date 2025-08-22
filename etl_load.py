#!/usr/bin/env python3
"""
ETL script to load experiment CSV files into PostgreSQL database.
Usage: python etl_load.py --csv-dir results/deletion_capacity --dsn postgresql://unlearning:unlearning@localhost/unlearning_db
"""

import os
import sys
import argparse
import hashlib
import glob
import json
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from urllib.parse import urlparse


def generate_run_id(grid_id: str, seed: int) -> str:
    """Generate a unique run_id from grid_id and seed using SHA1 hash."""
    combined = f"{grid_id}_{seed}"
    return hashlib.sha1(combined.encode()).hexdigest()


def parse_dsn(dsn: str) -> dict:
    """Parse PostgreSQL connection string into components."""
    parsed = urlparse(dsn)
    return {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path[1:],  # Remove leading slash
        'user': parsed.username,
        'password': parsed.password
    }


def load_csv_file(csv_path: str) -> tuple:
    """Load a CSV file and extract metadata."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"Warning: Empty CSV file {csv_path}")
            return None, None
        
        # Extract metadata from the first row (all rows should have same metadata)
        metadata = df.iloc[0].to_dict()
        
        # Handle potential column name mapping for C_hat
        if 'C_hat' in df.columns:
            df = df.rename(columns={'C_hat': 'C_hat_upper'})
            metadata['C_hat_upper'] = metadata.pop('C_hat', None)
        
        return df, metadata
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None


def process_csv_directory(csv_dir: str) -> list:
    """Find and process all CSV files in the directory structure."""
    csv_files = []
    
    # Look for seed CSV files in the typical structure
    pattern = os.path.join(csv_dir, "**", "seed_*.csv")
    found_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(found_files)} CSV files in {csv_dir}")
    
    processed_data = []
    for csv_file in found_files:
        df, metadata = load_csv_file(csv_file)
        if df is not None and metadata is not None:
            # Generate run_id
            grid_id = metadata.get('grid_id', 'unknown')
            seed = metadata.get('seed', 0)
            run_id = generate_run_id(grid_id, seed)
            
            # Add run_id to all rows
            df['run_id'] = run_id
            
            processed_data.append((df, metadata, run_id))
            print(f"Processed {csv_file}: {len(df)} rows, run_id={run_id[:8]}...")
    
    return processed_data


def insert_data_to_db(processed_data: list, conn_params: dict):
    """Insert processed data into PostgreSQL database."""
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()
    
    try:
        print("Inserting data into PostgreSQL...")
        
        # Prepare data for batch insert
        rows_to_insert = []
        
        for df, metadata, run_id in processed_data:
            # For simplicity, we'll insert one row per run (seed) with the metadata
            # In a more sophisticated approach, we'd insert individual events
            
            row_data = {
                'run_id': run_id,
                'grid_id': metadata.get('grid_id'),
                'seed': int(metadata.get('seed', 0)),
                'gamma_bar': metadata.get('gamma_bar'),
                'gamma_split': metadata.get('gamma_split'),
                'accountant': metadata.get('accountant'),
                'G_hat': metadata.get('G_hat'),
                'D_hat': metadata.get('D_hat'),
                'c_hat': metadata.get('c_hat'),
                'C_hat_upper': metadata.get('C_hat_upper'),
                'lambda_est': metadata.get('lambda_est'),
                'S_scalar': metadata.get('S_scalar'),
                'sigma_step_theory': metadata.get('sigma_step_theory'),
                'N_star_live': metadata.get('N_star_live'),
                'N_star_theory': metadata.get('N_star_theory'),
                'm_theory_live': metadata.get('m_theory_live'),
                'blocked_reason': metadata.get('blocked_reason'),
                'eta_t': metadata.get('eta_t'),
                'path_type': metadata.get('path_type'),
                'rotate_angle': metadata.get('rotate_angle'),
                'drift_rate': metadata.get('drift_rate'),
                'feature_scale': metadata.get('feature_scale'),
                'w_scale': metadata.get('w_scale'),
                'fix_w_norm': bool(metadata.get('fix_w_norm')) if metadata.get('fix_w_norm') is not None and not pd.isna(metadata.get('fix_w_norm')) else None,
                'noise_std': metadata.get('noise_std'),
                'eps_spent': metadata.get('eps_spent'),
                'eps_remaining': metadata.get('eps_remaining'),
                'delta_total': metadata.get('delta_total'),
                'avg_regret_empirical': metadata.get('avg_regret_empirical'),
                'N_star_emp': int(metadata.get('N_star_emp', 0)) if metadata.get('N_star_emp') is not None else None,
                'm_emp': int(metadata.get('m_emp', 0)) if metadata.get('m_emp') is not None else None,
                'final_acc': metadata.get('final_acc'),
                'total_events': int(metadata.get('total_events', 0)) if metadata.get('total_events') is not None else None,
            }
            
            rows_to_insert.append(row_data)
        
        # Create insert statement
        if rows_to_insert:
            # Get column names (excluding those that might be None)
            sample_row = rows_to_insert[0]
            columns = [k for k, v in sample_row.items() if v is not None]
            
            # Prepare values for each row
            values_list = []
            for row in rows_to_insert:
                values = tuple(row.get(col) for col in columns)
                values_list.append(values)
            
            # Build INSERT statement
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            insert_sql = f"""
                INSERT INTO fact_event ({columns_str}) 
                VALUES %s
                ON CONFLICT DO NOTHING
            """
            
            # Execute batch insert using execute_values
            execute_values(
                cursor, 
                insert_sql, 
                values_list, 
                template=None, 
                page_size=100
            )
            
            conn.commit()
            print(f"Successfully inserted {len(values_list)} rows into fact_event table")
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def verify_data(conn_params: dict):
    """Verify that data was loaded correctly."""
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()
    
    try:
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM fact_event;")
        total_count = cursor.fetchone()[0]
        print(f"Total rows in fact_event: {total_count}")
        
        # Check unique grids and seeds
        cursor.execute("SELECT COUNT(DISTINCT grid_id), COUNT(DISTINCT seed) FROM fact_event;")
        grid_count, seed_count = cursor.fetchone()
        print(f"Unique grids: {grid_count}, Unique seeds: {seed_count}")
        
        # Sample data
        cursor.execute("SELECT run_id, seed, avg_regret_empirical FROM fact_event WHERE run_id IS NOT NULL LIMIT 3;")
        sample_rows = cursor.fetchall()
        print("Sample data:")
        for row in sample_rows:
            run_id_short = row[0][:8] if row[0] else 'N/A'
            seed = row[1] if row[1] is not None else 'N/A'
            regret = f"{row[2]:.3f}" if row[2] is not None else 'N/A'
            print(f"  Run {run_id_short}...: seed={seed}, regret={regret}")
        
    except Exception as e:
        print(f"Error verifying data: {e}")
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Load experiment CSV files into PostgreSQL')
    parser.add_argument('--csv-dir', help='Directory containing CSV files')
    parser.add_argument('--dsn', required=True, help='PostgreSQL connection string')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing data')
    
    args = parser.parse_args()
    
    # Parse connection parameters
    try:
        conn_params = parse_dsn(args.dsn)
        print(f"Connecting to PostgreSQL at {conn_params['host']}:{conn_params['port']}/{conn_params['database']}")
    except Exception as e:
        print(f"Error parsing DSN: {e}")
        return 1
    
    if args.verify_only:
        verify_data(conn_params)
        return 0
    
    # CSV directory is required for non-verify operations
    if not args.csv_dir:
        print("Error: --csv-dir is required when not using --verify-only")
        return 1
    
    # Check CSV directory exists
    if not os.path.exists(args.csv_dir):
        print(f"Error: CSV directory {args.csv_dir} does not exist")
        return 1
    
    # Process CSV files
    processed_data = process_csv_directory(args.csv_dir)
    
    if not processed_data:
        print("No valid CSV files found to process")
        return 1
    
    # Insert data into database
    try:
        insert_data_to_db(processed_data, conn_params)
        print("ETL process completed successfully!")
        
        # Verify the loaded data
        print("\nVerifying loaded data:")
        verify_data(conn_params)
        
    except Exception as e:
        print(f"ETL process failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())