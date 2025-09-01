#!/usr/bin/env python3
"""
Command line interface for exp_engine operations.
"""

import argparse
import os
import sys
from pathlib import Path

# Add exp_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.duck import create_connection_and_views, query_seeds, query_events


def cmd_convert(args):
    """Convert CSV files to Parquet."""
    from converter import convert_csv_to_parquet
    
    convert_csv_to_parquet(args.csv_dir, args.output_dir, args.granularity)
    print(f"Conversion complete: {args.csv_dir} -> {args.output_dir}")


def cmd_query(args):
    """Query Parquet datasets using DuckDB."""
    if not os.path.exists(args.base_out):
        print(f"Error: Base output directory {args.base_out} does not exist")
        return
    
    conn = create_connection_and_views(args.base_out)
    
    if args.table == "seeds":
        df = query_seeds(conn, args.where, args.limit)
    elif args.table == "events":
        df = query_events(conn, args.where, args.limit)
    else:
        # Custom query
        df = conn.execute(args.table).df()
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        print(df.to_string(index=False))


def cmd_info(args):
    """Show information about Parquet datasets."""
    if not os.path.exists(args.base_out):
        print(f"Error: Base output directory {args.base_out} does not exist")
        return
    
    conn = create_connection_and_views(args.base_out)
    
    print("Dataset Information")
    print("=" * 40)
    
    # Check what datasets exist
    seeds_path = os.path.join(args.base_out, "seeds")
    events_path = os.path.join(args.base_out, "events")
    
    if os.path.exists(seeds_path):
        try:
            count = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
            grids = conn.execute("SELECT COUNT(DISTINCT grid_id) FROM seeds").fetchone()[0]
            print(f"Seeds: {count} records across {grids} grids")
        except Exception as e:
            print(f"Seeds: Error reading ({e})")
    else:
        print("Seeds: No data")
    
    if os.path.exists(events_path):
        try:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] 
            print(f"Events: {count} records")
        except Exception as e:
            print(f"Events: Error reading ({e})")
    else:
        print("Events: No data")
    
    # Show available views
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        view_names = [row[0] for row in tables]
        print(f"Available views: {', '.join(view_names)}")
    except Exception:
        pass


def cmd_rollup(args):
    """Run Snakemake rollup workflow."""
    import subprocess
    
    snakefile = os.path.join(os.path.dirname(__file__), "Snakefile")
    config_file = os.path.join(os.path.dirname(__file__), "snakemake_config.yaml")
    
    cmd = [
        "snakemake", 
        "-s", snakefile,
        "--configfile", config_file,
        f"--config", f"base_out={args.base_out}", f"csv_dir={args.csv_dir}"
    ]
    
    if args.cores:
        cmd.extend(["--cores", str(args.cores)])
    
    if args.dry_run:
        cmd.append("--dry-run")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✓ Rollup complete")
    else:
        print("✗ Rollup failed")


def main():
    parser = argparse.ArgumentParser(description="Exp_engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert CSV to Parquet")
    convert_parser.add_argument("csv_dir", help="Directory containing CSV files")
    convert_parser.add_argument("output_dir", help="Output directory for Parquet files")
    convert_parser.add_argument("--granularity", choices=["seed", "event"], default="seed")
    convert_parser.set_defaults(func=cmd_convert)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query Parquet datasets")
    query_parser.add_argument("base_out", help="Base output directory")
    query_parser.add_argument("table", choices=["seeds", "events"], help="Table to query")
    query_parser.add_argument("--where", default="1=1", help="WHERE clause")
    query_parser.add_argument("--limit", type=int, help="Limit number of results")
    query_parser.add_argument("--output", help="Output CSV file")
    query_parser.set_defaults(func=cmd_query)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("base_out", help="Base output directory")
    info_parser.set_defaults(func=cmd_info)
    
    # Rollup command
    rollup_parser = subparsers.add_parser("rollup", help="Run Snakemake rollup")
    rollup_parser.add_argument("--base-out", default="results/parquet", help="Base output directory")
    rollup_parser.add_argument("--csv-dir", default="experiments/deletion_capacity/results", help="CSV input directory")
    rollup_parser.add_argument("--cores", type=int, default=1, help="Number of cores for Snakemake")
    rollup_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    rollup_parser.set_defaults(func=cmd_rollup)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()