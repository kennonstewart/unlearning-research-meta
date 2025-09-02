#!/usr/bin/env python3
"""
Command line interface for exp_engine operations (event-level Parquet only).
"""

import argparse
import os
import sys

# Add exp_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.duck import create_connection_and_views, query_events


def cmd_query(args):
    """Query Parquet datasets using DuckDB (events only)."""
    if not os.path.exists(args.base_out):
        print(f"Error: Base output directory {args.base_out} does not exist")
        return

    conn = create_connection_and_views(args.base_out)
    if args.table == "events":
        df = query_events(conn, args.where, args.limit)
    else:
        # Custom SQL
        df = conn.execute(args.table).df()

    if args.output:
        df.to_parquet(args.output, index=False) if args.output.endswith(".parquet") else df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    else:
        print(df.to_string(index=False))


def cmd_info(args):
    """Show information about Parquet datasets (events only)."""
    if not os.path.exists(args.base_out):
        print(f"Error: Base output directory {args.base_out} does not exist")
        return

    conn = create_connection_and_views(args.base_out)

    print("Dataset Information")
    print("=" * 40)

    events_path = os.path.join(args.base_out, "events")
    if os.path.exists(events_path):
        try:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            grids = conn.execute("SELECT COUNT(DISTINCT grid_id) FROM events").fetchone()[0]
            seeds = conn.execute("SELECT COUNT(DISTINCT seed) FROM events").fetchone()[0]
            print(f"Events: {count} records across {grids} grids and {seeds} seeds")
        except Exception as e:
            print(f"Events: Error reading ({e})")
    else:
        print("Events: No data")

    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        view_names = [row[0] for row in tables]
        print(f"Available views: {', '.join(view_names)}")
    except Exception:
        pass


def build_parser():
    p = argparse.ArgumentParser(prog="exp_engine")
    sub = p.add_subparsers(dest="cmd")

    q = sub.add_parser("query", help="Query event-level Parquet via DuckDB")
    q.add_argument("--base-out", default="results_parquet", dest="base_out")
    q.add_argument("--table", default="events", help='Either "events" or a custom SQL string')
    q.add_argument("--where", default="1=1", help="WHERE clause for events")
    q.add_argument("--limit", type=int, default=50)
    q.add_argument("--output", help="Optional output file (.csv or .parquet)")
    q.set_defaults(func=cmd_query)

    i = sub.add_parser("info", help="Show dataset information")
    i.add_argument("--base-out", default="results_parquet", dest="base_out")
    i.set_defaults(func=cmd_info)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()