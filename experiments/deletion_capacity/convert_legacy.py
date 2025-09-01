import argparse
import os
import sys

# Add project root to path for exp_engine imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_engine.cli import convert

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("parquet_out")
    ap.add_argument("--granularity", choices=["seed","event","both"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Delegate to exp_engine CLI
    if args.granularity in ("seed","both"):
        convert(args.csv_dir, args.parquet_out, granularity="seed", dry_run=args.dry_run)
    if args.granularity in ("event","both"):
        convert(args.csv_dir, args.parquet_out, granularity="event", dry_run=args.dry_run)