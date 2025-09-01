import argparse
import os
import sys

# Add project root to path for exp_engine imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_engine.converter import convert_csv_to_parquet

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("parquet_out")
    ap.add_argument("--granularity", choices=["seed","event","both"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    
    # Delegate to exp_engine converter
    if args.granularity in ("seed","both"):
        convert_csv_to_parquet(args.csv_dir, args.parquet_out, granularity="seed")
        print(f"✓ Converted seed data: {args.csv_dir} -> {args.parquet_out}")
    if args.granularity in ("event","both"):
        convert_csv_to_parquet(args.csv_dir, args.parquet_out, granularity="event")
        print(f"✓ Converted event data: {args.csv_dir} -> {args.parquet_out}")