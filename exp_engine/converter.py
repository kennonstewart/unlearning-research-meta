#!/usr/bin/env python3
"""
Convert legacy CSV sweep results to Parquet format.

This script reads CSV files from grid runs and converts them to the
new Parquet-based experiment engine format.
"""

import argparse
import os
import sys
import glob
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Add exp_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import write_seed_rows, write_event_rows, attach_grid_id


def parse_csv_filename(csv_path: str) -> Dict[str, Any]:
    """Extract parameters from CSV filename.
    
    Expected format: seed_XXX_algo_paramstring.csv
    """
    basename = os.path.basename(csv_path)
    parts = basename.replace('.csv', '').split('_')
    
    params = {}
    
    # Extract seed
    for i, part in enumerate(parts):
        if part == 'seed' and i + 1 < len(parts):
            try:
                params['seed'] = int(parts[i + 1])
            except ValueError:
                pass
            break
    
    # Extract algo
    for i, part in enumerate(parts):
        if part in ['memorypair', 'baseline']:
            params['algo'] = part
            break
    
    return params


def extract_params_from_grid_dir(grid_dir: str) -> Dict[str, Any]:
    """Extract grid parameters from directory structure or config files."""
    params = {}
    
    # Try to find a params.json or config file
    config_files = glob.glob(os.path.join(grid_dir, '*.json'))
    config_files.extend(glob.glob(os.path.join(grid_dir, 'config.yaml')))
    config_files.extend(glob.glob(os.path.join(grid_dir, 'params.yaml')))
    
    if config_files:
        try:
            with open(config_files[0]) as f:
                if config_files[0].endswith('.json'):
                    params.update(json.load(f))
                else:
                    import yaml
                    params.update(yaml.safe_load(f))
        except Exception as e:
            print(f"Warning: Could not read config file {config_files[0]}: {e}")
    
    # Extract from directory name if no config found
    if not params:
        dirname = os.path.basename(grid_dir)
        # Parse directory names like "split_0.7-0.3_q0.95_k10_default"
        parts = dirname.split('_')
        for i, part in enumerate(parts):
            if part == 'split' and i + 1 < len(parts) and '-' in parts[i + 1]:
                # Handle "split" followed by "0.7-0.3" format
                gamma_str = parts[i + 1]
                gamma_parts = gamma_str.split('-')
                if len(gamma_parts) == 2:
                    try:
                        params['gamma_bar'] = float(gamma_parts[0])
                        params['gamma_split'] = float(gamma_parts[1])
                    except ValueError:
                        pass
            elif part.startswith('split') and '-' in part:
                # Handle "split0.7-0.3" format (no underscore)
                gamma_str = part.replace('split', '')
                if gamma_str and '-' in gamma_str:
                    gamma_parts = gamma_str.split('-')
                    if len(gamma_parts) == 2:
                        try:
                            params['gamma_bar'] = float(gamma_parts[0])
                            params['gamma_split'] = float(gamma_parts[1])
                        except ValueError:
                            pass
            elif part.startswith('q'):
                try:
                    params['quantile'] = float(part[1:])
                except ValueError:
                    pass
            elif part.startswith('k'):
                try:
                    params['delete_ratio'] = int(part[1:])
                except ValueError:
                    pass
            elif part in ['default', 'rdp', 'zcdp']:
                params['accountant'] = part
    
    return params


def convert_csv_to_parquet(
    csv_dir: str,
    output_dir: str,
    granularity: str = "seed"
) -> None:
    """Convert CSV files to Parquet format.
    
    Args:
        csv_dir: Directory containing CSV files (can be nested)
        output_dir: Output directory for Parquet files
        granularity: Output granularity - "seed" or "event"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Group by grid directory
    grid_dirs = {}
    for csv_file in csv_files:
        # Find the parent directory that might contain grid info
        parent_dir = os.path.dirname(csv_file)
        
        # Keep going up until we find a meaningful directory name or hit the base
        while parent_dir != csv_dir and not any(x in os.path.basename(parent_dir).lower() 
                                              for x in ['grid', 'split', 'gamma', 'ratio']):
            parent_dir = os.path.dirname(parent_dir)
        
        if parent_dir not in grid_dirs:
            grid_dirs[parent_dir] = []
        grid_dirs[parent_dir].append(csv_file)
    
    print(f"Found {len(grid_dirs)} grid directories")
    
    # Process each grid
    for grid_dir, csv_files in grid_dirs.items():
        print(f"\nProcessing {len(csv_files)} files in {grid_dir}")
        
        # Extract grid parameters
        grid_params = extract_params_from_grid_dir(grid_dir)
        
        # Add default values if missing
        if 'algo' not in grid_params:
            grid_params['algo'] = 'memorypair'
        if 'accountant' not in grid_params:
            grid_params['accountant'] = 'zcdp'
        
        # Process based on granularity
        if granularity == "seed":
            convert_to_seed_level(csv_files, output_dir, grid_params)
        elif granularity == "event":
            convert_to_event_level(csv_files, output_dir, grid_params)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")


def convert_to_seed_level(csv_files: List[str], output_dir: str, grid_params: Dict[str, Any]) -> None:
    """Convert CSV files to seed-level Parquet format."""
    seed_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Extract seed and other params from filename
            file_params = parse_csv_filename(csv_file)
            
            # Create seed summary
            summary = {
                **grid_params,
                **file_params,
                "avg_regret_empirical": df["regret"].mean() if "regret" in df.columns else None,
                "N_star_emp": len(df[df["op"] == "insert"]) if "op" in df.columns else None,
                "m_emp": len(df[df["op"] == "delete"]) if "op" in df.columns else None,
                "final_acc": df["acc"].iloc[-1] if "acc" in df.columns and len(df) > 0 else None,
                "total_events": len(df),
            }
            
            # Add privacy metrics from last row if available
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in ["eps_spent", "capacity_remaining", "rho_spent"]:
                    if col in last_row:
                        summary[col] = last_row[col]
            
            seed_data.append(summary)
            
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
    
    if seed_data:
        write_seed_rows(seed_data, output_dir, grid_params)
        print(f"Wrote {len(seed_data)} seed records")


def convert_to_event_level(csv_files: List[str], output_dir: str, grid_params: Dict[str, Any]) -> None:
    """Convert CSV files to event-level Parquet format."""
    all_events = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Extract seed and other params from filename
            file_params = parse_csv_filename(csv_file)
            
            # Add grid and file params to each row
            for _, row in df.iterrows():
                event = {
                    **grid_params,
                    **file_params,
                    **row.to_dict()
                }
                all_events.append(event)
                
        except Exception as e:
            print(f"Warning: Could not process {csv_file}: {e}")
    
    if all_events:
        write_event_rows(all_events, output_dir, grid_params)
        print(f"Wrote {len(all_events)} event records")


def main():
    parser = argparse.ArgumentParser(description="Convert legacy CSV sweeps to Parquet format")
    parser.add_argument("csv_dir", help="Directory containing CSV files")
    parser.add_argument("output_dir", help="Output directory for Parquet files")
    parser.add_argument("--granularity", choices=["seed", "event"], default="seed",
                       help="Output granularity (default: seed)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print what would be converted without doing it")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_dir):
        print(f"Error: CSV directory {args.csv_dir} does not exist")
        sys.exit(1)
    
    if args.dry_run:
        csv_files = glob.glob(os.path.join(args.csv_dir, "**/*.csv"), recursive=True)
        print(f"Would convert {len(csv_files)} CSV files from {args.csv_dir} to {args.output_dir}")
        print(f"Granularity: {args.granularity}")
        return
    
    convert_csv_to_parquet(args.csv_dir, args.output_dir, args.granularity)
    print(f"\nConversion complete! Results written to {args.output_dir}")


if __name__ == "__main__":
    main()