#!/usr/bin/env python3
"""
Grid search runner for deletion capacity experiments.
Implements the workflow described in AGENTS.md.
"""

import itertools
import yaml
import os
import sys
import argparse
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import Config
from runner import ExperimentRunner


def load_grid(grid_file: str) -> Dict[str, List[Any]]:
    """Load parameter grid from YAML file."""
    with open(grid_file, 'r') as f:
        grid = yaml.safe_load(f)
    
    # Validate that gamma_learn and gamma_priv have same length for pairing
    if 'gamma_learn' in grid and 'gamma_priv' in grid:
        if len(grid['gamma_learn']) != len(grid['gamma_priv']):
            raise ValueError("gamma_learn and gamma_priv must have same length for pairing")
    
    return grid


def generate_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid.
    
    Special handling: gamma_learn and gamma_priv are paired, not crossed.
    """
    # Separate gamma pairs from other parameters
    gamma_pairs = []
    other_params = {}
    
    if 'gamma_learn' in grid and 'gamma_priv' in grid:
        gamma_pairs = list(zip(grid['gamma_learn'], grid['gamma_priv']))
        other_params = {k: v for k, v in grid.items() 
                       if k not in ['gamma_learn', 'gamma_priv']}
    else:
        # No gamma pairs, treat all as independent
        other_params = grid
    
    # Generate combinations
    if gamma_pairs:
        # Create combinations with gamma pairs
        other_combos = list(itertools.product(*other_params.values()))
        other_keys = list(other_params.keys())
        
        combinations = []
        for gamma_learn, gamma_priv in gamma_pairs:
            for other_combo in other_combos:
                combo_dict = {'gamma_learn': gamma_learn, 'gamma_priv': gamma_priv}
                combo_dict.update(dict(zip(other_keys, other_combo)))
                combinations.append(combo_dict)
    else:
        # No gamma pairs, simple Cartesian product
        combos = list(itertools.product(*grid.values()))
        param_names = list(grid.keys())
        combinations = [dict(zip(param_names, combo)) for combo in combos]
    
    return combinations


def create_grid_id(params: Dict[str, Any]) -> str:
    """Create a unique identifier for this parameter combination."""
    # Format: split_0.7-0.3_q0.95_k10_default_eps1.0
    gamma_l = params.get('gamma_learn', 1.0)
    gamma_p = params.get('gamma_priv', 0.5)
    quantile = params.get('quantile', 0.95)
    delete_ratio = params.get('delete_ratio', 10)
    accountant = params.get('accountant', 'default')
    eps_total = params.get('eps_total', 1.0)
    
    return f"split_{gamma_l:.1f}-{gamma_p:.1f}_q{quantile:.2f}_k{delete_ratio:.0f}_{accountant}_eps{eps_total:.1f}"


def run_single_experiment(params: Dict[str, Any], seed: int, base_out_dir: str) -> str:
    """Run a single experiment with given parameters and seed."""
    # Create config from parameters
    config_kwargs = params.copy()
    config_kwargs['seeds'] = 1  # Single seed per run
    
    # Set output directory for this specific run
    grid_id = create_grid_id(params)
    run_out_dir = os.path.join(base_out_dir, "sweep", grid_id)
    os.makedirs(run_out_dir, exist_ok=True)
    config_kwargs['out_dir'] = run_out_dir
    
    # Set some defaults for faster testing
    config_kwargs.setdefault('max_events', 1000)  # Reduced for testing
    config_kwargs.setdefault('bootstrap_iters', 50)  # Reduced for testing
    
    try:
        # Create config and runner
        cfg = Config.from_cli_args(**config_kwargs)
        runner = ExperimentRunner(cfg)
        
        # Run for this specific seed
        runner.run_single_seed(seed)
        
        # The runner should create a CSV file directly in the out_dir
        # Look for files matching the expected pattern
        csv_pattern = os.path.join(run_out_dir, f"seed_{seed:03d}_*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if csv_files:
            # File already in correct location, just rename to standardized format
            src_file = csv_files[0]
            dst_file = os.path.join(run_out_dir, f"seed_{seed:03d}.csv")
            if src_file != dst_file:
                shutil.move(src_file, dst_file)
            return dst_file
        else:
            print(f"Warning: No CSV file found for seed {seed} with pattern {csv_pattern}")
            return None
            
    except Exception as e:
        print(f"Error running experiment for seed {seed} with params {params}: {e}")
        return None


def run_parameter_combination(params: Dict[str, Any], seeds: List[int], base_out_dir: str, parallel: int = 1) -> List[str]:
    """Run all seeds for a single parameter combination."""
    grid_id = create_grid_id(params)
    print(f"\n=== Running grid cell: {grid_id} ===")
    print(f"Parameters: {params}")
    
    csv_paths = []
    
    if parallel == 1:
        # Sequential execution
        for seed in seeds:
            result = run_single_experiment(params, seed, base_out_dir)
            if result:
                csv_paths.append(result)
    else:
        # Parallel execution
        with mp.Pool(parallel) as pool:
            run_func = partial(run_single_experiment, params, base_out_dir=base_out_dir)
            results = pool.map(run_func, seeds)
            csv_paths = [r for r in results if r is not None]
    
    print(f"Completed {len(csv_paths)}/{len(seeds)} seeds for {grid_id}")
    return csv_paths


def aggregate_results(sweep_dir: str) -> str:
    """Aggregate all CSV results into a master file."""
    print("\n=== Aggregating results ===")
    
    # Find all seed CSV files
    csv_files = glob.glob(os.path.join(sweep_dir, "*", "seed_*.csv"))
    
    if not csv_files:
        print("Warning: No CSV files found to aggregate")
        return None
    
    print(f"Found {len(csv_files)} CSV files to aggregate")
    
    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract grid parameters from directory name
            grid_id = os.path.basename(os.path.dirname(csv_file))
            seed_file = os.path.basename(csv_file)
            seed = int(seed_file.split('_')[1].split('.')[0])
            
            # Add metadata columns
            df['grid_id'] = grid_id
            df['seed'] = seed
            
            # Parse grid_id to extract parameters
            # Format: split_0.7-0.3_q0.95_k10_default
            parts = grid_id.split('_')
            if len(parts) >= 4:
                # Parse split
                split_part = parts[1]  # "0.7-0.3"
                gamma_parts = split_part.split('-')
                if len(gamma_parts) == 2:
                    df['gamma_learn_grid'] = float(gamma_parts[0])
                    df['gamma_priv_grid'] = float(gamma_parts[1])
                
                # Parse quantile
                q_part = parts[2]  # "q0.95"
                if q_part.startswith('q'):
                    df['quantile_grid'] = float(q_part[1:])
                
                # Parse delete ratio
                k_part = parts[3]  # "k10"
                if k_part.startswith('k'):
                    df['delete_ratio_grid'] = float(k_part[1:])
                
                # Parse accountant
                if len(parts) >= 5:
                    df['accountant_grid'] = parts[4]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")
    
    if not dfs:
        print("Error: No valid CSV files could be read")
        return None
    
    # Concatenate all dataframes
    master_df = pd.concat(dfs, ignore_index=True)
    
    # Write master CSV
    master_path = os.path.join(sweep_dir, "all_runs.csv")
    master_df.to_csv(master_path, index=False)
    
    print(f"Aggregated {len(master_df)} rows into {master_path}")
    print(f"Columns: {list(master_df.columns)}")
    
    return master_path


def validate_schema(csv_path: str, expected_accountants: List[str]) -> bool:
    """Validate that aggregated CSV has expected schema."""
    if not csv_path or not os.path.exists(csv_path):
        return False
    
    try:
        df = pd.read_csv(csv_path)
        
        # Base columns that should always be present
        base_cols = {'event', 'op', 'regret', 'acc', 'grid_id', 'seed'}
        
        # Accountant-specific columns
        rdp_cols = {'eps_converted', 'eps_remaining', 'delta_total'}
        legacy_cols = {'eps_spent', 'capacity_remaining'}
        
        actual_cols = set(df.columns)
        
        # Check base columns
        missing_base = base_cols - actual_cols
        if missing_base:
            print(f"Warning: Missing base columns: {missing_base}")
            return False
        
        # Check accountant-specific columns
        for accountant in expected_accountants:
            accountant_rows = df[df['accountant_grid'] == accountant]
            if len(accountant_rows) == 0:
                continue
                
            if accountant == 'rdp':
                missing_rdp = rdp_cols - actual_cols
                if missing_rdp:
                    print(f"Warning: Missing RDP columns: {missing_rdp}")
            elif accountant == 'legacy':
                missing_legacy = legacy_cols - actual_cols
                if missing_legacy:
                    print(f"Warning: Missing legacy columns: {missing_legacy}")
        
        print("Schema validation passed")
        return True
        
    except Exception as e:
        print(f"Schema validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Grid search runner for deletion capacity experiments")
    parser.add_argument("--grid-file", default="grids.yaml", help="YAML file with parameter grid")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--base-out", default="results/grid_2025_01_01", help="Base output directory")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per grid cell")
    parser.add_argument("--dry-run", action="store_true", help="Show combinations without running")
    
    args = parser.parse_args()
    
    # Load parameter grid
    print(f"Loading grid from {args.grid_file}")
    if not os.path.exists(args.grid_file):
        print(f"Error: Grid file {args.grid_file} not found")
        return 1
    
    grid = load_grid(args.grid_file)
    print(f"Grid parameters: {list(grid.keys())}")
    
    # Generate parameter combinations
    combinations = generate_combinations(grid)
    print(f"Generated {len(combinations)} parameter combinations")
    
    if args.dry_run:
        print("\nDry run - parameter combinations:")
        for i, combo in enumerate(combinations):
            grid_id = create_grid_id(combo)
            print(f"{i+1:3d}. {grid_id}: {combo}")
        return 0
    
    # Create base output directory
    os.makedirs(args.base_out, exist_ok=True)
    sweep_dir = os.path.join(args.base_out, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Generate seed list
    seeds = list(range(args.seeds))
    
    # Run all combinations
    print(f"\nRunning {len(combinations)} combinations x {len(seeds)} seeds = {len(combinations) * len(seeds)} total experiments")
    
    all_csv_paths = []
    for i, params in enumerate(combinations):
        print(f"\nProgress: {i+1}/{len(combinations)}")
        csv_paths = run_parameter_combination(params, seeds, args.base_out, args.parallel)
        all_csv_paths.extend(csv_paths)
    
    # Aggregate results
    master_csv = aggregate_results(sweep_dir)
    
    # Validate schema
    expected_accountants = grid.get('accountant', ['legacy'])
    if master_csv:
        validate_schema(master_csv, expected_accountants)
    
    print(f"\nGrid search complete!")
    print(f"Results in: {sweep_dir}")
    print(f"Master CSV: {master_csv}")
    print(f"Total experiments completed: {len(all_csv_paths)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())