#!/usr/bin/env python3
"""
Unified experiment runner for theory-first deletion capacity experiments.

This script consolidates experiment execution logic, supporting both single runs
and grid searches with full exp_engine integration for Parquet output.

Usage:
    # Single run with theory-first parameters
    python experiment/run.py --target-G 2.0 --target-D 2.0 --rho-total 1.0 --max-events 1000
    
    # Grid search from YAML
    python experiment/run.py --grid-file grids/01_theory_grid.yaml --parallel 4
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from functools import partial
from typing import Any, Dict, List, Optional
import yaml

import numpy as np

# Add parent directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Add code directory for algorithm and data loader
_CODE_DIR = os.path.join(_REPO_ROOT, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# Import local modules
from experiment.config import ExperimentConfig

# Import exp_engine for Parquet output and grid management
from exp_engine.engine.cah import attach_grid_id, grid_hash
from exp_engine.engine.io import write_event_rows

# Import algorithm and data loader
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.accountant import ZCDPAccountant
from data_loader import get_theory_stream, parse_event_record


def load_grid_from_yaml(grid_file: str) -> Dict[str, Any]:
    """Load parameter grid from YAML file.
    
    Args:
        grid_file: Path to YAML grid file
        
    Returns:
        Dictionary with matrix, selectors, limit, etc.
    """
    with open(grid_file, "r") as f:
        raw = yaml.safe_load(f) or {}
    
    # If no structured keys, treat entire file as matrix
    if not isinstance(raw, dict):
        return {"matrix": {}}
    
    if "matrix" not in raw:
        return {"matrix": raw}
    
    return raw


def generate_grid_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid specification.
    
    Args:
        grid: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of parameter dictionaries (one per combination)
    """
    if not grid:
        return []
    
    # Convert all values to lists
    options = {}
    for k, v in grid.items():
        if isinstance(v, list):
            options[k] = v
        else:
            options[k] = [v]
    
    # Generate Cartesian product
    keys = list(options.keys())
    value_lists = [options[k] for k in keys]
    
    combinations = []
    for values in itertools.product(*value_lists):
        combo = dict(zip(keys, values))
        combinations.append(combo)
    
    return combinations


def run_single_experiment(
    params: Dict[str, Any],
    seed: int,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    """Run a single experiment with given parameters and seed.
    
    Args:
        params: Parameter dictionary with grid_id attached
        seed: Random seed for this run
        output_dir: Base output directory for Parquet files
        
    Returns:
        Summary dictionary with results, or None on error
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create config from parameters
    config = ExperimentConfig.from_dict(params)
    
    # Initialize data stream with theory-first parameters
    stream = get_theory_stream(
        dim=config.dim,
        seed=seed,
        T=config.max_events,
        target_G=config.target_G,
        target_D=config.target_D,
        target_c=config.target_c,
        target_C=config.target_C,
        target_lambda=config.target_lambda,
        target_PT=config.target_PT,
        target_ST=config.target_ST,
        accountant=config.accountant,
        rho_total=config.rho_total,
        eps_total=config.eps_total,
        delta_total=config.delta_total,
        path_style=config.path_style,
    )
    
    # Get first event to determine dimension
    first_record = next(stream)
    first_x, first_y, first_meta = parse_event_record(first_record)
    dim = first_x.shape[0]
    
    # Initialize accountant
    accountant = ZCDPAccountant(
        rho_total=config.rho_total,
        delta_total=config.delta_total,
        T=config.max_events,
        gamma=config.gamma_delete,
        lambda_=config.lambda_,
        delta_b=config.delta_b,
        m_max=config.m_max,
    )
    
    # Initialize model
    # MemoryPair takes G, D, c, C directly (these come from theory targets)
    model = MemoryPair(
        dim=dim,
        G=config.target_G,
        D=config.target_D,
        c=config.target_c,
        C=config.target_C,
        accountant=accountant,
        cfg=config,  # Pass config for additional parameters
    )
    
    # Process events
    event_rows = []
    event_count = 0
    
    # Process first event
    x, y = first_x, first_y
    event_type = first_meta.get("op", "insert")
    
    if event_type == "insert":
        prediction = model.insert(x, y)  # Prediction before update
        loss = 0.5 * (prediction - y) ** 2  # Compute loss from prediction
    else:  # delete
        # For delete, we need to get prediction first
        prediction = float(model.theta @ x)
        result = model.delete(x, y)
        loss = 0.5 * (prediction - y) ** 2  # Compute loss from prediction
    
    # Record event
    event_row = {
        "event": event_count,
        "seed": seed,
        "op": event_type,
        "loss": float(loss),
        "prediction": float(prediction),
        "label": float(y),
    }
    
    # Add regret if available
    if hasattr(model, "cumulative_regret"):
        event_row["regret"] = float(model.cumulative_regret)
    
    event_rows.append(event_row)
    event_count += 1
    
    # Process remaining events
    for record in stream:
        if event_count >= config.max_events:
            break
        
        x, y, meta = parse_event_record(record)
        event_type = meta.get("op", "insert")
        
        if event_type == "insert":
            prediction = model.insert(x, y)  # Prediction before update
            loss = 0.5 * (prediction - y) ** 2  # Compute loss from prediction
        else:  # delete
            prediction = float(model.theta @ x)
            result = model.delete(x, y)
            loss = 0.5 * (prediction - y) ** 2
        
        # Record event
        event_row = {
            "event": event_count,
            "seed": seed,
            "op": event_type,
            "loss": float(loss),
            "prediction": float(prediction),
            "label": float(y),
        }
        
        # Add regret if available
        if hasattr(model, "cumulative_regret"):
            event_row["regret"] = float(model.cumulative_regret)
        
        event_rows.append(event_row)
        event_count += 1
    
    # Write event-level results to Parquet using exp_engine
    if event_rows and output_dir:
        try:
            write_event_rows(event_rows, output_dir, params)
        except Exception as e:
            print(f"Warning: Failed to write Parquet events for seed {seed}: {e}")
    
    # Create summary
    summary = {
        "seed": seed,
        "events_processed": event_count,
        "final_regret": float(model.cumulative_regret) if hasattr(model, "cumulative_regret") else 0.0,
    }
    
    return summary


def run_grid_cell(
    params: Dict[str, Any],
    seeds: List[int],
    output_dir: str,
    parallel: int = 1,
) -> List[Dict[str, Any]]:
    """Run all seeds for a single grid cell (parameter combination).
    
    Args:
        params: Parameter dictionary (grid_id will be attached)
        seeds: List of random seeds to run
        output_dir: Base output directory for Parquet files
        parallel: Number of parallel workers (1 = sequential)
        
    Returns:
        List of summary dictionaries (one per seed)
    """
    # Attach grid_id using exp_engine
    params_with_grid = attach_grid_id(params)
    grid_id = params_with_grid["grid_id"]
    
    print(f"\n=== Running grid cell: {grid_id} ===")
    print(f"Parameters: {params}")
    print(f"Seeds: {len(seeds)}")
    
    summaries = []
    
    if parallel == 1:
        # Sequential execution
        for seed in seeds:
            summary = run_single_experiment(params_with_grid, seed, output_dir)
            if summary:
                summaries.append(summary)
    else:
        # Parallel execution
        with mp.Pool(parallel) as pool:
            run_func = partial(
                run_single_experiment,
                params_with_grid,
                output_dir=output_dir,
            )
            results = pool.map(run_func, seeds)
            summaries = [r for r in results if r is not None]
    
    return summaries


def main():
    """Main entry point for unified experiment runner."""
    parser = argparse.ArgumentParser(
        description="Unified runner for theory-first deletion capacity experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Grid search mode
    parser.add_argument(
        "--grid-file",
        type=str,
        help="YAML file with parameter grid for grid search",
    )
    
    # Single run mode (theory-first parameters)
    parser.add_argument("--target-G", type=float, help="Gradient norm bound")
    parser.add_argument("--target-D", type=float, help="Parameter diameter")
    parser.add_argument("--target-c", type=float, help="Min inverse-Hessian eigenvalue")
    parser.add_argument("--target-C", type=float, help="Max inverse-Hessian eigenvalue")
    parser.add_argument("--target-lambda", type=float, help="Strong convexity")
    parser.add_argument("--target-PT", type=float, help="Total path length")
    parser.add_argument("--target-ST", type=float, help="Cumulative squared gradient")
    
    # Privacy parameters
    parser.add_argument("--rho-total", type=float, default=1.0, help="zCDP privacy budget")
    parser.add_argument("--eps-total", type=float, help="(ε,δ)-DP epsilon budget")
    parser.add_argument("--delta-total", type=float, default=1e-5, help="(ε,δ)-DP delta")
    
    # Execution parameters
    parser.add_argument("--max-events", type=int, default=1000, help="Horizon T")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_parquet",
        help="Base output directory for Parquet files",
    )
    
    # Other options
    parser.add_argument("--dry-run", action="store_true", help="Show grid without running")
    
    args = parser.parse_args()
    
    # Determine execution mode
    if args.grid_file:
        # Grid search mode
        print(f"Loading grid from {args.grid_file}")
        
        # Resolve path relative to script directory if not absolute
        if not os.path.isabs(args.grid_file):
            grid_file_path = os.path.join(_SCRIPT_DIR, args.grid_file)
        else:
            grid_file_path = args.grid_file
        
        if not os.path.exists(grid_file_path):
            print(f"Error: Grid file not found: {grid_file_path}")
            return 1
        
        # Load grid
        grid_spec = load_grid_from_yaml(grid_file_path)
        matrix = grid_spec.get("matrix", {})
        limit = grid_spec.get("limit", None)
        
        # Generate combinations
        combinations = generate_grid_combinations(matrix)
        print(f"Generated {len(combinations)} parameter combinations")
        
        # Apply limit if specified
        if limit and isinstance(limit, int) and limit > 0:
            combinations = combinations[:limit]
            print(f"Limited to {len(combinations)} combinations")
        
        if args.dry_run:
            print("\nDry run - parameter combinations:")
            for i, combo in enumerate(combinations):
                grid_id = grid_hash(combo)
                print(f"{i+1:3d}. {grid_id}: {combo}")
            return 0
        
        # Run grid search
        seeds = list(range(args.seeds))
        
        print(f"\nRunning {len(combinations)} combinations x {len(seeds)} seeds")
        print(f"Output directory: {args.output_dir}")
        
        for i, params in enumerate(combinations):
            print(f"\nProgress: {i+1}/{len(combinations)}")
            run_grid_cell(params, seeds, args.output_dir, args.parallel)
        
        print("\n✓ Grid search complete!")
        
    else:
        # Single run mode
        print("Running single experiment with theory-first parameters")
        
        # Build parameter dictionary from CLI args
        params = {
            "target_G": args.target_G,
            "target_D": args.target_D,
            "target_c": args.target_c,
            "target_C": args.target_C,
            "target_lambda": args.target_lambda,
            "target_PT": args.target_PT,
            "target_ST": args.target_ST,
            "rho_total": args.rho_total,
            "eps_total": args.eps_total,
            "delta_total": args.delta_total,
            "max_events": args.max_events,
            "seeds": 1,  # Single seed in single-run mode
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Validate required parameters
        required = ["target_G", "target_D", "target_c", "target_C", "target_lambda", "target_PT", "target_ST"]
        missing = [k for k in required if k not in params]
        if missing:
            print(f"Error: Missing required theory-first parameters: {missing}")
            print("Use --grid-file for grid search, or provide all theory-first parameters for single run")
            return 1
        
        if args.dry_run:
            print("\nDry run - single experiment parameters:")
            grid_id = grid_hash(params)
            print(f"Grid ID: {grid_id}")
            print(f"Parameters: {params}")
            return 0
        
        # Run single experiment
        seeds = list(range(args.seeds))
        summaries = run_grid_cell(params, seeds, args.output_dir, args.parallel)
        
        print(f"\n✓ Completed {len(summaries)} seed runs")
        print(f"Results written to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
