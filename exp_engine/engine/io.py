from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .cah import attach_grid_id


def _ensure_partition_columns(df: pd.DataFrame, required_partitions: List[str]) -> pd.DataFrame:
    """Ensure all required partition columns are present in DataFrame."""
    for col in required_partitions:
        if col not in df.columns:
            df[col] = None  # Add missing columns as None
    return df


def _write_parquet_partitioned(df: pd.DataFrame, base_path: str, partition_cols: List[str]):
    """Write DataFrame as partitioned Parquet using HIVE scheme."""
    if df.empty:
        return
        
    # Ensure directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Use pandas to_parquet with HIVE partitioning
    df.to_parquet(
        base_path,
        partition_cols=partition_cols,
        engine='pyarrow',
        index=False
    )


def write_seed_rows(
    seed_data: List[Dict[str, Any]], 
    base_out: str,
    params: Optional[Dict[str, Any]] = None
) -> str:
    """Write seed-level summary data to partitioned Parquet.
    
    Args:
        seed_data: List of dictionaries with seed-level metrics
        base_out: Base output directory
        params: Optional parameters to attach grid_id (if not already present)
        
    Returns:
        Path to the written Parquet dataset
    """
    if not seed_data:
        return ""
        
    # Convert to DataFrame
    df = pd.DataFrame(seed_data)
    
    # Attach grid_id if not present and params provided
    if 'grid_id' not in df.columns and params:
        params_with_grid = attach_grid_id(params)
        df['grid_id'] = params_with_grid['grid_id']
    
    # Standard partition columns for seeds
    partition_cols = [
        'algo', 'accountant', 'gamma_bar', 'gamma_split', 
        'target_PT', 'target_ST', 'delete_ratio', 'rho_total',
        'grid_id', 'seed'
    ]
    
    # Ensure partition columns exist
    df = _ensure_partition_columns(df, partition_cols)
    
    # Write to seeds dataset
    seeds_path = os.path.join(base_out, "seeds")
    _write_parquet_partitioned(df, seeds_path, partition_cols)
    
    # Also save grid parameters if provided
    if params and 'grid_id' in df.columns:
        grid_id = df['grid_id'].iloc[0]
        grid_dir = os.path.join(base_out, "grids", f"grid_id={grid_id}")
        os.makedirs(grid_dir, exist_ok=True)
        
        params_path = os.path.join(grid_dir, "params.json")
        if not os.path.exists(params_path):
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2, sort_keys=True)
    
    return seeds_path


def write_event_rows(
    event_data: List[Dict[str, Any]],
    base_out: str, 
    params: Optional[Dict[str, Any]] = None
) -> str:
    """Write event-level data to partitioned Parquet.
    
    Args:
        event_data: List of dictionaries with event-level data
        base_out: Base output directory
        params: Optional parameters to attach grid_id (if not already present)
        
    Returns:
        Path to the written Parquet dataset
    """
    if not event_data:
        return ""
        
    # Convert to DataFrame
    df = pd.DataFrame(event_data)
    
    # Attach grid_id if not present and params provided
    if 'grid_id' not in df.columns and params:
        params_with_grid = attach_grid_id(params)
        df['grid_id'] = params_with_grid['grid_id']
    
    # Partition columns for events (same as seeds plus event columns)
    partition_cols = [
        'algo', 'accountant', 'gamma_bar', 'gamma_split',
        'target_PT', 'target_ST', 'delete_ratio', 'rho_total', 
        'grid_id', 'seed'
    ]
    
    # Ensure partition columns exist
    df = _ensure_partition_columns(df, partition_cols)
    
    # Write to events dataset
    events_path = os.path.join(base_out, "events")
    _write_parquet_partitioned(df, events_path, partition_cols)
    
    # Also save grid parameters if provided
    if params and 'grid_id' in df.columns:
        grid_id = df['grid_id'].iloc[0]
        grid_dir = os.path.join(base_out, "grids", f"grid_id={grid_id}")
        os.makedirs(grid_dir, exist_ok=True)
        
        params_path = os.path.join(grid_dir, "params.json")
        if not os.path.exists(params_path):
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2, sort_keys=True)
    
    return events_path