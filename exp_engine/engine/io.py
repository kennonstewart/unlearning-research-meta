from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .cah import attach_grid_id


def _ensure_partition_columns(df: pd.DataFrame, required_partitions: List[str]) -> pd.DataFrame:
    """Ensure all required partition columns are present in DataFrame."""
    for col in required_partitions:
        if col not in df.columns:
            df[col] = None
    return df


def _write_parquet_partitioned(df: pd.DataFrame, base_path: str, partition_cols: List[str]) -> None:
    """Write DataFrame as partitioned Parquet using HIVE scheme."""
    if df.empty:
        return
    os.makedirs(base_path, exist_ok=True)
    df.to_parquet(base_path, partition_cols=partition_cols, engine="pyarrow", index=False)


def write_event_rows(
    event_data: List[Dict[str, Any]],
    base_out: str,
    params: Optional[Dict[str, Any]] = None
) -> str:
    """Write event-level data to HIVE-partitioned Parquet.

    Partition columns: grid_id, seed.

    Args:
        event_data: List of per-event dicts. Must include at least grid_id and seed,
                    or provide params so grid_id can be attached.
        base_out: Base output directory.
        params: Optional parameter dict used to attach grid_id and persist params.json.

    Returns:
        Path to the written Parquet dataset.
    """
    if not event_data:
        return ""

    df = pd.DataFrame(event_data)

    # Attach grid_id if not present and params provided
    if "grid_id" not in df.columns and params:
        params_with_grid = attach_grid_id(params)
        df["grid_id"] = params_with_grid["grid_id"]
        params = params_with_grid  # ensure consistency

    # Ensure required partitions exist
    partition_cols = ["grid_id", "seed"]
    df = _ensure_partition_columns(df, partition_cols)

    # Write to events dataset
    events_path = os.path.join(base_out, "events")
    _write_parquet_partitioned(df, events_path, partition_cols)

    # Save grid parameters once (no master manifest)
    if params and "grid_id" in df.columns:
        grid_id = str(df["grid_id"].iloc[0])
        grid_dir = os.path.join(base_out, "grids", f"grid_id={grid_id}")
        os.makedirs(grid_dir, exist_ok=True)
        params_path = os.path.join(grid_dir, "params.json")
        if not os.path.exists(params_path):
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2, sort_keys=True)

    return events_path