"""
Thin integration layer to use exp_engine from deletion_capacity experiment.

Provides:
- build_params_from_config(cfg) -> params dict (with canonical fields)
- write_event_rows_parquet(events, base_out, params)
- write_seed_summary_parquet(summaries, base_out, params)
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List

# Ensure repository root is on sys.path so `exp_engine` is importable when
# running scripts from the experiment/ directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from exp_engine.engine import write_event_rows, attach_grid_id
from exp_engine.engine.cah import canonicalize_params


def build_params_from_config(cfg) -> Dict[str, Any]:
    """Build a parameter dict for content-addressed hashing.

    Uses the full configuration (minus volatile/path-like fields via
    canonicalize_params) so that each unique parameter combination gets a
    distinct grid_id without embedding parameters in the ID string.
    """
    # Accept either a dataclass-like object or a raw dict
    if isinstance(cfg, dict):
        raw = dict(cfg)
    else:
        # Fallback for objects: use __dict__ if available
        raw = dict(getattr(cfg, "__dict__", {}))

    # Ensure algo is present for downstream context
    raw.setdefault("algo", "memorypair")

    # Canonicalize to drop volatile keys and normalize floats
    params = canonicalize_params(raw)

    # Attach content-addressed grid_id
    return attach_grid_id(params)


def write_event_rows_parquet(
    event_rows: List[Dict[str, Any]], base_out: str, params: Dict[str, Any]
) -> None:
    write_event_rows(event_rows, base_out, params)


def write_seed_summary_parquet(
    summaries: List[Dict[str, Any]], base_out: str, params: Dict[str, Any]
) -> None:
    """Write seed-level summaries to HIVE-partitioned Parquet.
    
    Args:
        summaries: List of seed summary dictionaries
        base_out: Base output directory for Parquet files
        params: Parameter dict with grid_id
    """
    if not summaries:
        return
    
    # Convert summaries to DataFrame
    df = pd.DataFrame(summaries)
    
    # Ensure grid_id is present
    if "grid_id" not in df.columns and params and "grid_id" in params:
        df["grid_id"] = params["grid_id"]
    
    # Ensure required partition columns exist
    partition_cols = ["grid_id", "seed"]
    for col in partition_cols:
        if col not in df.columns:
            df[col] = None
    
    # Write to seeds dataset
    seeds_path = os.path.join(base_out, "seeds")
    os.makedirs(seeds_path, exist_ok=True)
    df.to_parquet(seeds_path, partition_cols=partition_cols, engine="pyarrow", index=False)
    
    # Save grid parameters once (no master manifest)
    if params and "grid_id" in params:
        grid_id = str(params["grid_id"])
        grid_dir = os.path.join(base_out, "grids", f"grid_id={grid_id}")
        os.makedirs(grid_dir, exist_ok=True)
        params_path = os.path.join(grid_dir, "params.json")
        if not os.path.exists(params_path):
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2, sort_keys=True)
