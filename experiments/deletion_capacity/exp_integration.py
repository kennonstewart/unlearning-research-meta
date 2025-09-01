# exp_integration.py
import os
import sys
from typing import Dict, List, Any, Iterable
import math

# Add project root to path for exp_engine imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_engine.engine import attach_grid_id, write_seed_rows, write_event_rows
from exp_engine.engine.duck import create_connection_and_views

PARTITION_KEYS = [
    "algo","accountant","gamma_bar","gamma_split","target_PT","target_ST",
    "delete_ratio","rho_total","grid_id","seed"
]

# --------------- param shaping ---------------
VOLATILE_KEYS = {"seed","base_out","parquet_out","csv_out","timestamp","run_id","out_dir"}

def build_params_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract stable parameters for hashing + partitioning from runner config/args.
    Remove volatile keys; normalize numeric types.
    """
    params = {k: v for k, v in cfg.items() if k not in VOLATILE_KEYS}
    # Set defaults if missing
    params.setdefault("algo", "memorypair")
    params.setdefault("accountant", "zcdp")
    # Ensure floats for gamma/rho/PT/ST
    for k in ["gamma_bar","gamma_split","rho_total","target_PT","target_ST"]:
        if k in params and params[k] is not None:
            params[k] = float(params[k])
    return attach_grid_id(params)

# --------------- writers ---------------
def write_seed_summary_parquet(seed_rows: List[Dict[str, Any]], base_out: str, params_with_grid: Dict[str, Any]):
    """Write seed summary data to Parquet with grid parameters."""
    # Combine parameters with seed data, ensuring each seed row has all grid params
    enriched_rows = []
    for row in seed_rows:
        enriched_row = {**params_with_grid, **row}
        enriched_rows.append(enriched_row)
    
    write_seed_rows(enriched_rows, base_out, params_with_grid)

def write_event_rows_parquet(event_rows: Iterable[Dict[str, Any]], base_out: str, params_with_grid: Dict[str, Any]):
    """Write event data to Parquet with grid parameters."""
    # Accept generator or list to allow chunked writes
    def _gen():
        for r in event_rows:
            yield {**params_with_grid, **r}
    write_event_rows(list(_gen()), base_out, params_with_grid)

# --------------- duckdb ---------------
def open_duckdb(base_out: str):
    """Return duckdb connection with views registered."""
    return create_connection_and_views(base_out)