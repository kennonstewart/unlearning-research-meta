"""
Thin integration layer to use exp_engine from deletion_capacity experiment.

Provides:
- build_params_from_config(cfg) -> params dict (with canonical fields)
- write_event_rows_parquet(events, base_out, params)
"""

from typing import Dict, Any, List
from exp_engine.engine import write_event_rows, attach_grid_id


def build_params_from_config(cfg) -> Dict[str, Any]:
    # Pick canonical fields used for content addressing and partitioning
    params = {
        "algo": "memorypair",
        "accountant": getattr(cfg, "accountant", "zcdp"),
        "gamma_bar": getattr(cfg, "gamma_bar", None),
        "gamma_split": getattr(cfg, "gamma_split", None),
        "rho_total": getattr(cfg, "rho_total", None),
        "target_PT": getattr(cfg, "target_PT", None),
        "target_ST": getattr(cfg, "target_ST", None),
        "delete_ratio": getattr(cfg, "delete_ratio", None),
        # Add more stable params as needed...
    }
    return attach_grid_id(params)


def write_event_rows_parquet(event_rows: List[Dict[str, Any]], base_out: str, params: Dict[str, Any]) -> None:
    write_event_rows(event_rows, base_out, params)