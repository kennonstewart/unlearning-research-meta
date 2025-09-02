"""
Integration example showing how to add exp_engine APIs to an existing runner.

Event-level Parquet only.
"""

import os
import sys
from typing import List, Dict, Any

# Add paths for imports (adjust as needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import exp_engine APIs
from exp_engine.engine import write_event_rows, attach_grid_id


def save_events_parquet(event_rows: List[Dict[str, Any]], base_out: str, config_params: Dict[str, Any]):
    """
    Save event logs in Parquet format (single source of truth).
    """
    params = attach_grid_id(config_params)
    write_event_rows(event_rows, base_out, params)
    print(f"✓ Event data saved to Parquet under: {base_out}/events/")