"""
I/O utilities for JSON and Parquet writing.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any


class EventLogger:
    """Simplified event logging with pandas-style interface."""

    def __init__(self):
        self.events = []

    def log(self, event_type: str, **kwargs):
        """Log an event with arbitrary key-value pairs."""
        event = {"event_type": event_type, **kwargs}
        self.events.append(event)



def write_summary_json(summary: Dict[str, Any], path: str):
    """Write summary dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def write_seed_summary_json(summary: list, path: str):
    """Write seed-level summaries to JSON file."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def git_commit_results(summary_path: str, figs_dir: str, dataset: str, algo: str):
    """Stage and commit results with standard format."""
    # Git stage & commit (optional)
    os.system(f"git add {summary_path}")
    os.system(f"git add {figs_dir}/*.pdf")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP2:auto_warmup {dataset}-{algo} {hash_short}'")
