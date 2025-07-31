"""
I/O utilities for CSV/JSON writing, plotting, and git operations.
"""

import csv
import json
import os
from typing import List, Dict, Any
from plots import plot_capacity_curve, plot_regret


class EventLogger:
    """Simplified event logging with pandas-style interface."""

    def __init__(self):
        self.events = []

    def log(self, event_type: str, **kwargs):
        """Log an event with arbitrary key-value pairs."""
        event = {"event_type": event_type, **kwargs}
        self.events.append(event)

    def to_csv(self, path: str):
        """Write events to CSV file."""
        if not self.events:
            return

        # Get all unique keys across all events
        fieldnames = set()
        for event in self.events:
            fieldnames.update(event.keys())
        fieldnames = sorted(fieldnames)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)

    def clear(self):
        """Clear all logged events."""
        self.events.clear()


def write_summary_json(summary: Dict[str, Any], path: str):
    """Write summary dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def write_seed_summary_json(summary: list, path: str):
    """Write seed-level summaries to JSON file."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def create_plots(csv_paths: List[str], figs_dir: str):
    """Generate plots from CSV data."""
    os.makedirs(figs_dir, exist_ok=True)

    try:
        plot_capacity_curve(csv_paths, os.path.join(figs_dir, "capacity_curve.pdf"))
        plot_regret(csv_paths, os.path.join(figs_dir, "regret.pdf"))
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")


def git_commit_results(summary_path: str, figs_dir: str, dataset: str, algo: str):
    """Stage and commit results with standard format."""
    # Git stage & commit (optional)
    os.system(f"git add {summary_path}")
    os.system(f"git add {figs_dir}/*.pdf")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP2:auto_warmup {dataset}-{algo} {hash_short}'")
