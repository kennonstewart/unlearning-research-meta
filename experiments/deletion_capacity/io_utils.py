"""
I/O utilities for CSV/JSON writing, plotting, and git operations.
"""

from __future__ import annotations

import csv
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

    def to_csv(self, path: str):
        """Write events to CSV file with standardized regret aliases.

        This normalizes common variations into canonical columns:
          - loss: aliases loss_half_mse -> loss if needed
          - regret (incremental), cum_regret, avg_regret
          - cum_regret_with_noise, avg_regret_with_noise
        """
        if not self.events:
            return

        def _normalize_event(e: Dict[str, Any]) -> Dict[str, Any]:
            row = dict(e)  # shallow copy

            # --- loss aliases ---
            if "loss" not in row and "loss_half_mse" in row:
                row["loss"] = row["loss_half_mse"]

            # --- regret incremental ---
            if "regret" not in row:
                if "regret_increment" in row:
                    row["regret"] = row["regret_increment"]
                elif "instant_regret" in row:
                    row["regret"] = row["instant_regret"]

            # --- cumulative regret ---
            if "cum_regret" not in row:
                if "cumulative_regret" in row:
                    row["cum_regret"] = row["cumulative_regret"]
                elif "regret_cum" in row:
                    row["cum_regret"] = row["regret_cum"]

            # --- average regret ---
            if "avg_regret" not in row:
                if "average_regret" in row:
                    row["avg_regret"] = row["average_regret"]
                elif "regret_avg" in row:
                    row["avg_regret"] = row["regret_avg"]

            # --- noise-aware variants (prefer explicit '...with_noise') ---
            if "cum_regret_with_noise" not in row:
                if "cumulative_regret_with_noise" in row:
                    row["cum_regret_with_noise"] = row["cumulative_regret_with_noise"]
                elif "cum_regret_noise" in row:
                    row["cum_regret_with_noise"] = row["cum_regret_noise"]

            if "avg_regret_with_noise" not in row:
                if "average_regret_with_noise" in row:
                    row["avg_regret_with_noise"] = row["average_regret_with_noise"]
                elif "avg_regret_noise" in row:
                    row["avg_regret_with_noise"] = row["avg_regret_noise"]

            return row

        normalized_events = [_normalize_event(e) for e in self.events]

        # Get all unique keys across all events
        fieldnames = set()
        for event in normalized_events:
            fieldnames.update(event.keys())
        fieldnames = sorted(fieldnames)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized_events)

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


def git_commit_results(summary_path: str, figs_dir: str, dataset: str, algo: str):
    """Stage and commit results with standard format."""
    # Git stage & commit (optional)
    os.system(f"git add {summary_path}")
    os.system(f"git add {figs_dir}/*.pdf")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP2:auto_warmup {dataset}-{algo} {hash_short}'")
