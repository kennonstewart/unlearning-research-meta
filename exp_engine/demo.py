#!/usr/bin/env python3
"""
Demo script showing the exp_engine functionality end-to-end (event-only).
"""

import os
import sys
import tempfile

# Add exp_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import write_event_rows, attach_grid_id
from engine.duck import create_connection_and_views


def demo_basic_functionality():
    print("=== Exp_engine Demo: Event-Level Parquet Only ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}")

        params = {
            "algo": "memorypair",
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
            "rho_total": 1.0,
        }
        params = attach_grid_id(params)
        grid_id = params["grid_id"]
        print(f"Generated grid_id: {grid_id}")
        print(f"Parameters: {params}\n")

        event_data = []
        for seed in [1, 2, 3]:
            for i in range(5):
                event_data.append({
                    **params,
                    "seed": seed,
                    "event_id": i,
                    "op": "insert" if i < 3 else "delete",
                    "regret": 0.05 + 0.01 * i,
                    "acc": 0.8 + 0.01 * i,
                    "eps_spent": 0.1 * i
                })

        events_path = write_event_rows(event_data, tmpdir, params)
        print(f"✓ Wrote event data to: {events_path}")

        conn = create_connection_and_views(tmpdir)
        df = conn.execute("SELECT * FROM events WHERE regret < 0.08").df()
        print("\nSample query (regret < 0.08):")
        print(df.to_string(index=False))


if __name__ == "__main__":
    demo_basic_functionality()