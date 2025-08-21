#!/usr/bin/env python3
"""Smoke test to ensure synthetic stream parameters propagate from Config to events."""

import os
import sys

# Add paths for imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(REPO_ROOT, 'code'))

from config import Config
from runner import _get_data_stream
from data_loader import parse_event_record

def main():
    cfg = Config(
        dataset="synthetic",
        seeds=1,
        rotate_angle=0.2,
        drift_rate=0.05,
        G_hat=3.0,
        D_hat=1.5,
        c_hat=0.2,
        C_hat=4.0,
    )
    gen = _get_data_stream(cfg, seed=0)
    record = next(gen)
    _, _, meta = parse_event_record(record)
    metrics = meta["metrics"]

    assert metrics["rotate_angle"] == 0.2
    assert metrics["drift_rate"] == 0.05
    assert metrics["G_hat"] == 3.0
    assert metrics["D_hat"] == 1.5
    assert metrics["c_hat"] == 0.2
    assert metrics["C_hat"] == 4.0

    print("Stream parameters propagated correctly.")

if __name__ == "__main__":
    main()
