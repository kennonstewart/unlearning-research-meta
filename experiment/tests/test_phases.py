#!/usr/bin/env python3
"""
Unit-style tests for the experiment lifecycle phases in experiment/utils/phases.py.

This focuses on:
- Exercising each phase individually
- Verifying what gets logged via EventLogger at each step
"""

import os
import sys

# Make repository root importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..", "..")
sys.path.insert(0, REPO_ROOT)

import numpy as np

# Model and config
from code.memory_pair.src.memory_pair import MemoryPair
from experiment.configs.config import Config

# Stream + record parsing
from code.data_loader import get_synthetic_linear_stream
from code.data_loader.event_schema import parse_event_record

# Phases and logger
from experiment.utils.phases import (
    PhaseState,
    bootstrap_phase,
    sensitivity_calibration_phase,
    warmup_phase,
    finalize_accountant_phase,
    workload_phase,
)
from experiment.utils.io_utils import EventLogger


def _prime_state_with_next_event(gen, state: PhaseState):
    """Pull one event from generator and populate state's current_x/y/record."""
    record = next(gen)
    x, y, _ = parse_event_record(record)
    state.current_x = x
    state.current_y = y
    state.current_record = record


def _make_cfg() -> Config:
    """
    Create a compact config for fast tests:
    - Small bootstrap_iters and sens_calib
    - Aggressive gamma to drive a very small N* (or we clamp N* manually)
    - Low delete_ratio so we see deletes in a short workload
    """
    return Config.from_cli_args(
        dataset="synthetic",
        gamma_bar=100.0,  # large gamma => small N*
        gamma_split=0.9,
        bootstrap_iters=5,
        sens_calib=3,
        delete_ratio=1,  # 1 insert per delete
        max_events=1000,
        seeds=1,
        out_dir="results/tmp",
        algo="memorypair",
        accountant="zcdp",
        rho_total=1.0,  # used by zCDP accountant
        delta_total=1e-5,
        lambda_=0.1,
    )


def _make_model(dim: int, cfg: Config) -> MemoryPair:
    """Construct a MemoryPair model compatible with the phases API."""
    return MemoryPair(dim=dim, cfg=cfg)


def test_bootstrap_phase_logging():
    cfg = _make_cfg()
    logger = EventLogger()
    state = PhaseState()

    dim = 6
    gen = get_synthetic_linear_stream(dim=dim, seed=123, path_type="rotating")
    _prime_state_with_next_event(gen, state)

    model = _make_model(dim, cfg)

    state, used = bootstrap_phase(
        model, gen, cfg, logger, state, max_events_left=cfg.bootstrap_iters
    )
    assert used > 0, "bootstrap_phase should consume events"
    assert len(logger.events) == used, "Expected one log record per calibration step"

    # Check a sample record shape
    rec = logger.events[0]
    assert rec["event_type"] == "calibrate"
    assert rec["op"] == "calibrate"
    # Extended fields from _create_extended_log_entry
    assert "event" in rec and "acc" in rec and "sample_id" in rec and "event_id" in rec


def test_sensitivity_calibration_phase_no_logs():
    cfg = _make_cfg()
    logger = EventLogger()
    state = PhaseState()

    dim = 6
    gen = get_synthetic_linear_stream(dim=dim, seed=321, path_type="rotating")
    _prime_state_with_next_event(gen, state)

    model = _make_model(dim, cfg)

    # Note: This phase does not emit logs by design
    before = len(logger.events)
    state, used = sensitivity_calibration_phase(
        model, gen, cfg, logger, state, max_events_left=cfg.sens_calib
    )
    after = len(logger.events)
    assert used > 0, "sensitivity_calibration_phase should advance the stream"
    assert after == before, (
        "sensitivity_calibration_phase should not emit logs by current design"
    )


def test_warmup_phase_logging():
    cfg = _make_cfg()
    logger = EventLogger()
    state = PhaseState()

    dim = 6
    gen = get_synthetic_linear_stream(dim=dim, seed=999, path_type="rotating")
    _prime_state_with_next_event(gen, state)

    model = _make_model(dim, cfg)

    # First calibrate to set calibration_stats and nominal N*
    state, _ = bootstrap_phase(
        model, gen, cfg, logger, state, max_events_left=cfg.bootstrap_iters
    )

    # Clamp N* small for a fast warmup test (optional but robust)
    model.N_star = min(getattr(model, "N_star", 5) or 5, 3)

    logs_before = len(logger.events)
    state, used = warmup_phase(model, gen, cfg, logger, state, max_events_left=10)
    logs_after = len(logger.events)
    new_logs = logs_after - logs_before

    assert used > 0, "warmup_phase should perform inserts and consume events"
    assert new_logs == used, "Expected one warmup log per insert"

    # Check a sample warmup record
    rec = logger.events[-1]
    assert rec["event_type"] == "warmup"
    assert rec["op"] == "warmup"
    assert "acc" in rec and "x_norm" in rec


def test_finalize_accountant_and_workload_phase_logging():
    cfg = _make_cfg()
    logger = EventLogger()
    state = PhaseState()

    dim = 6
    gen = get_synthetic_linear_stream(dim=dim, seed=777, path_type="rotating")
    _prime_state_with_next_event(gen, state)

    model = _make_model(dim, cfg)

    # 1) Bootstrap (calibration)
    state, _ = bootstrap_phase(
        model, gen, cfg, logger, state, max_events_left=cfg.bootstrap_iters
    )

    # 2) Set small N* and warmup to reach interleaving
    model.N_star = min(getattr(model, "N_star", 5) or 5, 2)
    state, _ = warmup_phase(model, gen, cfg, logger, state, max_events_left=10)

    # 3) Finalize accountant (no logs; should not throw)
    finalize_accountant_phase(model, cfg)
    # Accountant should be ready at this point (either via phase transition or explicit finalize)
    if hasattr(model, "accountant"):
        metrics = model.accountant.metrics()
        # sigma_step may be None until first delete; but accountant presence and fields should exist
        assert "m_capacity" in metrics and "rho_spent" in metrics

    # 4) Workload: interleave inserts/deletes with low delete_ratio
    logs_before = len(logger.events)
    state, used = workload_phase(model, gen, cfg, logger, state, max_events_left=6)
    logs_after = len(logger.events)
    assert used > 0
    assert logs_after - logs_before == used, "Expected one log per workload event"

    # Verify we got both insert and delete rows (with delete_ratio=1)
    kinds = {e["event_type"] for e in logger.events[logs_before:]}
    assert "insert" in kinds, "Expected at least one insert log"
    assert "delete" in kinds, "Expected at least one delete log"

    # Check typical extended fields on a workload row
    sample = logger.events[logs_before]
    for key in ["event", "op", "sample_id", "event_id", "x_norm"]:
        assert key in sample


if __name__ == "__main__":
    np.random.seed(0)
    test_bootstrap_phase_logging()
    test_sensitivity_calibration_phase_no_logs()
    test_warmup_phase_logging()
    test_finalize_accountant_and_workload_phase_logging()
    print("\n✅ Phase tests passed")
