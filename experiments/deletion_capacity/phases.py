"""
Phase implementations for experiment workflow.
Breaks down monolithic main loop into testable functions.
"""

import numpy as np
from typing import Tuple, Any, Generator
from config import Config
from io_utils import EventLogger
from metrics import abs_error
from metrics_utils import get_privacy_metrics

# Add the data loader to path for event parsing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
from data_loader import parse_event_record


def get_pred_and_grad(model, x, y, is_calibration=False):
    """Helper to obtain (pred, grad). Adjust for your API.
    For calibration phase, uses calibrate_step(). For other phases, uses insert().
    """
    if is_calibration and hasattr(model, "calibrate_step"):
        # During calibration, use calibrate_step and get gradient from last_grad
        pred = model.calibrate_step(x, y)
        if hasattr(model, "last_grad") and model.last_grad is not None:
            grad = model.last_grad
            return pred, grad
        else:
            raise RuntimeError("Model must expose last_grad after calibrate_step.")
    elif hasattr(model, "insert"):
        try:
            return model.insert(x, y, return_grad=True)
        except (TypeError, RuntimeError):
            # Fallback if insert doesn't return grad or is in wrong phase
            pred = model.insert(x, y)
            if hasattr(model, "last_grad") and model.last_grad is not None:
                grad = model.last_grad
            elif hasattr(model, "gradient"):
                grad = model.gradient(x, y)
            else:
                raise RuntimeError("Model must expose gradient to estimate G.")
            return pred, grad
    raise RuntimeError("Model has no insert or calibrate_step method.")


class PhaseState:
    """Tracks state across phases."""
    def __init__(self):
        self.event = 0
        self.inserts = 0
        self.deletes = 0
        self.cum_regret = 0.0
        self.current_x = None
        self.current_y = None
        self.current_record = None  # Full event record for metadata


def _get_next_event(gen: Generator, state: PhaseState):
    """Get next event from generator and update state."""
    record = next(gen)
    x, y, meta = parse_event_record(record)
    state.current_x = x
    state.current_y = y
    state.current_record = record
    return x, y, meta


def _create_extended_log_entry(base_entry: dict, state: PhaseState, model, cfg: Config) -> dict:
    """Create log entry with extended columns (set to None if not computed)."""
    # Start with base entry
    entry = base_entry.copy()
    
    # Add metadata from current record
    if state.current_record:
        _, _, meta = parse_event_record(state.current_record)
        entry.update({
            "sample_id": meta.get("sample_id"),
            "event_id": meta.get("event_id"), 
            "segment_id": meta.get("segment_id", 0),
        })
        
        # Add x_norm from metrics if available
        if "metrics" in meta and "x_norm" in meta["metrics"]:
            entry["x_norm"] = meta["metrics"]["x_norm"]
        else:
            entry["x_norm"] = None
    
    # Add extended columns (set to None for now as per requirements)
    entry.update({
        "S_scalar": None,        # Sensitivity scalar
        "eta_t": None,           # Learning rate at time t  
        "lambda_est": None,      # Lambda estimate
        "rho_step": None,        # RDP step parameter
        "sigma_step": None,      # Noise step parameter
        "sens_delete": None,     # Sensitivity for deletions
        "P_T_est": None,         # Probability estimate
    })
    
    return entry


def bootstrap_phase(
    model, 
    gen: Generator,
    cfg: Config,
    logger: EventLogger,
    state: PhaseState,
    max_events_left: int
) -> Tuple[PhaseState, int]:
    """Bootstrap/calibration phase to estimate G, D, c, C."""
    print(f"[Bootstrap] Collecting {cfg.bootstrap_iters} steps to estimate G, D, c, C...")
    
    events_used = 0
    
    # Use the new calibration API
    for _ in range(cfg.bootstrap_iters):
        if events_used >= max_events_left:
            break
            
        pred, grad = get_pred_and_grad(model, state.current_x, state.current_y, is_calibration=True)
        gnorm = np.linalg.norm(grad)
        
        acc_val = abs_error(pred, state.current_y)
        
        # During calibration, odometer isn't finalized yet
        base_log_entry = {
            "event": state.event,
            "op": "calibrate",
            "regret": model.cumulative_regret if hasattr(model, "cumulative_regret") else np.nan,
            "acc": acc_val,
        }
        
        # Add default privacy metrics for calibration phase
        odometer = getattr(model, "odometer", None)
        if cfg.accountant == "rdp" and odometer:
            base_log_entry.update({
                "eps_converted": 0.0,
                "delta_total": odometer.delta_total,
                "eps_remaining": odometer.eps_total,
            })
        elif odometer:
            base_log_entry.update({
                "eps_spent": 0.0,
                "capacity_remaining": float("inf"),
            })
        
        # Create extended log entry with new columns
        log_entry = _create_extended_log_entry(base_log_entry, state, model, cfg)
        
        logger.log("calibrate", **log_entry)
        state.inserts += 1
        state.event += 1
        events_used += 1
        
        # Get next event if more iterations needed
        if events_used < max_events_left and _ < cfg.bootstrap_iters - 1:
            _get_next_event(gen, state)
    
    # Finalize calibration using learning gamma
    print("[Bootstrap] Finalizing calibration...")
    model.finalize_calibration(gamma=cfg.gamma_insert)  # Use insert gamma for N*
    
    return state, events_used


def sensitivity_calibration_phase(
    model,
    gen: Generator,
    cfg: Config,
    logger: EventLogger,
    state: PhaseState,
    max_events_left: int
) -> Tuple[PhaseState, int]:
    """Sensitivity calibration phase to collect delete sensitivity statistics."""
    if cfg.sens_calib <= 0:
        return state, 0
        
    print(f"[SensCalib] Collecting {cfg.sens_calib} sensitivity samples...")
    
    events_used = 0
    
    for _ in range(cfg.sens_calib):
        if events_used >= max_events_left:
            break
            
        # Estimate sensitivity without actually performing delete
        if hasattr(model, "estimate_delete_sensitivity"):
            sensitivity = model.estimate_delete_sensitivity(state.current_x, state.current_y)
            # Record sensitivity in odometer for capacity optimization
            if hasattr(model.odometer, "record_sensitivity"):
                model.odometer.record_sensitivity(sensitivity)
        
        state.event += 1
        events_used += 1
        
        # Get next event if more iterations needed
        if events_used < max_events_left and _ < cfg.sens_calib - 1:
            _get_next_event(gen, state)
    
    return state, events_used


def warmup_phase(
    model,
    gen: Generator,
    cfg: Config,
    logger: EventLogger,
    state: PhaseState,
    max_events_left: int
) -> Tuple[PhaseState, int]:
    """Warmup phase for N* inserts."""
    N_star = getattr(model, "N_star", 0)
    
    # Ensure N_star is an integer
    try:
        N_star = int(N_star)
    except (ValueError, TypeError):
        print(f"Warning: Invalid N_star value '{N_star}'. Skipping warmup.")
        return state, 0

    warmup_needed = N_star - state.inserts
    if warmup_needed <= 0:
        print(f"[Warmup] No warmup needed (N*={N_star}, inserts={state.inserts}).")
        return state, 0

    print(f"[Warmup] Running {warmup_needed} warmup inserts to reach N*={N_star}...")
    
    events_used = 0
    
    for _ in range(warmup_needed):
        if events_used >= max_events_left:
            print("[Warmup] Stopping warmup early due to max_events limit.")
            break
            
        pred, grad = get_pred_and_grad(model, state.current_x, state.current_y)
        gnorm = np.linalg.norm(grad)
        
        acc_val = abs_error(pred, state.current_y)
        
        base_log_entry = {
            "event": state.event,
            "op": "warmup",
            "regret": model.cumulative_regret if hasattr(model, "cumulative_regret") else np.nan,
            "acc": acc_val,
        }
        
        # Add privacy metrics
        base_log_entry.update(get_privacy_metrics(model))
        
        # Create extended log entry with new columns
        log_entry = _create_extended_log_entry(base_log_entry, state, model, cfg)
        
        logger.log("warmup", **log_entry)
        state.inserts += 1
        state.event += 1
        events_used += 1
        
        # Get next event if more iterations needed
        if events_used < max_events_left and _ < warmup_needed - 1:
            try:
                _get_next_event(gen, state)
            except StopIteration:
                print("[Warmup] Data stream exhausted during warmup.")
                break
    
    return state, events_used


def finalize_accountant_phase(model, cfg: Config):
    """Finalize odometer and prepare for interleaving phase."""
    print("[Finalize] Finalizing odometer...")
    
    # Use the proper finalization approach based on model type
    if hasattr(model, "calibration_stats") and model.calibration_stats:
        # MemoryPair model with calibration stats
        model.odometer.finalize_with(model.calibration_stats, T_estimate=cfg.max_events)
    elif hasattr(model.odometer, "finalize_with"):
        # Direct finalization for non-MemoryPair models or fallback when calibration_stats is missing
        stats = {
            "G": getattr(model.calibrator, "finalized_G", 1.0) if hasattr(model, "calibrator") else 1.0,
            "D": getattr(model.calibrator, "D", 1.0) if hasattr(model, "calibrator") else 1.0,
            "c": getattr(model.calibrator, "c_hat", 1.0) if hasattr(model, "calibrator") else 1.0,
            "C": getattr(model.calibrator, "C_hat", 1.0) if hasattr(model, "calibrator") else 1.0,
        }
        T_estimate = cfg.max_events
        model.odometer.finalize_with(stats, T_estimate)
    else:
        # Fallback to simple finalize
        model.odometer.finalize()


def workload_phase(
    model,
    gen: Generator,
    cfg: Config,
    logger: EventLogger,
    state: PhaseState,
    max_events_left: int
) -> Tuple[PhaseState, int]:
    """Main interleaving workload phase."""
    print(f"[Workload] Starting interleaving phase (up to {max_events_left} events)...")
    
    events_used = 0
    k_inserts = 0  # Track inserts since last delete
    
    while events_used < max_events_left:
        try:
            # Decide: insert or delete based on ratio
            if k_inserts < cfg.delete_ratio:
                # Insert
                pred, grad = get_pred_and_grad(model, state.current_x, state.current_y)
                gnorm = np.linalg.norm(grad)
                
                acc_val = abs_error(pred, state.current_y)
                
                base_log_entry = {
                    "event": state.event,
                    "op": "insert",
                    "regret": model.cumulative_regret if hasattr(model, "cumulative_regret") else np.nan,
                    "acc": acc_val,
                }
                
                # Add privacy metrics
                base_log_entry.update(get_privacy_metrics(model))
                
                # Create extended log entry with new columns
                log_entry = _create_extended_log_entry(base_log_entry, state, model, cfg)
                
                logger.log("insert", **log_entry)
                state.inserts += 1
                k_inserts += 1
                
            else:
                # Delete
                model.delete(state.current_x, state.current_y)
                
                base_log_entry = {
                    "event": state.event,
                    "op": "delete",
                    "regret": model.cumulative_regret if hasattr(model, "cumulative_regret") else np.nan,
                    "acc": np.nan,  # No prediction during delete
                }
                
                # Add privacy metrics
                base_log_entry.update(get_privacy_metrics(model))
                
                # Create extended log entry with new columns
                log_entry = _create_extended_log_entry(base_log_entry, state, model, cfg)
                
                logger.log("delete", **log_entry)
                state.deletes += 1
                k_inserts = 0  # Reset insert counter
                
        except Exception as e:
            print(f"Stopping workload: {e}")
            break
        
        state.event += 1
        events_used += 1
        
        # Get next event if more iterations needed
        if events_used < max_events_left:
            try:
                _get_next_event(gen, state)
            except StopIteration:
                print("Data stream exhausted")
                break
    
    return state, events_used