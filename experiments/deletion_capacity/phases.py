"""
Phase implementations for experiment workflow.
Breaks down monolithic main loop into testable functions.
"""

import numpy as np
from typing import Tuple, Any, Generator
from config import Config
from io_utils import EventLogger

# Import local modules from deletion_capacity experiment directory
import importlib.util
import os

# Import abs_error from local metrics module
_this_dir = os.path.dirname(os.path.abspath(__file__))
_metrics_path = os.path.join(_this_dir, "metrics.py")
_metrics_spec = importlib.util.spec_from_file_location("local_metrics", _metrics_path)
local_metrics = importlib.util.module_from_spec(_metrics_spec)
_metrics_spec.loader.exec_module(local_metrics)
abs_error = local_metrics.abs_error

# Import get_privacy_metrics from local metrics_utils module
_metrics_utils_path = os.path.join(_this_dir, "metrics_utils.py")
_metrics_utils_spec = importlib.util.spec_from_file_location("local_metrics_utils", _metrics_utils_path)
local_metrics_utils = importlib.util.module_from_spec(_metrics_utils_spec)
_metrics_utils_spec.loader.exec_module(local_metrics_utils)
get_privacy_metrics = local_metrics_utils.get_privacy_metrics

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
    """Create log entry with extended columns from model metrics."""
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
    
    # Get model metrics if available
    model_metrics = {}
    if hasattr(model, 'get_metrics_dict'):
        try:
            model_metrics = model.get_metrics_dict()
        except Exception:
            pass  # Fallback to empty if error
    
    # Add odometer metrics for noise calibration
    odometer_metrics = {}
    if hasattr(model, 'odometer') and model.odometer:
        try:
            if hasattr(model.odometer, 'sigma_step') and model.odometer.sigma_step is not None:
                odometer_metrics['sigma_step_theory'] = model.odometer.sigma_step
            if hasattr(model.odometer, 'eps_step') and model.odometer.eps_step is not None:
                odometer_metrics['eps_step_theory'] = model.odometer.eps_step  
            if hasattr(model.odometer, 'delta_step') and model.odometer.delta_step is not None:
                odometer_metrics['delta_step_theory'] = model.odometer.delta_step
            if hasattr(model.odometer, 'rho_total') and hasattr(model.odometer, 'deletion_capacity'):
                if model.odometer.deletion_capacity > 0:
                    odometer_metrics['rho_step'] = model.odometer.rho_total / model.odometer.deletion_capacity
        except Exception:
            pass  # Fallback to None if error
    
    # Add extended columns with actual values from model or None as fallback
    entry.update({
        # Model learning metrics
        "S_scalar": model_metrics.get("S_scalar", getattr(model, 'S_scalar', None)),
        "eta_t": model_metrics.get("eta_t", getattr(model, 'eta_t', None)),
        "lambda_est": model_metrics.get("lambda_est", getattr(model, 'lambda_est', None)),
        
        # Noise calibration fields (as specified in comment)
        "sigma_step_theory": odometer_metrics.get('sigma_step_theory', None),
        "eps_step_theory": odometer_metrics.get('eps_step_theory', None), 
        "delta_step_theory": odometer_metrics.get('delta_step_theory', None),
        "rho_step": odometer_metrics.get('rho_step', None),
        
        # Add sigma_step from accountant metrics for test compatibility
        "sigma_step": model_metrics.get("sigma_step", odometer_metrics.get('sigma_step_theory', None)),
        
        # Comparator and drift fields (as specified in comment)
        "P_T": model_metrics.get("P_T", None),
        "comparator_type": model_metrics.get("comparator_type", "none"),
        "drift_flag": model_metrics.get("drift_flag", False),
        
        # Additional useful fields
        "sens_delete": None,  # Will be populated during actual deletions
        "P_T_est": model_metrics.get("P_T_est", model_metrics.get("P_T", None)),  # Legacy alias
        
        # Drift-responsive fields
        "drift_boost_remaining": model_metrics.get("drift_boost_remaining", 0),
        "base_eta_t": model_metrics.get("base_eta_t", None),
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
        
        # Add privacy metrics from accountant strategy
        odometer = getattr(model, "odometer", None)
        if odometer and hasattr(odometer, "metrics"):
            privacy_metrics = odometer.metrics()
            base_log_entry.update(privacy_metrics)
        elif odometer:
            # Fallback for legacy odometers
            base_log_entry.update({
                "accountant_type": getattr(cfg, "accountant", "unknown"),
                "eps_spent": 0.0,
                "capacity_remaining": float("inf"),
                "sigma_step_theory": None,
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
    
    # Finalize calibration using insertion gamma
    print("[Bootstrap] Finalizing calibration...")
    model.finalize_calibration(gamma=cfg.gamma_insert)  # Use insertion gamma for N*
    
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
    
    # Check if model uses new accountant interface
    if hasattr(model, "accountant") and model.accountant is not None:
        # New accountant interface - finalization should already be done in MemoryPair
        # during transition to INTERLEAVING phase, so this is a no-op
        if hasattr(model, "phase") and hasattr(model, "Phase"):
            if model.phase == model.Phase.INTERLEAVING:
                print("[Finalize] Accountant already finalized during phase transition")
                return
        
        # If not in INTERLEAVING phase yet, finalize accountant directly
        if hasattr(model, "calibration_stats") and model.calibration_stats:
            stats = {
                "G": model.calibration_stats.get("G", 1.0),
                "D": model.calibration_stats.get("D", 1.0),
                "c": model.calibration_stats.get("c", 1.0),
                "C": model.calibration_stats.get("C", 1.0),
            }
            model.accountant.finalize(stats, T_estimate=cfg.max_events)
        return
    
    # Legacy odometer interface
    if not hasattr(model, "odometer") or model.odometer is None:
        print("[Finalize] No odometer to finalize")
        return
        
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