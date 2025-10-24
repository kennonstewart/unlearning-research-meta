"""
Main experiment runner class.

Fixes imports to use the local experiment utils and the code/ modules directly,
and adds a lightweight aggregator for seed summaries.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# Local experiment utils
_EXP_DIR = os.path.dirname(__file__)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from configs.config import Config
from utils.io_utils import (
    EventLogger,
    write_summary_json,
    write_seed_summary_json,
    git_commit_results,
)
from utils.phases import (
    PhaseState,
    bootstrap_phase,
    sensitivity_calibration_phase,
    warmup_phase,
    finalize_accountant_phase,
    workload_phase,
)
from utils.phases import get_privacy_metrics  # delegated import in phases

# code/ modules (data loader + algorithm)
_CODE_DIR = os.path.abspath(os.path.join(_EXP_DIR, "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
from data_loader import (
    get_rotating_linear_stream,
    parse_event_record,
)
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.calibrator import Calibrator
from memory_pair.src.accountant import get_adapter

# Optional Parquet writer integration (exp_engine)
try:  # pragma: no cover - optional dependency
    from exp_engine.engine import write_event_rows as _ee_write_event_rows
except Exception:
    _ee_write_event_rows = None

from exp_integration import write_event_rows_parquet, build_params_from_config


def _get_data_stream(cfg: Config, seed: int):
    """Get synthetic data stream (rotating linear stream only).

    This runner uses the rotating-only synthetic linear stream regardless of
    theory-target fields. Legacy theory-first functionality was removed.
    """
    return get_rotating_linear_stream(
        seed=seed,
        rotate_angle=getattr(cfg, "rotate_angle", 0.01),
        G_hat=getattr(cfg, "G_hat", None),
        D_hat=getattr(cfg, "D_hat", None),
        c_hat=getattr(cfg, "c_hat", None),
        C_hat=getattr(cfg, "C_hat", None),
    )


# Algorithm mapping - MemoryPair only
ALGO_MAP = {"memorypair": MemoryPair}


@dataclass
class SeedResult:
    """Result from one seed run."""

    summary: Dict[str, Any]


def estimate_lbfgs_bounds(model) -> Tuple[float, float]:
    """Estimate eigenvalue bounds (c, C) of the L-BFGS Hessian approx."""
    # If the model provides direct bounds
    if hasattr(model, "lbfgs_bounds"):
        return model.lbfgs_bounds()

    # Try to access B matrix directly
    if hasattr(model, "lbfgs") and hasattr(model.lbfgs, "B_matrix"):
        try:
            B = model.lbfgs.B_matrix()
            if B is not None and B.shape[0] > 0:
                eigs = np.linalg.eigvals(B)
                eigs = eigs[eigs > 0]  # Only positive eigenvalues
                if len(eigs) > 0:
                    c_hat = float(eigs.min())
                    C_hat = float(eigs.max())
                    return c_hat, C_hat
        except Exception:
            pass

    # Fallback
    return 1.0, 1.0


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight average-based aggregator over seed summaries.

    - Averages numeric fields where sensible.
    - Copies through stable configuration fields if present.
    """
    if not summaries:
        return {}

    import math

    keys = set().union(*[s.keys() for s in summaries])
    agg: Dict[str, Any] = {}
    n = len(summaries)

    for k in keys:
        vals = [s.get(k) for s in summaries]
        # average numeric
        num_vals: List[float] = []
        for v in vals:
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)
            ):
                num_vals.append(float(v))
        if num_vals and len(num_vals) == len(vals):
            agg[f"{k}_mean"] = float(sum(num_vals) / n)
        elif k in ("gamma_bar", "gamma_split", "algo", "accountant"):
            # Copy stable config-ish fields from first summary if present
            for s in summaries:
                if k in s:
                    agg[k] = s[k]
                    break
    return agg


class ExperimentRunner:
    """Main experiment runner."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_all(self):
        """Run experiment across all seeds and aggregate results."""
        summaries = []

        for seed in range(self.cfg.seeds):
            print(f"\n=== Running seed {seed + 1}/{self.cfg.seeds} ===")
            result = self.run_one_seed(seed)
            summaries.append(result.summary)

        self.aggregate_and_save(summaries)

    def run_one_seed(self, seed: int) -> SeedResult:
        """Run experiment for one seed."""
        # Set random seed
        np.random.seed(seed)

        # Initialize data stream
        gen = _get_data_stream(self.cfg, seed)

        # Get first sample to determine dimensions
        first_record = next(gen)
        first_x, first_y, first_meta = parse_event_record(first_record)

        # Initialize model and accountant
        model = self._create_model(first_x)

        # Initialize logger and state
        logger = EventLogger()
        state = PhaseState()
        state.current_x, state.current_y = first_x, first_y
        state.current_record = first_record  # Store full record for metadata

        max_events_left = self.cfg.max_events

        # Phase 1: Bootstrap/Calibration
        state, events_used = bootstrap_phase(
            model, gen, self.cfg, logger, state, max_events_left
        )
        max_events_left -= events_used

        # Phase 2: Sensitivity Calibration (if enabled)
        state, events_used = sensitivity_calibration_phase(
            model, gen, self.cfg, logger, state, max_events_left
        )
        max_events_left -= events_used

        # Phase 3: Warmup
        state, events_used = warmup_phase(
            model, gen, self.cfg, logger, state, max_events_left
        )
        max_events_left -= events_used

        # Phase 4: Finalize Accountant
        finalize_accountant_phase(model, self.cfg)

        # Phase 5: Workload
        state, events_used = workload_phase(
            model, gen, self.cfg, logger, state, max_events_left
        )

        # Save results and create summary
        summary = self._create_seed_summary(state, model)

        # Persist event-level Parquet rows if integration is available
        try:
            if write_event_rows_parquet is not None:
                params_with_grid = build_params_from_config(self.cfg.__dict__)
                # Enrich events with seed id
                event_rows = []
                for ev in logger.events:
                    row = dict(ev)
                    row.setdefault("seed", seed)
                    event_rows.append(row)
                base_out = getattr(self.cfg, "parquet_out", "results_parquet")
                write_event_rows_parquet(event_rows, base_out, params_with_grid)
        except Exception as _e:
            print(f"[warn] Failed to write Parquet events for seed {seed}: {_e}")

        return SeedResult(summary=summary)

    def _create_model(self, first_x):
        """Create model with zCDP accountant."""
        accountant = get_adapter(
            "zcdp",
            rho_total=self.cfg.rho_total,
            delta_total=self.cfg.delta_total,
            T=self.cfg.max_events,
            gamma=self.cfg.gamma_delete,
            lambda_=self.cfg.lambda_,
            delta_b=self.cfg.delta_b,
            m_max=self.cfg.m_max,
        )
        calibrator = Calibrator(
            quantile=self.cfg.quantile, D_cap=self.cfg.D_cap, ema_beta=self.cfg.ema_beta
        )
        return MemoryPair(
            dim=first_x.shape[0],
            accountant=accountant,
            calibrator=calibrator,
            recal_window=self.cfg.recal_window,
            recal_threshold=self.cfg.recal_threshold,
            cfg=self.cfg,
        )

    def _save_seed_results(
        self, seed: int, logger: EventLogger, state: PhaseState, model
    ) -> str:
        """Save CSV results for one seed."""
        os.makedirs(self.cfg.out_dir, exist_ok=True)

        # Create parameter-specific filename to prevent overwriting when different
        # parameter combinations use the same output directory
        param_suffix = f"gamma{self.cfg.gamma_bar:.1f}-split{self.cfg.gamma_split:.1f}_{self.cfg.accountant}_eps{self.cfg.eps_total:.1f}"
        csv_path = os.path.join(
            self.cfg.out_dir, f"seed_{seed:03d}_{self.cfg.algo}_{param_suffix}.csv"
        )
        logger.to_csv(csv_path)
        return csv_path

    def _create_seed_summary(self, state: PhaseState, model) -> Dict[str, Any]:
        """Create summary dictionary for one seed."""
        # Base summary
        summary = {
            "inserts": state.inserts,
            "deletes": state.deletes,
            "final_regret": getattr(model, "cumulative_regret", 0.0),
            "gamma_bar": self.cfg.gamma_bar,
            "gamma_split": self.cfg.gamma_split,
            "gamma_insert": self.cfg.gamma_insert,
            "gamma_delete": self.cfg.gamma_delete,
            "quantile": self.cfg.quantile,
            "D_cap": self.cfg.D_cap,
            "accountant_type": self.cfg.accountant,
        }

        # Oracle metrics if available
        if hasattr(model, "get_metrics_dict"):
            m_metrics = model.get_metrics_dict()
            summary.update(
                {
                    "regret_dynamic": m_metrics.get("regret_dynamic"),
                    "regret_static_term": m_metrics.get("regret_static_term"),
                    "regret_path_term": m_metrics.get("regret_path_term"),
                    "P_T": m_metrics.get("P_T"),
                }
            )

        # Add calibration results
        if hasattr(model, "calibration_stats") and model.calibration_stats:
            stats = model.calibration_stats
            summary.update(
                {
                    "G_hat": stats.get("G"),
                    "D_hat": stats.get("D"),
                    "c_hat": stats.get("c"),
                    "C_hat": stats.get("C"),
                }
            )
        else:
            # Fallback: try to get from calibrator attributes if calibration_stats not available
            if hasattr(model, "calibrator"):
                cal = model.calibrator
                summary.update(
                    {
                        "G_hat": getattr(cal, "finalized_G", None),
                        "D_hat": getattr(cal, "D", None),
                        "c_hat": getattr(cal, "c_hat", None),
                        "C_hat": getattr(cal, "C_hat", None),
                    }
                )
            else:
                summary.update(
                    {
                        "G_hat": None,
                        "D_hat": None,
                        "c_hat": None,
                        "C_hat": None,
                    }
                )

        # Add theoretical values
        if hasattr(model, "N_star"):
            summary["N_star_theory"] = model.N_star

        if hasattr(model, "odometer"):
            odometer = model.odometer
            if hasattr(odometer, "deletion_capacity"):
                summary["m_theory"] = odometer.deletion_capacity
            if hasattr(odometer, "sigma_step"):
                summary["sigma_step_theory"] = odometer.sigma_step

            summary.update(get_privacy_metrics(model))

        # Compute empirical regret
        if state.inserts > 0:
            summary["avg_regret_empirical"] = summary["final_regret"] / state.inserts

        # Add RDP-specific or legacy-specific fields
        if self.cfg.accountant == "rdp":
            summary["m_theory_rdp"] = summary.get("m_theory", 1)

        return summary

    def aggregate_and_save(self, summaries: List[Dict[str, Any]]):
        """Aggregate results across seeds and save final outputs."""
        # Aggregate summaries
        aggregated = aggregate_summaries(summaries)

        # Save aggregated summary
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        summary_path = os.path.join(self.cfg.out_dir, f"summary_{self.cfg.algo}.json")
        write_summary_json(aggregated, summary_path)

        # Save aggregated summary
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        summary_path = os.path.join(
            self.cfg.out_dir, f"seed_summary_{self.cfg.algo}.json"
        )
        write_seed_summary_json(summaries, summary_path)

        # Optional: Save to Parquet if enabled
        parquet_out = getattr(self.cfg, "parquet_out", None)
        if parquet_out:
            try:
                from exp_integration import (
                    build_params_from_config,
                    write_seed_summary_parquet,
                )

                params_with_grid = build_params_from_config(self.cfg.__dict__)
                write_seed_summary_parquet(summaries, parquet_out, params_with_grid)
                print(f"✓ Seed summaries also saved to Parquet: {parquet_out}/seeds/")
            except Exception as e:
                print(f"Warning: Could not save seed summaries to Parquet: {e}")

        # Create plots
        figs_dir = os.path.join(self.cfg.out_dir, "figs")

        # Git commit results (optional)
        if getattr(self.cfg, "commit_results", False):
            git_commit_results(summary_path, figs_dir, "synthetic", self.cfg.algo)

        print(f"\nResults saved to {summary_path}")
