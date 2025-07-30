"""
Main experiment runner class.
Orchestrates the entire deletion capacity experiment workflow.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

# Add code path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from data_loader import get_rotating_mnist_stream, get_synthetic_linear_stream
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.odometer import PrivacyOdometer, RDPOdometer
from memory_pair.src.calibrator import Calibrator
from baselines import SekhariBatchUnlearning, QiaoHessianFree

from config import Config
from protocols import AccountantAdapter, ModelAdapter
from phases import PhaseState, bootstrap_phase, sensitivity_calibration_phase, warmup_phase, finalize_accountant_phase, workload_phase
from io_utils import EventLogger, write_summary_json, create_plots, git_commit_results
from metrics_utils import aggregate_summaries, get_privacy_metrics
from metrics import regret


# Algorithm and dataset mappings
ALGO_MAP = {
    "memorypair": MemoryPair,
    "sekhari": SekhariBatchUnlearning,
    "qiao": QiaoHessianFree,
}

DATASET_MAP = {
    "rot-mnist": get_rotating_mnist_stream,
    "synthetic": get_synthetic_linear_stream,
}


@dataclass
class SeedResult:
    """Result from one seed run."""
    summary: Dict[str, Any]
    csv_path: str


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


class ExperimentRunner:
    """Main experiment runner."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def run_all(self):
        """Run experiment across all seeds and aggregate results."""
        summaries = []
        csv_paths = []
        
        for seed in range(self.cfg.seeds):
            print(f"\n=== Running seed {seed + 1}/{self.cfg.seeds} ===")
            result = self.run_one_seed(seed)
            summaries.append(result.summary)
            csv_paths.append(result.csv_path)
        
        self.aggregate_and_save(summaries, csv_paths)
    
    def run_single_seed(self, seed: int) -> str:
        """Run experiment for a single seed and return CSV path."""
        result = self.run_one_seed(seed)
        return result.csv_path
    
    def run_one_seed(self, seed: int) -> SeedResult:
        """Run experiment for one seed."""
        # Set random seed
        np.random.seed(seed)
        
        # Initialize data stream
        stream_fn = DATASET_MAP[self.cfg.dataset]
        gen = stream_fn(seed=seed)
        
        # Get first sample to determine dimensions
        first_x, first_y = next(gen)
        
        # Initialize model and accountant
        model = self._create_model(first_x)
        
        # Initialize logger and state
        logger = EventLogger()
        state = PhaseState()
        state.current_x, state.current_y = first_x, first_y
        
        max_events_left = self.cfg.max_events
        
        # Phase 1: Bootstrap/Calibration
        state, events_used = bootstrap_phase(model, gen, self.cfg, logger, state, max_events_left)
        max_events_left -= events_used
        
        # Phase 2: Sensitivity Calibration (if enabled)
        state, events_used = sensitivity_calibration_phase(model, gen, self.cfg, logger, state, max_events_left)
        max_events_left -= events_used
        
        # Phase 3: Warmup
        state, events_used = warmup_phase(model, gen, self.cfg, logger, state, max_events_left)
        max_events_left -= events_used
        
        # Phase 4: Finalize Accountant
        finalize_accountant_phase(model, self.cfg)
        
        # Phase 5: Workload
        state, events_used = workload_phase(model, gen, self.cfg, logger, state, max_events_left)
        
        # Save results and create summary
        csv_path = self._save_seed_results(seed, logger, state, model)
        summary = self._create_seed_summary(state, model)
        
        return SeedResult(summary=summary, csv_path=csv_path)
    
    def _create_model(self, first_x):
        """Create model with appropriate accountant."""
        # Parse alphas for RDP
        alpha_list = self.cfg.alphas.copy()
        
        # Create accountant
        if self.cfg.accountant == "rdp":
            odometer = RDPOdometer(
                eps_total=self.cfg.eps_total,
                delta_total=self.cfg.delta_total,
                T=self.cfg.max_events,
                gamma=self.cfg.gamma_priv,
                lambda_=self.cfg.lambda_,
                delta_b=self.cfg.delta_b,
                alphas=alpha_list,
                m_max=self.cfg.m_max,
            )
        else:  # legacy
            odometer = PrivacyOdometer(
                eps_total=self.cfg.eps_total,
                delta_total=self.cfg.delta_total,
                T=self.cfg.max_events,
                gamma=self.cfg.gamma_priv,
                lambda_=self.cfg.lambda_,
                delta_b=self.cfg.delta_b,
            )
        
        # Create calibrator
        calibrator = Calibrator(
            quantile=self.cfg.quantile,
            D_cap=self.cfg.D_cap,
            ema_beta=self.cfg.ema_beta
        )
        
        # Create model
        model_class = ALGO_MAP[self.cfg.algo]
        if self.cfg.algo == "memorypair":
            model = model_class(
                dim=first_x.shape[0],
                odometer=odometer,
                calibrator=calibrator,
                recal_window=self.cfg.recal_window,
                recal_threshold=self.cfg.recal_threshold,
            )
        else:
            model = model_class(dim=first_x.shape[0], odometer=odometer)
        
        return model
    
    def _save_seed_results(self, seed: int, logger: EventLogger, state: PhaseState, model) -> str:
        """Save CSV results for one seed."""
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        csv_path = os.path.join(self.cfg.out_dir, f"seed_{seed:03d}_{self.cfg.dataset}_{self.cfg.algo}.csv")
        logger.to_csv(csv_path)
        return csv_path
    
    def _create_seed_summary(self, state: PhaseState, model) -> Dict[str, Any]:
        """Create summary dictionary for one seed."""
        # Base summary
        summary = {
            "inserts": state.inserts,
            "deletes": state.deletes,
            "final_regret": getattr(model, "cumulative_regret", 0.0),
            "gamma_learn": self.cfg.gamma_learn,
            "gamma_priv": self.cfg.gamma_priv,
            "quantile": self.cfg.quantile,
            "D_cap": self.cfg.D_cap,
            "accountant_type": self.cfg.accountant,
        }
        
        # Add calibration results
        if hasattr(model, "calibrator"):
            cal = model.calibrator
            summary.update({
                "G_hat": getattr(cal, "G_hat", np.nan),
                "D_hat": getattr(cal, "D_hat", np.nan),
                "c_hat": getattr(cal, "c_hat", np.nan),
                "C_hat": getattr(cal, "C_hat", np.nan),
            })
        
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
    
    def aggregate_and_save(self, summaries: List[Dict[str, Any]], csv_paths: List[str]):
        """Aggregate results across seeds and save final outputs."""
        # Aggregate summaries
        aggregated = aggregate_summaries(summaries)
        
        # Save aggregated summary
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        summary_path = os.path.join(self.cfg.out_dir, f"summary_{self.cfg.dataset}_{self.cfg.algo}.json")
        write_summary_json(aggregated, summary_path)
        
        # Create plots
        figs_dir = os.path.join(self.cfg.out_dir, "figs")
        create_plots(csv_paths, figs_dir)
        
        # Git commit results
        git_commit_results(summary_path, figs_dir, self.cfg.dataset, self.cfg.algo)
        
        print(f"\nResults saved to {summary_path}")
        print(f"Plots saved to {figs_dir}")