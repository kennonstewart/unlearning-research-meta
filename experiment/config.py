"""
Simplified configuration dataclass for the theory-first experiment framework.

This replaces the legacy configs/config.py with a cleaner, more focused design
that emphasizes theory-first parameters and integration with exp_engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for theory-first experiments."""
    
    # Experiment identification
    algo: str = "memorypair"
    accountant: str = "zcdp"
    
    # Theory-first stream parameters (targets)
    target_G: Optional[float] = None  # gradient norm bound
    target_D: Optional[float] = None  # parameter/domain diameter
    target_c: Optional[float] = None  # min inverse-Hessian eigenvalue
    target_C: Optional[float] = None  # max inverse-Hessian eigenvalue
    target_lambda: Optional[float] = None  # strong convexity of loss
    target_PT: Optional[float] = None  # total path length over horizon T
    target_ST: Optional[float] = None  # cumulative squared gradient over T
    
    # Privacy parameters
    rho_total: float = 1.0  # zCDP privacy budget
    eps_total: Optional[float] = None  # (ε,δ)-DP budget (alternative)
    delta_total: float = 1e-5  # (ε,δ)-DP delta parameter
    
    # Algorithm parameters
    lambda_: float = 0.1  # strong convexity parameter used by algorithm
    lambda_reg: float = 0.0  # L2 regularization
    quantile: float = 0.95  # calibration quantile
    D_cap: float = 10.0  # diameter cap for calibration
    
    # Execution parameters
    max_events: int = 10000  # horizon T
    seeds: int = 1  # number of random seeds to run
    bootstrap_iters: int = 500  # bootstrap iterations
    
    # Stream configuration
    dim: int = 10  # feature dimension
    path_style: str = "rotating"  # stream path style
    rotate_angle: float = 0.01  # rotation angle for rotating stream
    
    # Output configuration
    out_dir: str = "results"
    parquet_out: str = "results_parquet"
    
    # Advanced algorithm options
    recal_window: Optional[int] = None  # recalibration window
    recal_threshold: float = 0.3  # recalibration threshold
    ema_beta: float = 0.9  # EMA beta for calibration
    m_max: Optional[int] = 10  # max deletions
    delta_b: float = 0.05  # privacy parameter
    
    # Feature flags
    adaptive_geometry: bool = False
    dynamic_comparator: bool = True
    strong_convexity: bool = False
    
    @property
    def gamma_bar(self) -> float:
        """Total gamma budget (derived from privacy params)."""
        # This is typically derived from privacy accounting
        # For now, use a default value
        return 1.0
    
    @property
    def gamma_split(self) -> float:
        """Fraction of gamma_bar for insertions."""
        return 0.5
    
    @property
    def gamma_insert(self) -> float:
        """Gamma budget for insertions (learning)."""
        return self.gamma_bar * self.gamma_split
    
    @property
    def gamma_delete(self) -> float:
        """Gamma budget for deletions (privacy)."""
        return self.gamma_bar * (1.0 - self.gamma_split)
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Filter to only valid fields
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
