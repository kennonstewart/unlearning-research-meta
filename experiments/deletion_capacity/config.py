"""
Configuration dataclass for deletion capacity experiments.
Centralizes all CLI parameters and provides type safety.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration for deletion capacity experiments."""

    # Dataset and basic params
    dataset: str = "synthetic"
    gamma_learn: float = 1.0
    gamma_priv: float = 0.5
    bootstrap_iters: int = 500
    delete_ratio: float = 10.0
    max_events: int = 100_000
    seeds: int = 200
    out_dir: str = "results/"
    algo: str = "memorypair"

    # Privacy parameters
    eps_total: float = 1.0
    delta_total: float = 1e-5
    lambda_: float = 0.1
    delta_b: float = 0.05
    quantile: float = 0.95
    D_cap: float = 10.0

    # Accountant configuration
    accountant: str = "default"
    alphas: List[float] = field(
        default_factory=lambda: [1.5, 2, 3, 4, 8, 16, 32, 64, float("inf")]
    )

    # Adaptive recalibration
    ema_beta: float = 0.9
    recal_window: Optional[int] = None
    recal_threshold: float = 0.3
    m_max: Optional[int] = 10

    # Sensitivity calibration
    sens_calib: int = 50

    # Output granularity for grid search
    output_granularity: str = "seed"

    # Adaptive geometry defaults
    adagrad_eps: float = 1e-12
    D_bound: float = 1.0
    trim_quantile: float = 0.95
    lambda_floor: float = 1e-6
    lambda_cap: float = 1e3
    lambda_stability_min_steps: int = 100
    eta_max: float = 1.0

    # Strong convexity parameters
    lambda_reg: float = 0.0  # L2 regularization parameter
    lambda_est_beta: float = 0.1  # EMA beta for lambda estimation
    lambda_est_bounds: List[float] = field(default_factory=lambda: [1e-8, 1e6])  # bounds for lambda estimation
    pair_admission_m: float = 1e-6  # threshold for curvature pair admission
    hessian_clamp_eps: float = 1e-12  # epsilon for spectrum clamping
    d_max: float = float('inf')  # max direction norm (trust region style)
    lambda_min_threshold: float = 1e-6  # threshold for lambda stability
    lambda_stability_K: int = 100  # steps required for stability

    # Feature flags (all default False for no-op behavior)
    adaptive_geometry: bool = False
    dynamic_comparator: bool = False
    strong_convexity: bool = False
    adaptive_privacy: bool = False
    drift_mode: bool = False
    window_erm: bool = False
    online_standardize: bool = False

    @classmethod
    def from_cli_args(cls, **kwargs) -> "Config":
        """Create Config from CLI arguments, handling alphas parsing."""
        # Parse alphas string if provided
        if "alphas" in kwargs and isinstance(kwargs["alphas"], str):
            alphas_str = kwargs["alphas"]
            alphas = []
            for alpha_str in alphas_str.split(","):
                alpha_str = alpha_str.strip()
                if alpha_str.lower() in ("inf", "infinity"):
                    alphas.append(float("inf"))
                else:
                    alphas.append(float(alpha_str))
            kwargs["alphas"] = alphas

        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list) and len(v) > 0 and v[-1] == float("inf"):
                # Handle infinity for JSON serialization
                result[k] = [x if x != float("inf") else "inf" for x in v]
            else:
                result[k] = v
        return result
