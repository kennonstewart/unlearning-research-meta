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
    m_max: Optional[int] = None

    # Sensitivity calibration
    sens_calib: int = 50

    # Maximum warmup sample complexity to prevent excessively long warmup phases
    max_warmup_N: Optional[int] = 50_000

    # Output granularity for grid search
    output_granularity: str = "seed"

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
