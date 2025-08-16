"""
Configuration dataclass for deletion capacity experiments.
Centralizes all CLI parameters and provides type safety.
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional
from typing import get_origin, get_args, Union, Any


@dataclass
class Config:
    """Configuration for deletion capacity experiments."""

    # Dataset and basic params
    dataset: str = "synthetic"
    gamma_bar: float = 1.0
    gamma_split: float = 0.5  # Fraction of gamma_bar allocated to insertions
    bootstrap_iters: int = 500
    delete_ratio: float = 10.0
    max_events: int = 10000
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

    # Relaxed accountant specific parameters
    relaxation_factor: float = 0.8  # Factor to reduce noise for relaxed mode

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
    lambda_est_bounds: List[float] = field(
        default_factory=lambda: [1e-8, 1e6]
    )  # bounds for lambda estimation
    pair_admission_m: float = 1e-6  # threshold for curvature pair admission
    hessian_clamp_eps: float = 1e-12  # epsilon for spectrum clamping
    d_max: float = float("inf")  # max direction norm (trust region style)
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

    # Comparator and drift configuration
    comparator: str = "dynamic"  # "static" or "dynamic" (mapped from --comparator)
    enable_oracle: bool = False  # Enable oracle/comparator functionality
    drift_threshold: float = 0.1  # Threshold for drift detection
    drift_kappa: float = 0.5  # LR boost factor (1 + kappa)
    drift_window: int = 10  # Duration of LR boost in steps
    drift_adaptation: bool = False  # Enable drift-responsive learning rate

    @property
    def gamma_insert(self) -> float:
        """Gamma budget allocated to insertions (learning)."""
        return self.gamma_bar * self.gamma_split

    @property
    def gamma_delete(self) -> float:
        """Gamma budget allocated to deletions (privacy)."""
        return self.gamma_bar * (1.0 - self.gamma_split)

    @classmethod
    def from_cli_args(cls, **kwargs) -> "Config":
        """Create Config from CLI arguments, handling alphas parsing."""
        # Helper functions for robust type coercion
        import re

        def _is_none_string(val):
            return isinstance(val, str) and val.strip().lower() in ("none", "null")

        def _to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("true", "1", "yes", "y", "t"):
                    return True
                if v in ("false", "0", "no", "n", "f"):
                    return False
            raise ValueError(f"Cannot coerce {val!r} to bool")

        def _to_number(val):
            # Handle int/float, inf, scientific notation
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("inf", "+inf", "infinity", "+infinity"):
                    return float("inf")
                if v in ("-inf", "-infinity"):
                    return float("-inf")
                # Try int, then float
                try:
                    if re.match(r"^-?\d+$", v):
                        return int(v)
                    return float(v)
                except Exception:
                    pass
            raise ValueError(f"Cannot coerce {val!r} to number")

        def _coerce_sequence(val, elem_type):
            # Accept comma-separated string or list/tuple
            if isinstance(val, str):
                # Split on commas, strip whitespace
                items = [s.strip() for s in val.split(",") if s.strip() != ""]
                # Recursively coerce each element
                return [_coerce_to_annotation(item, elem_type) for item in items]
            elif isinstance(val, (list, tuple)):
                return [_coerce_to_annotation(item, elem_type) for item in val]
            else:
                # Single element, wrap in list
                return [_coerce_to_annotation(val, elem_type)]

        def _coerce_to_annotation(val, annotation):
            # Handle Optional[T], Union[T, NoneType]
            origin = get_origin(annotation)
            args = get_args(annotation)
            if origin is Union and type(None) in args:
                # Optional/Union: Try None, else try the other type
                non_none_args = [a for a in args if a is not type(None)]
                if _is_none_string(val) or val is None:
                    return None
                if len(non_none_args) == 1:
                    return _coerce_to_annotation(val, non_none_args[0])
                # Multiple non-None, try each
                for a in non_none_args:
                    try:
                        return _coerce_to_annotation(val, a)
                    except Exception:
                        continue
                raise ValueError(f"Cannot coerce {val!r} to {annotation}")
            # Handle List[T], Tuple[T,...]
            if origin in (list, List, tuple, Tuple):
                elem_type = args[0] if args else Any
                return _coerce_sequence(val, elem_type)
            # Handle bool
            if annotation is bool:
                return _to_bool(val)
            # Handle int/float
            if annotation is int:
                # Accept float if it's an integer value
                try:
                    n = _to_number(val)
                    if isinstance(n, float) and n.is_integer():
                        return int(n)
                    return int(n)
                except Exception:
                    pass
            if annotation is float:
                return float(_to_number(val))
            # Handle str
            if annotation is str:
                if val is None:
                    return ""
                return str(val)
            # Handle Any
            if annotation is Any:
                return val
            # Fallback: just return as is
            return val

        # Parse alphas string if provided (special-case, but still coerce recursively)
        if "alphas" in kwargs and isinstance(kwargs["alphas"], str):
            alphas_str = kwargs["alphas"]
            alphas = []
            for alpha_str in alphas_str.split(","):
                alpha_str = alpha_str.strip()
                if alpha_str.lower() in ("inf", "infinity"):
                    alphas.append(float("inf"))
                else:
                    alphas.append(alpha_str)
            kwargs["alphas"] = alphas

        # Filter kwargs to only include keys that are dataclass fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        # Coerce all filtered kwargs to the correct type
        field_types = {f.name: f.type for f in fields(cls)}
        for k in list(filtered_kwargs.keys()):
            if k in field_types:
                try:
                    filtered_kwargs[k] = _coerce_to_annotation(
                        filtered_kwargs[k], field_types[k]
                    )
                except Exception:
                    # Fallback: leave as is for now, may raise at construction
                    pass

        # Handle None values for non-Optional fields by removing them to use defaults
        field_info = {f.name: f for f in fields(cls)}
        for k in list(filtered_kwargs.keys()):
            if filtered_kwargs[k] is None:
                field = field_info.get(k)
                if field and field.type is not type(None):
                    # Check if it's Optional (Union with None)
                    origin = get_origin(field.type)
                    args = get_args(field.type)
                    is_optional = origin is Union and type(None) in args

                    if not is_optional:
                        # Remove None value to let dataclass use its default
                        del filtered_kwargs[k]

        return cls(**filtered_kwargs)

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
