import numpy as np
from typing import Optional, Iterator, Dict, Any, List

try:
    from .utils import set_global_seed
    from .streams import make_stream
    from .event_schema import create_event_record_with_diagnostics
except ImportError:
    from utils import set_global_seed
    from streams import make_stream
    from event_schema import create_event_record_with_diagnostics


class ParameterPathController:
    """Control the path of ground-truth parameters for controlled path-length."""
    
    def __init__(self, dim: int, seed: int = 42, path_type: str = "rotating"):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.path_type = path_type
        
        # Initialize parameter
        self.w_star = self.rng.normal(size=dim)
        self.t = 0
        self.P_T_cumulative = 0.0
        
    def get_next_parameter(self) -> tuple[np.ndarray, float]:
        """Get next ground-truth parameter and path increment."""
        self.t += 1
        
        if self.path_type == "rotating":
            # Simple rotation with controlled magnitude
            w_new = self._rotate_parameter()
        elif self.path_type == "drift":
            # Linear drift
            w_new = self._drift_parameter()
        else:
            # Static parameter
            w_new = self.w_star.copy()
            
        # Compute path increment
        delta_P = np.linalg.norm(w_new - self.w_star)
        self.P_T_cumulative += delta_P
        
        self.w_star = w_new
        return self.w_star.copy(), delta_P
        
    def _rotate_parameter(self) -> np.ndarray:
        """Apply controlled rotation to parameter."""
        # Rotate by small angle (0.01 radians per step)
        angle = 0.01
        if self.dim >= 2:
            # Rotate in first two dimensions
            c, s = np.cos(angle), np.sin(angle)
            R = np.eye(self.dim)
            R[0, 0] = c
            R[0, 1] = -s
            R[1, 0] = s
            R[1, 1] = c
            return R @ self.w_star
        else:
            return self.w_star.copy()
            
    def _drift_parameter(self) -> np.ndarray:
        """Apply linear drift to parameter."""
        drift_rate = 0.001
        drift = self.rng.normal(scale=drift_rate, size=self.dim)
        return self.w_star + drift


class CovarianceGenerator:
    """Generate Gaussian data with configurable covariance structure."""
    
    def __init__(
        self, 
        dim: int, 
        eigs: Optional[List[float]] = None,
        cond_number: Optional[float] = None,
        rand_orth_seed: int = 42
    ):
        self.dim = dim
        self.rng = np.random.default_rng(rand_orth_seed)
        
        # Generate eigenvalues
        if eigs is not None:
            if len(eigs) != dim:
                raise ValueError(f"eigs must have length {dim}, got {len(eigs)}")
            self.eigenvalues = np.array(eigs)
        elif cond_number is not None:
            # Geometric spacing from 1 to 1/cond_number
            self.eigenvalues = np.geomspace(1.0, 1.0/cond_number, dim)
        else:
            # Default: uniform eigenvalues
            self.eigenvalues = np.ones(dim)
            
        # Generate random orthogonal matrix Q
        A = self.rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)
        self.Q = Q
        
        # Construct covariance matrix Σ = Q * diag(λ) * Q^T
        self.Sigma = self.Q @ np.diag(self.eigenvalues) @ self.Q.T
        self.L = np.linalg.cholesky(self.Sigma)  # For efficient sampling
        
    def sample(self, n: int) -> np.ndarray:
        """Sample n points from the Gaussian distribution."""
        z = self.rng.normal(size=(n, self.dim))
        return z @ self.L.T


class StrongConvexityEstimator:
    """Online estimation of strong convexity parameter using secant method."""
    
    def __init__(self, ema_beta: float = 0.1, bounds: List[float] = [1e-8, 1e6]):
        self.ema_beta = ema_beta
        self.bounds = bounds
        self.lambda_est = None
        
        # Previous values for secant computation
        self.prev_grad = None
        self.prev_w = None
        
    def update(self, grad: np.ndarray, w: np.ndarray) -> Optional[float]:
        """Update lambda estimate using secant approximation."""
        if self.prev_grad is not None and self.prev_w is not None:
            # Compute secant estimate
            grad_diff = grad - self.prev_grad
            w_diff = w - self.prev_w
            
            w_diff_norm_sq = np.linalg.norm(w_diff) ** 2
            
            if w_diff_norm_sq > 1e-12:
                lambda_new = np.dot(grad_diff, w_diff) / w_diff_norm_sq
                lambda_new = np.clip(lambda_new, self.bounds[0], self.bounds[1])
                
                # EMA update
                if self.lambda_est is None:
                    self.lambda_est = lambda_new
                else:
                    self.lambda_est = (1 - self.ema_beta) * self.lambda_est + self.ema_beta * lambda_new
                    
        # Store for next iteration
        self.prev_grad = grad.copy()
        self.prev_w = w.copy()
        
        return self.lambda_est


def get_synthetic_linear_stream(
    dim=20, 
    seed=42, 
    noise_std=0.1, 
    use_event_schema=True,
    # New M8 parameters
    eigs: Optional[List[float]] = None,
    cond_number: Optional[float] = None,
    rand_orth_seed: int = 42,
    path_type: str = "rotating",
    path_control: bool = True,
    strong_convexity_estimation: bool = True
):
    """Generate an infinite linear regression stream with configurable covariance.

    Parameters
    ----------
    dim : int
        Feature dimension.
    seed : int
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    use_event_schema : bool
        If True, emit event records. If False, emit legacy (x, y) tuples.
    eigs : list of float, optional
        Eigenvalues for covariance matrix. If None, use cond_number or uniform.
    cond_number : float, optional
        Condition number for geometric eigenvalue spacing.
    rand_orth_seed : int
        Seed for generating random orthogonal matrix.
    path_type : str
        Type of parameter path: "static", "rotating", or "drift".
    path_control : bool
        Whether to enable controlled parameter path.
    strong_convexity_estimation : bool
        Whether to estimate strong convexity parameter.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    
    # Initialize components
    cov_gen = CovarianceGenerator(dim, eigs, cond_number, rand_orth_seed)
    path_controller = ParameterPathController(dim, seed, path_type) if path_control else None
    sc_estimator = StrongConvexityEstimator() if strong_convexity_estimation else None
    
    if use_event_schema:
        return _generate_linear_stream_with_schema(
            cov_gen, path_controller, sc_estimator, noise_std, rng
        )
    else:
        # Legacy behavior - simplified
        X = rng.normal(size=(1000000, dim)).astype(np.float32)
        w_star = rng.normal(size=dim)
        noise = rng.normal(scale=noise_std, size=1000000).astype(np.float32)
        y = X @ w_star + noise
        return make_stream(X, y, mode="iid", seed=seed)


def _generate_linear_stream_with_schema(
    cov_gen: CovarianceGenerator,
    path_controller: Optional[ParameterPathController],
    sc_estimator: Optional[StrongConvexityEstimator],
    noise_std: float,
    rng: np.random.Generator
) -> Iterator[Dict[str, Any]]:
    """Generate infinite linear stream with event schema and diagnostics."""
    event_id = 0
    
    # Dummy parameters for gradient computation (would normally come from learning algorithm)
    w_learner = rng.normal(size=cov_gen.dim) * 0.1
    
    while True:
        # Generate data point
        x = cov_gen.sample(1)[0].astype(np.float32)
        
        # Get current ground-truth parameter
        if path_controller is not None:
            w_star, delta_P = path_controller.get_next_parameter()
            P_T_true = path_controller.P_T_cumulative
        else:
            w_star = rng.normal(size=cov_gen.dim)
            delta_P = 0.0
            P_T_true = 0.0
            
        # Generate target
        noise = rng.normal(scale=noise_std)
        y = float(x @ w_star + noise)
        
        # Compute gradient for strong convexity estimation
        lambda_est = None
        if sc_estimator is not None:
            # Simulate gradient: ∇f(w) = (w - w*)
            grad = w_learner - w_star
            lambda_est = sc_estimator.update(grad, w_learner)
            
            # Simple update to w_learner (for next iteration)
            w_learner = w_learner - 0.01 * grad
            
        # Create sample ID
        sample_id = f"linear_{event_id:06d}"
        
        # Create event record with diagnostics
        yield create_event_record_with_diagnostics(
            x=x,
            y=y,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=0,
            metrics={"delta_P": delta_P},
            lambda_est=lambda_est,
            P_T_true=P_T_true
        )
        
        event_id += 1
