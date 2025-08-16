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
    """Control the path of ground-truth parameters for controlled path-length.

    Supports configurable rotation angle, drift rate, and optional norm control
    of the parameter vector w* via w_scale and fix_w_norm.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 42,
        path_type: str = "rotating",
        rotate_angle: float = 0.01,
        drift_rate: float = 0.001,
        w_scale: Optional[float] = None,
        fix_w_norm: bool = True,
    ):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.path_type = path_type
        self.rotate_angle = rotate_angle
        self.drift_rate = drift_rate
        self.fix_w_norm = fix_w_norm

        # Initialize on a sphere of radius w_scale
        v = self.rng.normal(size=dim)
        v /= np.linalg.norm(v) + 1e-12
        self.w_star = v * w_scale
        self.w_scale = float(w_scale)
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

        if self.fix_w_norm and np.linalg.norm(w_new) > 0:
            w_new = (w_new / np.linalg.norm(w_new)) * self.w_scale

        # Compute path increment
        delta_P = np.linalg.norm(w_new - self.w_star)
        self.P_T_cumulative += delta_P

        self.w_star = w_new
        return self.w_star.copy(), delta_P

    def _rotate_parameter(self) -> np.ndarray:
        """Apply controlled rotation to parameter."""
        # Rotate by small angle (self.rotate_angle radians per step)
        angle = self.rotate_angle
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
        drift_rate = self.drift_rate
        drift = self.rng.normal(scale=drift_rate, size=self.dim)
        return self.w_star + drift


class CovarianceGenerator:
    """Generate Gaussian data with configurable covariance structure.

    Supports configurable feature scaling.
    """

    def __init__(
        self,
        dim: int,
        eigs: Optional[List[float]] = None,
        cond_number: Optional[float] = None,
        rand_orth_seed: int = 42,
        feature_scale: float = 1.0,
    ):
        self.dim = dim
        self.rng = np.random.default_rng(rand_orth_seed)
        self.feature_scale = feature_scale

        # Generate eigenvalues
        if eigs is not None:
            if len(eigs) != dim:
                raise ValueError(f"eigs must have length {dim}, got {len(eigs)}")
            self.eigenvalues = np.array(eigs)
        elif cond_number is not None:
            # Geometric spacing from 1 to 1/cond_number
            self.eigenvalues = np.geomspace(1.0, 1.0 / cond_number, dim)
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
        return (z @ self.L.T) * self.feature_scale


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
                    self.lambda_est = (
                        1 - self.ema_beta
                    ) * self.lambda_est + self.ema_beta * lambda_new

        # Store for next iteration
        self.prev_grad = grad.copy()
        self.prev_w = w.copy()

        return self.lambda_est


def get_synthetic_linear_stream(
    dim=20,
    seed=42,
    noise_std=0.1,
    use_event_schema=True,
    # Covariance/feature scaling
    eigs: Optional[List[float]] = None,
    cond_number: Optional[float] = None,
    rand_orth_seed: int = 42,
    feature_scale: float = 1.0,
    # Parameter path controls
    path_type: str = "rotating",
    path_control: bool = True,
    rotate_angle: float = 0.01,
    drift_rate: float = 0.001,
    w_scale: Optional[float] = None,
    fix_w_norm: bool = True,
    # Estimation diagnostics
    strong_convexity_estimation: bool = True,
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
    feature_scale : float
        Scaling applied to features after covariance generation.
    path_type : str
        Type of parameter path: "static", "rotating", or "drift".
    path_control : bool
        Whether to enable controlled parameter path.
    rotate_angle : float
        Rotation angle per step in radians for rotating path.
    drift_rate : float
        Magnitude of drift per step for drifting path.
    w_scale : float or None
        Norm to fix w* to, if fix_w_norm is True. If None, use initial norm.
    fix_w_norm : bool
        Whether to keep w* norm fixed during path updates.
    strong_convexity_estimation : bool
        Whether to estimate strong convexity parameter.

    Returns
    -------
    Iterator or Stream
        Infinite stream of data points with event schema or legacy tuples.

    Notes
    -----
    Diagnostics include metrics related to parameter drift, feature norms, and noise.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    # Initialize components
    cov_gen = CovarianceGenerator(dim, eigs, cond_number, rand_orth_seed, feature_scale)
    path_controller = (
        ParameterPathController(
            dim,
            seed,
            path_type,
            rotate_angle=rotate_angle,
            drift_rate=drift_rate,
            w_scale=w_scale,
            fix_w_norm=fix_w_norm,
        )
        if path_control
        else None
    )
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
    rng: np.random.Generator,
) -> Iterator[Dict[str, Any]]:
    """Generate infinite linear stream with event schema and diagnostics.

    Includes metrics for drift magnitude, feature norm, parameter norm, and noise.
    """
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

        # Compute diagnostics
        x_norm = float(np.linalg.norm(x))
        w_star_norm = float(np.linalg.norm(w_star))
        noise_draw = float(noise)

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
            metrics={
                "delta_P": delta_P,
                "x_norm": x_norm,
                "w_star_norm": w_star_norm,
                "noise": noise_draw,
            },
            lambda_est=lambda_est,
            P_T_true=P_T_true,
        )

        event_id += 1
