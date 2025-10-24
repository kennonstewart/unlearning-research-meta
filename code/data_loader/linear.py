import numpy as np
from typing import Optional, Iterator, Dict, Any, List

try:
    from .utils import set_global_seed
    from .event_schema import create_event_record_with_diagnostics
except ImportError:
    from utils import set_global_seed
    from event_schema import create_event_record_with_diagnostics


class ParameterPathController:
    """Controls the ground-truth parameter path. Only rotating path is supported.

    The controller rotates the parameter in the first two coordinates by a small
    angle each step, optionally keeping the norm fixed to a provided scale.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 42,
        rotate_angle: float = 0.01,
        w_scale: Optional[float] = None,
        fix_w_norm: bool = True,
    ):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        # Only rotating path supported
        self.rotate_angle = rotate_angle
        self.fix_w_norm = fix_w_norm

        # Initialize parameter (optionally with controlled norm)
        if w_scale is None:
            # Random initialization; keep its norm as w_scale
            self.w_star = self.rng.normal(size=dim)
            self.w_scale = float(np.linalg.norm(self.w_star))
        else:
            v = self.rng.normal(size=dim)
            v /= np.linalg.norm(v) + 1e-12
            self.w_star = v * float(w_scale)
            self.w_scale = float(w_scale)

        self.t = 0
        self.P_T_cumulative = 0.0

    def get_next_parameter(self) -> tuple[np.ndarray, float]:
        """Return the next w* and the path increment delta_P."""
        self.t += 1

        # Always apply rotation
        w_new = self._rotate_parameter()

        if self.fix_w_norm and np.linalg.norm(w_new) > 0:
            w_new = (w_new / np.linalg.norm(w_new)) * self.w_scale

        delta_P = float(np.linalg.norm(w_new - self.w_star))
        self.P_T_cumulative += delta_P

        self.w_star = w_new
        return self.w_star.copy(), delta_P

    def _rotate_parameter(self) -> np.ndarray:
        """Rotate the parameter by self.rotate_angle in the first two dims.

        If dim < 2 the parameter is unchanged.
        """
        angle = float(self.rotate_angle)
        if self.dim >= 2:
            c, s = np.cos(angle), np.sin(angle)
            R = np.eye(self.dim)
            R[0, 0] = c
            R[0, 1] = -s
            R[1, 0] = s
            R[1, 1] = c
            return R @ self.w_star
        return self.w_star.copy()


class CovarianceGenerator:
    """Generate Gaussian features with configurable covariance.

    The covariance is constructed as Q diag(eigs) Q^T where Q is a random
    orthogonal matrix. If eigs is not provided, eigenvalues are set by
    geometric spacing using cond_number.
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
        self.feature_scale = float(feature_scale)

        if eigs is not None:
            if len(eigs) != dim:
                raise ValueError(f"eigs must have length {dim}, got {len(eigs)}")
            self.eigenvalues = np.array(eigs, dtype=float)
        elif cond_number is not None:
            cn = max(float(cond_number), 1e-12)
            # Geometric spacing from 1 to 1/cond_number
            self.eigenvalues = np.geomspace(1.0, 1.0 / cn, dim)
        else:
            self.eigenvalues = np.ones(dim, dtype=float)

        # Random orthogonal basis
        A = self.rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)
        self.Q = Q

        self.Sigma = self.Q @ np.diag(self.eigenvalues) @ self.Q.T
        # Cholesky may fail if Sigma is not positive definite due to numerical
        # issues; add small jitter if necessary.
        jitter = 1e-12
        try:
            self.L = np.linalg.cholesky(self.Sigma)
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(self.Sigma + jitter * np.eye(dim))

    def sample(self, n: int) -> np.ndarray:
        z = self.rng.normal(size=(n, self.dim))
        return (z @ self.L.T) * self.feature_scale


class StrongConvexityEstimator:
    """Online secant-based estimate of strong convexity (lambda).

    This estimator simulates the earlier behavior used in diagnostics/tests.
    """

    def __init__(self, ema_beta: float = 0.1, bounds: List[float] = [1e-8, 1e6]):
        self.ema_beta = float(ema_beta)
        self.bounds = bounds
        self.lambda_est: Optional[float] = None
        self.prev_grad: Optional[np.ndarray] = None
        self.prev_w: Optional[np.ndarray] = None

    def update(self, grad: np.ndarray, w: np.ndarray) -> Optional[float]:
        if self.prev_grad is not None and self.prev_w is not None:
            grad_diff = grad - self.prev_grad
            w_diff = w - self.prev_w
            w_diff_norm_sq = float(np.linalg.norm(w_diff) ** 2)
            if w_diff_norm_sq > 1e-12:
                lambda_new = float(np.dot(grad_diff, w_diff) / w_diff_norm_sq)
                lambda_new = float(np.clip(lambda_new, self.bounds[0], self.bounds[1]))
                if self.lambda_est is None:
                    self.lambda_est = lambda_new
                else:
                    self.lambda_est = (
                        1 - self.ema_beta
                    ) * self.lambda_est + self.ema_beta * lambda_new

        self.prev_grad = grad.copy()
        self.prev_w = w.copy()
        return self.lambda_est


def _generate_linear_stream_with_schema(
    cov_gen: CovarianceGenerator,
    path_controller: ParameterPathController,
    sc_estimator: Optional[StrongConvexityEstimator],
    noise_std: float,
    rng: np.random.Generator,
    rotate_angle: Optional[float] = None,
    drift_rate: Optional[float] = None,
    G_hat: Optional[float] = None,
    D_hat: Optional[float] = None,
    c_hat: Optional[float] = None,
    C_hat: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Infinite generator that yields event records with diagnostics."""
    event_id = 0
    w_learner = rng.normal(size=cov_gen.dim) * 0.1

    while True:
        x = cov_gen.sample(1)[0].astype(np.float32)
        w_star, delta_P = path_controller.get_next_parameter()
        P_T_true = path_controller.P_T_cumulative

        noise = float(rng.normal(scale=noise_std))
        y = float(x @ w_star + noise)

        x_norm = float(np.linalg.norm(x))
        w_star_norm = float(np.linalg.norm(w_star))

        lambda_est = None
        if sc_estimator is not None:
            grad = w_learner - w_star
            lambda_est = sc_estimator.update(grad, w_learner)
            w_learner = w_learner - 0.01 * grad

        sample_id = f"linear_{event_id:06d}"

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
                "noise": noise,
                "rotate_angle": rotate_angle,
                "drift_rate": drift_rate,
                "G_hat": G_hat,
                "D_hat": D_hat,
                "c_hat": c_hat,
                "C_hat": C_hat,
            },
            lambda_est=float(lambda_est) if lambda_est is not None else None,
            P_T_true=float(P_T_true) if P_T_true is not None else None,
        )

        event_id += 1


def get_rotating_linear_stream(
    dim: int = 20,
    seed: int = 42,
    noise_std: float = 0.1,
    cond_number: float = 10.0,
    feature_scale: float = 1.0,
    rotate_angle: float = 0.01,
    w_scale: Optional[float] = None,
    fix_w_norm: bool = True,
    strong_convexity_estimation: bool = True,
    G_hat: Optional[float] = None,
    D_hat: Optional[float] = None,
    c_hat: Optional[float] = None,
    C_hat: Optional[float] = None,
    _legacy_drift_rate: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """Rotating-only linear stream with unified seeding."""
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    cov_gen = CovarianceGenerator(
        dim=dim,
        eigs=None,
        cond_number=cond_number,
        rand_orth_seed=seed,
        feature_scale=feature_scale,
    )

    path_controller = ParameterPathController(
        dim=dim,
        seed=seed,
        rotate_angle=rotate_angle,
        w_scale=w_scale,
        fix_w_norm=fix_w_norm,
    )

    sc_estimator = StrongConvexityEstimator() if strong_convexity_estimation else None

    return _generate_linear_stream_with_schema(
        cov_gen,
        path_controller,
        sc_estimator,
        noise_std,
        rng,
        rotate_angle=rotate_angle,
        drift_rate=_legacy_drift_rate,
        G_hat=G_hat,
        D_hat=D_hat,
        c_hat=c_hat,
        C_hat=C_hat,
    )


# Helper functions for PT controllers (kept for convenience)
def set_rotation_by_PT(T: int, w_norm: float, target_PT: float) -> float:
    import math

    sin_half_theta = target_PT / (T * 2 * w_norm)
    sin_half_theta = min(max(sin_half_theta, 0.0), 0.99)
    if sin_half_theta <= 0:
        return 0.0
    theta = 2 * math.asin(sin_half_theta)
    return float(np.clip(theta, 1e-6, 0.3))


def set_brownian_by_PT(T: int, dim: int, target_PT: float) -> float:
    import math

    drift_std = target_PT / (T * math.sqrt(dim))
    return float(max(drift_std, 0.0))
