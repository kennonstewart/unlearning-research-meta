"""
Theory-first synthetic linear stream implementation.

This module implements a theory-driven data loader that enforces theoretical
constants as primary constraints, rather than exposing implementation details.
"""

import numpy as np
import math
from typing import Optional, Iterator, Dict, Any, List, Literal, Union
from dataclasses import dataclass

try:
    from .utils import set_global_seed
    from .event_schema import create_event_record_with_diagnostics
    from .linear import CovarianceGenerator, StrongConvexityEstimator
except ImportError:
    from utils import set_global_seed
    from event_schema import create_event_record_with_diagnostics
    from linear import CovarianceGenerator, StrongConvexityEstimator


@dataclass
class TheoryTargets:
    """Theory constants that should be enforced by the stream."""
    target_G: float  # gradient norm bound
    target_D: float  # parameter/domain diameter
    target_c: float  # min inverse-Hessian eigenvalue (LBFGS clamp)
    target_C: float  # max inverse-Hessian eigenvalue (LBFGS clamp)
    target_lambda: float  # strong convexity of loss
    target_PT: float  # desired total path length over horizon T
    target_ST: float  # desired cumulative squared gradient over T
    accountant: Literal["zcdp", "eps-delta"]
    rho_total: Optional[float] = None
    eps_total: Optional[float] = None
    delta_total: Optional[float] = None


class PrivacyController:
    """Controls privacy budget and noise schedule."""
    
    def __init__(self, targets: TheoryTargets, T: int, seed: int = 42):
        self.targets = targets
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.rho_spent = 0.0
        self.eps_spent = 0.0
        self.t = 0
        
        # Compute per-step budget
        if targets.accountant == "zcdp" and targets.rho_total is not None:
            self.rho_step = targets.rho_total / T
            self.sigma_step = self._compute_zcdp_sigma(self.rho_step)
        elif targets.accountant == "eps-delta" and targets.eps_total is not None:
            # Convert (ε,δ) to zCDP approximation for simplicity
            # Using standard conversion: ρ ≈ ε²/(2*log(1/δ))
            if targets.delta_total is not None and targets.delta_total > 0:
                self.rho_total = targets.eps_total**2 / (2 * math.log(1/targets.delta_total))
                self.rho_step = self.rho_total / T
                self.sigma_step = self._compute_zcdp_sigma(self.rho_step)
            else:
                # Fallback for pure ε-DP
                self.sigma_step = 1.0  # Default noise scale
        else:
            self.sigma_step = 1.0  # Default noise scale
    
    def _compute_zcdp_sigma(self, rho: float, sensitivity: float = 1.0) -> float:
        """Compute noise scale for zCDP guarantee."""
        if rho <= 0:
            return float('inf')
        return sensitivity / math.sqrt(2 * rho)
    
    def get_noise_scale(self, sensitivity: float = 1.0) -> float:
        """Get current noise scale for given sensitivity."""
        return self.sigma_step * sensitivity
    
    def step(self) -> Dict[str, float]:
        """Advance to next step and return privacy metrics."""
        self.t += 1
        if hasattr(self, 'rho_step'):
            self.rho_spent += self.rho_step
            if self.targets.accountant == "eps-delta":
                # Convert back to (ε,δ) for reporting
                if self.targets.delta_total is not None:
                    self.eps_spent = math.sqrt(2 * self.rho_spent * math.log(1/self.targets.delta_total))
        
        return {
            "sigma_step": self.sigma_step,
            "rho_spent": getattr(self, 'rho_spent', 0.0),
            "eps_spent": getattr(self, 'eps_spent', 0.0),
        }


class PathLengthController:
    """Controls parameter path to enforce target P_T."""
    
    def __init__(self, targets: TheoryTargets, T: int, dim: int, 
                 path_style: Literal["piecewise-constant", "rotating", "brownian"],
                 w_scale: float, seed: int = 42):
        self.targets = targets
        self.T = T
        self.dim = dim
        self.path_style = path_style
        self.w_scale = w_scale
        self.rng = np.random.default_rng(seed)
        
        # Compute per-step drift budget
        self.delta_P_budget = targets.target_PT / T
        
        # Initialize parameters based on path style
        if path_style == "rotating":
            # For rotation: P_T ≈ T * 2 * ||w|| * sin(θ/2)
            # Solve for θ: θ = 2 * arcsin(P_T / (T * 2 * ||w||))
            sin_half_theta = targets.target_PT / (T * 2 * w_scale)
            sin_half_theta = min(sin_half_theta, 0.99)  # Clamp to feasible range, leave some margin
            if sin_half_theta <= 0:
                self.rotate_angle = 0.0
            else:
                self.rotate_angle = 2 * math.asin(sin_half_theta)
        elif path_style == "brownian":
            # For Brownian: E[P_T] ≈ T * E[||d_t||] ≈ T * σ_d * sqrt(d)
            # Solve for σ_d: σ_d = P_T / (T * sqrt(d))
            self.drift_std = targets.target_PT / (T * math.sqrt(dim))
        else:
            # Piecewise constant - no drift
            self.rotate_angle = 0.0
            self.drift_std = 0.0
        
        # Tracking
        self.P_T_cumulative = 0.0
        self.t = 0
        
        # Initialize parameter
        v = self.rng.normal(size=dim)
        v /= np.linalg.norm(v) + 1e-12
        self.w_star = v * w_scale
        
        # Adaptive control for path length - more conservative
        self.adaptive_factor = 1.0
        self.check_interval = max(10, T // 20)  # Check less frequently
    
    def get_next_parameter(self) -> tuple[np.ndarray, float]:
        """Get next parameter and path increment."""
        self.t += 1
        
        if self.path_style == "rotating":
            w_new = self._rotate_parameter()
        elif self.path_style == "brownian":
            w_new = self._drift_parameter()
        else:
            w_new = self.w_star.copy()
        
        # Enforce norm constraint
        if np.linalg.norm(w_new) > 0:
            w_new = (w_new / np.linalg.norm(w_new)) * self.w_scale
        
        # Compute path increment
        delta_P = np.linalg.norm(w_new - self.w_star)
        self.P_T_cumulative += delta_P
        
        # Adaptive adjustment for path length (gentle feedback control)
        if self.t > 50 and self.t % self.check_interval == 0:  # Check less frequently and start later
            expected_PT_so_far = self.targets.target_PT * self.t / self.T
            if expected_PT_so_far > 0:
                current_ratio = self.P_T_cumulative / expected_PT_so_far
                if current_ratio < 0.7:  # Significantly too slow
                    self.adaptive_factor *= 1.1
                elif current_ratio > 1.3:  # Significantly too fast
                    self.adaptive_factor *= 0.9
                self.adaptive_factor = np.clip(self.adaptive_factor, 0.8, 1.5)
        
        self.w_star = w_new
        return self.w_star.copy(), float(delta_P)
    
    def _rotate_parameter(self) -> np.ndarray:
        """Apply controlled rotation."""
        if self.dim >= 2:
            angle = self.rotate_angle * self.adaptive_factor
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
        """Apply Brownian drift."""
        drift = self.rng.normal(scale=self.drift_std * self.adaptive_factor, size=self.dim)
        return self.w_star + drift


class GradientBoundController:
    """Controls gradient norms to enforce target G."""
    
    def __init__(self, targets: TheoryTargets):
        self.target_G = targets.target_G
        self.clip_count = 0
        self.total_count = 0
    
    def clip_gradient(self, grad: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clip gradient to enforce bound and return (clipped_grad, was_clipped)."""
        self.total_count += 1
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm > self.target_G:
            self.clip_count += 1
            clipped_grad = grad * (self.target_G / grad_norm)
            return clipped_grad, True
        else:
            return grad, False
    
    def get_clip_rate(self) -> float:
        """Get current clipping rate."""
        if self.total_count == 0:
            return 0.0
        return self.clip_count / self.total_count


class CurvaturePlanner:
    """Plans covariance and regularization to achieve target curvature properties."""
    
    @staticmethod
    def derive_eigs_for_lambda(dim: int, target_lambda: float, cond_number: float) -> List[float]:
        """Derive eigenvalues that ensure min eigenvalue of Hessian ≈ target_lambda."""
        # For H = E[xx^T] + λI, we want λ_min(H) ≈ target_lambda
        # If Σ has eigenvalues σ_i, then H has eigenvalues σ_i + regularization
        # Set regularization = 0 and scale Σ eigenvalues to achieve target
        max_eig = target_lambda * cond_number
        eigs = np.geomspace(max_eig, target_lambda, dim).tolist()
        return eigs


class AdaGradEnergyController:
    """Controls feature scaling and noise to enforce target S_T."""
    
    def __init__(self, targets: TheoryTargets, T: int):
        self.target_ST = targets.target_ST
        self.T = T
        self.ST_running = 0.0
        self.t = 0
        
        # Target per-step energy
        self.target_per_step = targets.target_ST / T
        
        # Adjustment factor (simple feedback control)
        self.adjustment_factor = 1.0
    
    def update(self, grad_norm_sq: float) -> float:
        """Update running S_T and return adjustment factor."""
        self.t += 1
        self.ST_running += grad_norm_sq
        
        # Simple feedback: if we're off track, adjust scaling
        if self.t > 20:  # Wait for some samples
            expected_ST_so_far = self.target_per_step * self.t
            if expected_ST_so_far > 0:
                current_ratio = self.ST_running / expected_ST_so_far
                
                # More aggressive feedback control
                if current_ratio < 0.8:  # Too low
                    self.adjustment_factor *= 1.02
                elif current_ratio > 1.2:  # Too high
                    self.adjustment_factor *= 0.98
                
                # Clamp adjustment factor
                self.adjustment_factor = np.clip(self.adjustment_factor, 0.1, 10.0)
        
        return self.adjustment_factor
    
    def get_target_residual(self) -> float:
        """Get relative error vs target."""
        if self.t == 0:
            return 0.0
        expected_ST = self.target_per_step * self.t
        return (self.ST_running - expected_ST) / max(expected_ST, 1e-6)


def get_theory_stream(
    dim: int,
    T: int,
    # Theoretical targets (hard or soft constraints)
    target_G: float,                # gradient norm bound
    target_D: float,                # parameter/domain diameter
    target_c: float,                # min inverse-Hessian eigenvalue (LBFGS clamp)
    target_C: float,                # max inverse-Hessian eigenvalue (LBFGS clamp)
    target_lambda: float,           # strong convexity of loss
    target_PT: float,               # desired total path length over horizon T
    target_ST: float,               # desired cumulative squared gradient over T
    # Privacy / accounting
    accountant: Literal["zcdp","eps-delta"],
    rho_total: Optional[float] = None,
    eps_total: Optional[float] = None,
    delta_total: Optional[float] = None,
    # Stream style (choose path shape, but scale to hit PT)
    path_style: Literal["piecewise-constant","rotating","brownian"] = "rotating",
    segments: Optional[List[dict]] = None,   # optional piecewise schedule for abrupt/periodic drift
    # Geometry controls (optional overrides)
    cond_number: Optional[float] = None,
    eigs: Optional[List[float]] = None,
    feature_scale: float = 1.0,
    fix_w_norm: bool = True,
    seed: int = 42,
    # Tolerances
    tol_rel: float = 0.05,
    # Additional parameters
    noise_std: float = 0.1,
    use_event_schema: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Generate theory-first synthetic linear stream.
    
    This function creates a synthetic linear regression stream where the primary
    inputs are theoretical constants that are enforced as constraints, rather
    than implementation details.
    
    Parameters
    ----------
    dim : int
        Feature dimension
    T : int
        Horizon (number of steps)
    target_G : float
        Gradient norm bound to enforce
    target_D : float
        Parameter domain diameter to enforce
    target_c : float
        Minimum inverse-Hessian eigenvalue for LBFGS clamp
    target_C : float
        Maximum inverse-Hessian eigenvalue for LBFGS clamp
    target_lambda : float
        Strong convexity parameter to enforce
    target_PT : float
        Total path length to achieve over horizon
    target_ST : float
        Cumulative squared gradient norm to achieve
    accountant : {"zcdp", "eps-delta"}
        Privacy accounting mechanism
    rho_total : float, optional
        Total zCDP budget (if accountant="zcdp")
    eps_total : float, optional
        Total epsilon budget (if accountant="eps-delta")
    delta_total : float, optional
        Delta parameter for (ε,δ)-DP
    path_style : {"piecewise-constant", "rotating", "brownian"}
        Style of parameter path
    segments : list of dict, optional
        Piecewise schedule for path segments
    cond_number : float, optional
        Condition number for covariance matrix
    eigs : list of float, optional
        Explicit eigenvalues for covariance matrix
    feature_scale : float
        Base feature scaling factor
    fix_w_norm : bool
        Whether to maintain fixed parameter norm
    seed : int
        Random seed
    tol_rel : float
        Relative tolerance for target enforcement
    noise_std : float
        Base noise standard deviation
    use_event_schema : bool
        Whether to emit event schema records
        
    Returns
    -------
    Iterator[Dict[str, Any]]
        Stream of event records with enforced theoretical properties
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    
    # Create theory targets structure
    targets = TheoryTargets(
        target_G=target_G,
        target_D=target_D,
        target_c=target_c,
        target_C=target_C,
        target_lambda=target_lambda,
        target_PT=target_PT,
        target_ST=target_ST,
        accountant=accountant,
        rho_total=rho_total,
        eps_total=eps_total,
        delta_total=delta_total,
    )
    
    # Initialize controllers
    privacy_ctrl = PrivacyController(targets, T, seed)
    
    # Determine parameter scale from diameter constraint
    w_scale = target_D / 2  # Radius of ball with diameter D
    
    path_ctrl = PathLengthController(targets, T, dim, path_style, w_scale, seed)
    grad_ctrl = GradientBoundController(targets)
    st_ctrl = AdaGradEnergyController(targets, T)
    
    # Set up covariance with curvature planning
    if eigs is None:
        if cond_number is None:
            cond_number = target_C / target_c  # Derive from clamp targets
        eigs = CurvaturePlanner.derive_eigs_for_lambda(dim, target_lambda, cond_number)
    
    # Start with base feature scaling
    current_feature_scale = feature_scale
    
    cov_gen = CovarianceGenerator(dim, eigs, None, seed, current_feature_scale)
    sc_estimator = StrongConvexityEstimator()
    
    # Emit theory targets block every K events (and at start)
    theory_targets_block = {
        "target_G": target_G,
        "target_D": target_D,
        "target_c": target_c,
        "target_C": target_C,
        "target_lambda": target_lambda,
        "target_PT": target_PT,
        "target_ST": target_ST,
        "accountant": accountant,
        "rho_total": rho_total,
        "eps_total": eps_total,
        "delta_total": delta_total,
    }
    
    # Main generation loop
    event_id = 0
    w_learner = rng.normal(size=dim) * 0.1  # Dummy learner parameter
    
    while True:
        # Update feature scaling based on S_T control
        st_adjustment = st_ctrl.adjustment_factor
        if event_id > 0 and event_id % 50 == 0:  # Update covariance periodically
            adjusted_feature_scale = feature_scale * st_adjustment
            cov_gen = CovarianceGenerator(dim, eigs, None, seed + event_id, adjusted_feature_scale)
        
        # Generate feature vector
        x = cov_gen.sample(1)[0].astype(np.float32)
        
        # Get current ground-truth parameter
        w_star, delta_P = path_ctrl.get_next_parameter()
        
        # Generate target with noise
        base_noise = rng.normal(scale=noise_std)
        y = float(x @ w_star + base_noise)
        
        # Compute gradient for bound enforcement
        # For squared loss: ∇f(w) = x(x^T w - y) + λw
        residual = x @ w_learner - y
        grad_raw = x * residual + target_lambda * w_learner
        grad_clipped, was_clipped = grad_ctrl.clip_gradient(grad_raw)
        
        # Update S_T controller
        grad_norm_sq = np.linalg.norm(grad_clipped) ** 2
        st_ctrl.update(grad_norm_sq)
        
        # Update strong convexity estimation
        lambda_est = sc_estimator.update(grad_clipped, w_learner)
        
        # Simple learner update
        w_learner = w_learner - 0.01 * grad_clipped
        
        # Get privacy metrics
        privacy_metrics = privacy_ctrl.step()
        
        # Compute diagnostics
        x_norm = float(np.linalg.norm(x))
        w_star_norm = float(np.linalg.norm(w_star))
        g_norm = float(np.linalg.norm(grad_clipped))
        clip_rate = grad_ctrl.get_clip_rate()
        PT_target_residual = (path_ctrl.P_T_cumulative - targets.target_PT * (event_id + 1) / T) / targets.target_PT
        ST_target_residual = st_ctrl.get_target_residual()
        
        # Create sample ID
        sample_id = f"theory_{event_id:06d}"
        
        # Create metrics dict
        metrics = {
            # Original metrics
            "delta_P": delta_P,
            "x_norm": x_norm,
            "w_star_norm": w_star_norm,
            "noise": float(base_noise),
            # Theory targets (emitted periodically)
            "theory_targets": theory_targets_block if event_id % 1000 == 0 else None,
            # New theory metrics
            "g_norm": g_norm,
            "clip_applied": was_clipped,
            "ST_running": st_ctrl.ST_running,
            "PT_target_residual": PT_target_residual,
            "ST_target_residual": ST_target_residual,
            "sigma_step": privacy_metrics["sigma_step"],
            "privacy_spend_running": privacy_metrics.get("rho_spent", privacy_metrics.get("eps_spent", 0.0)),
            # Passthrough for LBFGS
            "G_hat": target_G,
            "D_hat": target_D,
            "c_hat": target_c,
            "C_hat": target_C,
        }
        
        # Create event record
        yield create_event_record_with_diagnostics(
            x=x,
            y=y,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=0,
            metrics=metrics,
            lambda_est=float(lambda_est) if lambda_est is not None else None,
            P_T_true=float(path_ctrl.P_T_cumulative),
        )
        
        event_id += 1