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
    
    def step(self, event_type: str = "insert") -> Dict[str, float]:
        """Advance to next step and return privacy metrics. Only update spend on deletes."""
        self.t += 1
        
        # Only spend privacy budget on delete events
        if event_type == "delete":
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
        
        # Block control parameters
        self.B = 200  # Block size
        self.sum_log_ratio_PT = 0.0  # I-term accumulator
        
        # Initialize parameters based on path style
        if path_style == "rotating":
            # PT controller initialization from problem statement: 
            # theta = target_PT / ((target_D/2) * T)
            self.theta = targets.target_PT / ((targets.target_D / 2) * T)
            # Clamp theta to [1e-6, 0.3]
            self.theta = np.clip(self.theta, 1e-6, 0.3)
        elif path_style == "brownian":
            # For Brownian: E[P_T] ≈ T * E[||d_t||] ≈ T * σ_d * sqrt(d)
            # Solve for σ_d: σ_d = P_T / (T * sqrt(d))
            self.drift_std = targets.target_PT / (T * math.sqrt(dim))
        else:
            # Piecewise constant - no drift
            self.theta = 0.0
            self.drift_std = 0.0
        
        # Tracking
        self.P_T_cumulative = 0.0
        self.t = 0
        
        # Initialize parameter
        v = self.rng.normal(size=dim)
        v /= np.linalg.norm(v) + 1e-12
        
        # For rotating path style, ensure vector has significant components 
        # in first two dimensions to get predictable rotation behavior
        if path_style == "rotating":
            # Set first two components to dominate
            v_dominant = np.zeros(dim)
            v_dominant[0] = 0.8  # Major component
            v_dominant[1] = 0.6  # Minor component
            # Fill remaining with small noise
            if dim > 2:
                v_dominant[2:] = self.rng.normal(size=dim-2) * 0.1
            # Normalize and scale
            v_dominant /= np.linalg.norm(v_dominant)
            self.w_star = v_dominant * w_scale
        else:
            self.w_star = v * w_scale
    
    def get_next_parameter(self) -> tuple[np.ndarray, float]:
        """Get next parameter and path increment."""
        self.t += 1
        
        if self.path_style == "rotating":
            w_new = self._rotate_parameter()
        elif self.path_style == "brownian":
            w_new = self._drift_parameter()
            # Only enforce norm constraint for brownian motion (which adds noise)
            if np.linalg.norm(w_new) > 0:
                w_new = (w_new / np.linalg.norm(w_new)) * self.w_scale
        else:
            w_new = self.w_star.copy()
        
        # Compute path increment
        delta_P = np.linalg.norm(w_new - self.w_star)
        self.P_T_cumulative += delta_P
        
        # Block control (every B=200 events)
        if self.t % self.B == 0 and self.path_style == "rotating":
            ratio_PT = self.P_T_cumulative / (self.targets.target_PT * self.t / self.T)
            if ratio_PT > 0:
                # Update theta with P-control: theta *= ratio_PT ** (-0.5)
                self.theta *= ratio_PT ** (-0.5)
                
                # Optional I-term: multiply by exp(-0.1 * sum_log_ratio_PT)
                log_ratio = math.log(ratio_PT)
                self.sum_log_ratio_PT += log_ratio
                i_factor = math.exp(-0.1 * self.sum_log_ratio_PT)
                self.theta *= i_factor
                
                # Clamp theta to [1e-6, 0.3]
                self.theta = np.clip(self.theta, 1e-6, 0.3)
        
        self.w_star = w_new
        return self.w_star.copy(), float(delta_P)
    
    def _rotate_parameter(self) -> np.ndarray:
        """Apply controlled rotation using theta."""
        if self.dim >= 2:
            # Simple rotation in first two dimensions
            c, s = np.cos(self.theta), np.sin(self.theta)
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
        drift = self.rng.normal(scale=self.drift_std, size=self.dim)
        return self.w_star + drift


class GradientBoundController:
    """Controls gradient norms to enforce target G."""
    
    def __init__(self, targets: TheoryTargets):
        self.target_G = targets.target_G
        self.clip_count = 0
        self.total_count = 0
    
    def soft_clip(self, grad: np.ndarray, target_G: float) -> np.ndarray:
        """Apply soft clipping using tanh-based function."""
        grad_norm = np.linalg.norm(grad)
        if grad_norm == 0:
            return grad
        
        # Tanh-based soft clipping: scale = tanh(norm/target) * target / norm
        ratio = grad_norm / target_G
        if ratio <= 1.0:
            return grad  # No clipping needed
        else:
            # Soft transition using tanh
            scale_factor = np.tanh(1.0 / ratio) * target_G / grad_norm
            return grad * scale_factor
    
    def hard_cap(self, grad: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
        """Apply hard cap as backstop."""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            return grad * (max_norm / grad_norm), True
        return grad, False
    
    def clip_gradient(self, grad: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clip gradient with soft-clip then hard cap and return (clipped_grad, was_clipped)."""
        self.total_count += 1
        
        # First apply soft-clip
        grad_soft_clipped = self.soft_clip(grad, self.target_G)
        
        # Then apply hard cap as backstop at 1.05 * target_G
        grad_final, was_hard_clipped = self.hard_cap(grad_soft_clipped, 1.05 * self.target_G)
        
        if was_hard_clipped:
            self.clip_count += 1
        
        return grad_final, was_hard_clipped
    
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
        
        # ST controller parameters
        self.mu = 1.0  # Scale factor (default 1.0)
        self.B = 200   # Block size for control
        self.sum_log_ratio_ST = 0.0  # I-term accumulator
        
        # Target per-step energy
        self.target_per_step = targets.target_ST / T
    
    def apply_scaling(self, grad: np.ndarray) -> np.ndarray:
        """Apply mu scaling to gradient before clipping."""
        return self.mu * grad
    
    def update(self, grad_norm_sq: float) -> float:
        """Update running S_T and return adjustment factor."""
        self.t += 1
        self.ST_running += grad_norm_sq
        
        # Block control (every B=200 events)  
        if self.t % self.B == 0:
            ratio_ST = self.ST_running / (self.target_ST * self.t / self.T)
            old_mu = self.mu
            if ratio_ST > 0:
                # Update mu: mu *= ratio_ST ** (-0.25)
                self.mu *= ratio_ST ** (-0.25)
                
                # Optional I-term: multiply by exp(-0.05 * sum_log_ratio_ST)
                log_ratio = math.log(ratio_ST)
                self.sum_log_ratio_ST += log_ratio
                i_factor = math.exp(-0.05 * self.sum_log_ratio_ST)
                self.mu *= i_factor
                
                # Clamp mu to reasonable bounds
                self.mu = np.clip(self.mu, 0.1, 50.0)  # Allow larger mu values
        
        return self.mu
    
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
        if event_id > 0 and event_id % 50 == 0:  # Update covariance periodically
            # Use mu from ST controller for feature scaling  
            adjusted_feature_scale = feature_scale * st_ctrl.mu
            cov_gen = CovarianceGenerator(dim, eigs, None, seed + event_id, adjusted_feature_scale)
        
        # Generate feature vector
        x_raw = cov_gen.sample(1)[0].astype(np.float32)
        
        # Bound feature norm ||x_t|| via feature_scale
        x_norm_target = feature_scale * math.sqrt(dim)  # Expected norm for unit Gaussian
        x_norm_actual = np.linalg.norm(x_raw)
        if x_norm_actual > 2.0 * x_norm_target:  # Clip extreme outliers
            x = x_raw * (2.0 * x_norm_target / x_norm_actual)
        else:
            x = x_raw
        
        # Get current ground-truth parameter
        w_star, delta_P = path_ctrl.get_next_parameter()
        
        # Generate target with noise
        base_noise = rng.normal(scale=noise_std)
        y_raw = float(x @ w_star + base_noise)
        
        # Bound residual by r_eps (simple clamping)
        r_eps = 2.0 * noise_std  # Reasonable bound for residual
        residual_raw = x @ w_learner - y_raw
        if abs(residual_raw) > r_eps:
            # Adjust y to bring residual into bounds
            y = y_raw + np.sign(residual_raw) * (abs(residual_raw) - r_eps)
        else:
            y = y_raw
        
        # Compute gradient for bound enforcement
        # Direct approach: generate gradient with target energy per step
        target_grad_norm = math.sqrt(st_ctrl.target_per_step)
        
        # Generate base gradient from loss
        residual = x @ w_star - y  
        grad_base = x * residual + target_lambda * w_star
        
        # Scale to have target norm (before mu scaling)
        grad_base_norm = np.linalg.norm(grad_base)
        if grad_base_norm > 0:
            grad_raw = grad_base * (target_grad_norm / grad_base_norm)
        else:
            grad_raw = rng.normal(size=dim) * target_grad_norm / math.sqrt(dim)
        
        # Apply ST controller scaling: g <- mu * g (before clipping)
        grad_scaled = st_ctrl.apply_scaling(grad_raw)
        
        # Apply gradient clipping (soft-clip + hard cap)
        grad_clipped, was_clipped = grad_ctrl.clip_gradient(grad_scaled)
        
        # Update S_T controller with clipped gradient norm squared
        grad_norm_sq = np.linalg.norm(grad_clipped) ** 2
        st_ctrl.update(grad_norm_sq)
        
        # Update strong convexity estimation
        lambda_est = sc_estimator.update(grad_clipped, w_star)
        
        # Get privacy metrics (only spend on delete events)
        privacy_metrics = privacy_ctrl.step("insert")  # All events are inserts in theory stream
        
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
            g_norm=g_norm,
            clip_applied=was_clipped,
            ST_running=st_ctrl.ST_running,
            PT_target_residual=PT_target_residual,
            ST_target_residual=ST_target_residual,
            sigma_step=privacy_metrics["sigma_step"],
            privacy_spend_running=privacy_metrics.get("rho_spent", privacy_metrics.get("eps_spent", 0.0)),
        )
        
        event_id += 1