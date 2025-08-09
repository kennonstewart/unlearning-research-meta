"""
Calibrator module for estimating constants from warmup dynamics.

This module implements the Calibrator helper class that collects gradients 
and parameter snapshots during a bootstrap phase, estimates theoretical 
constants (G, D, c, C), and calculates sample complexity N*.
"""

import numpy as np
from typing import Optional, Dict, Any


class Calibrator:
    """
    Calibrator helper for estimating theoretical constants from warmup dynamics.
    
    During the calibration phase, this class tracks:
    - Gradient norms: ||∇ℓ_t(w_t)||_2 to estimate G
    - Parameter trajectory: w_t to compute hypothesis diameter D  
    - L-BFGS curvature bounds: eigenvalue bounds (c, C) of Hessian approximation
    - EMA tracking: Exponential moving average of gradient norms for drift detection
    
    After collection, it computes the sample complexity N* = ⌈(G·D·√(cC)/γ)²⌉
    
    Attributes:
        grad_norms (List[float]): Collected gradient norms during calibration
        thetas (List[np.ndarray]): Parameter snapshots during calibration  
        quantile (float): Quantile for robust G estimation (default 0.95)
        D_cap (float): Upper bound for hypothesis diameter (default 10.0)
        c_hat (Optional[float]): Estimated lower curvature bound
        C_hat (Optional[float]): Estimated upper curvature bound
        ema_beta (float): EMA decay parameter for drift detection
        G_ema (float): Exponential moving average of gradient norms
        G_ema_window (List[float]): Recent EMA values for drift detection
        finalized_G (Optional[float]): G value used in last finalization
    """
    
    def __init__(self, quantile: float = 0.95, D_cap: float = 10.0, ema_beta: float = 0.9):
        """
        Initialize the Calibrator.
        
        Args:
            quantile: Quantile for gradient norm estimation (e.g., 0.95 to clip outliers)
            D_cap: Upper bound for hypothesis diameter to prevent extreme values
            ema_beta: EMA decay parameter (higher = more smoothing)
        """
        self.grad_norms = []
        self.thetas = []
        self.quantile = quantile
        self.D_cap = D_cap
        self.c_hat: Optional[float] = None
        self.C_hat: Optional[float] = None
        
        # EMA tracking for adaptive recalibration
        self.ema_beta = ema_beta
        self.G_ema: Optional[float] = None
        self.G_ema_window = []  # Store recent EMA values
        self.finalized_G: Optional[float] = None

        # Live bounds estimation 
        self.G_hat_live: Optional[float] = None
        self.D_hat_live: Optional[float] = None
        self.theta_0_live: Optional[np.ndarray] = None
        self.grad_norm_buffer = []  # Small window for trimmed quantile
        self.theta_norm_buffer = []  # Small window for parameter norms
        self.buffer_size = 100  # Size of rolling window

    def observe(self, grad: np.ndarray, theta: np.ndarray) -> None:
        """
        Record gradient and parameter snapshot during calibration.
        
        Args:
            grad: Gradient vector ∇ℓ_t(w_t)
            theta: Current parameter vector w_t
        """
        grad_norm = np.linalg.norm(grad)
        self.grad_norms.append(grad_norm)
        self.thetas.append(theta.copy())
        
        # Update EMA of gradient norms
        if self.G_ema is None:
            self.G_ema = grad_norm
        else:
            self.G_ema = self.ema_beta * self.G_ema + (1 - self.ema_beta) * grad_norm

    def observe_ongoing(self, grad: np.ndarray) -> None:
        """
        Update EMA tracking during ongoing learning (after calibration).
        
        Args:
            grad: Gradient vector ∇ℓ_t(w_t)
        """
        grad_norm = np.linalg.norm(grad)
        
        if self.G_ema is None:
            self.G_ema = grad_norm
        else:
            self.G_ema = self.ema_beta * self.G_ema + (1 - self.ema_beta) * grad_norm
        
        # Keep a sliding window of recent EMA values
        self.G_ema_window.append(self.G_ema)
        if len(self.G_ema_window) > 1000:  # Limit window size
            self.G_ema_window.pop(0)

    def estimate_bounds(self, model) -> tuple[float, float]:
        """
        Estimate eigenvalue bounds (c, C) of the L-BFGS Hessian approximation.
        
        Tries to access curvature information from the model in the following order:
        1. model.lbfgs_bounds() -> direct bounds method
        2. model.lbfgs.B_matrix() -> compute eigenvalues of B matrix
        3. Fallback to conservative (1.0, 1.0)
        
        Args:
            model: Model with L-BFGS optimizer that may expose curvature bounds
            
        Returns:
            Tuple of (c_hat, C_hat) representing lower and upper eigenvalue bounds
        """
        # Try direct bounds method
        if hasattr(model, "lbfgs_bounds"):
            try:
                bounds = model.lbfgs_bounds()
                if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                    return float(bounds[0]), float(bounds[1])
            except Exception:
                pass
        
        # Try to extract eigenvalues from B matrix
        if hasattr(model, "lbfgs") and hasattr(model.lbfgs, "B_matrix"):
            try:
                B = model.lbfgs.B_matrix()
                if B is not None:
                    eigs = np.linalg.eigvalsh(B)
                    c_hat = float(max(np.min(eigs), 1e-12))  # avoid zero
                    C_hat = float(np.max(eigs))
                    return c_hat, C_hat
            except Exception:
                pass
        
        # Fallback to conservative constants
        return 1.0, 1.0

    def finalize(self, gamma: float, model) -> Dict[str, Any]:
        """
        Compute final statistics and sample complexity after calibration.
        
        Estimates:
        - G_hat: Robust gradient norm upper bound (quantile of observed norms)
        - D_hat: Hypothesis diameter (max distance from initial parameters, clamped by D_cap)
        - c_hat, C_hat: L-BFGS curvature bounds
        - N_star: Sample complexity = ⌈(G·D·√(cC)/γ)²⌉
        
        Args:
            gamma: Target average regret per step
            model: Model with L-BFGS optimizer for curvature estimation
            
        Returns:
            Dictionary containing estimated constants and sample complexity
        """
        if not self.grad_norms or not self.thetas:
            raise ValueError("No observations collected. Call observe() during calibration.")
        
        # Estimate gradient norm upper bound using quantile for robustness
        G_hat = float(np.quantile(self.grad_norms, self.quantile))
        
        # Estimate hypothesis diameter with upper bound clamping
        theta_0 = self.thetas[0]
        distances = [np.linalg.norm(th - theta_0) for th in self.thetas]
        D_raw = float(max(distances)) if distances else 1.0
        D_hat = min(D_raw, self.D_cap)  # Clamp to prevent extreme values
        
        # Store for future use
        self.D = D_hat
        
        # Estimate curvature bounds with fallback to conservative defaults
        self.c_hat, self.C_hat = self.estimate_bounds(model)
        
        # Compute sample complexity
        numerator = G_hat * D_hat * np.sqrt(self.c_hat * self.C_hat)
        N_star_raw = (numerator / gamma) ** 2
        
        # Add reasonable bounds to prevent overflow
        N_star = int(np.ceil(min(N_star_raw, 1e6)))  # Cap at 1M
        
        # Store finalized G for drift detection
        self.finalized_G = G_hat
        
        print(f"[Calibrator] G_hat = {G_hat:.4f} (quantile {self.quantile})")
        print(f"[Calibrator] D_hat = {D_hat:.4f} (clamped by D_cap = {self.D_cap})")
        print(f"[Calibrator] c_hat = {self.c_hat:.4f}, C_hat = {self.C_hat:.4f}")
        
        return {
            "G": G_hat,
            "D": D_hat, 
            "c": self.c_hat,
            "C": self.C_hat,
            "N_star": max(1, N_star),
        }

    def check_drift(self, threshold: float = 0.3) -> bool:
        """
        Check if gradient norm EMA has drifted significantly from finalized value.
        
        Args:
            threshold: Relative threshold for drift detection (e.g., 0.3 = 30% increase)
            
        Returns:
            True if drift is detected, False otherwise
        """
        if self.finalized_G is None or self.G_ema is None:
            return False
        
        drift_ratio = (self.G_ema - self.finalized_G) / self.finalized_G
        return drift_ratio > threshold

    def get_updated_stats(self, model) -> Dict[str, Any]:
        """
        Get updated calibration statistics using current EMA values.
        
        Args:
            model: Model with L-BFGS optimizer for curvature estimation
            
        Returns:
            Dictionary with updated statistics for recalibration
        """
        if self.G_ema is None:
            raise ValueError("No EMA data available. Call observe_ongoing() first.")
        
        # Use current EMA as the updated G estimate
        G_updated = self.G_ema
        
        # Keep the same D (hypothesis diameter doesn't change much)
        D_updated = self.D if hasattr(self, 'D') else 1.0
        
        # Re-estimate curvature bounds
        c_updated, C_updated = self.estimate_bounds(model)
        
        print(f"[Calibrator] Updated stats: G = {G_updated:.4f}, D = {D_updated:.4f}")
        print(f"[Calibrator] Updated curvature: c = {c_updated:.4f}, C = {C_updated:.4f}")
        
        return {
            "G": G_updated,
            "D": D_updated,
            "c": c_updated,
            "C": C_updated,
        }

    def live_bounds_update(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update live bounds estimation with new observation.
        
        Uses EMA for G_hat and trimmed quantiles for robustness.
        Maintains rolling estimate of D_hat using parameter trajectory.
        
        Args:
            obs: Dictionary with observation data, may include:
                - 'grad_norm': float, current gradient norm
                - 'x_norm': float, current parameter norm  
                - 'loss': float, current loss value
                - 'theta': np.ndarray, current parameters
                
        Returns:
            Dictionary with current live bounds: {G_hat, D_hat, c_hat, C_hat}
        """
        # Update G_hat using EMA and trimmed quantiles
        if 'grad_norm' in obs:
            grad_norm = obs['grad_norm']
            
            # Update EMA
            if self.G_hat_live is None:
                self.G_hat_live = grad_norm
            else:
                self.G_hat_live = self.ema_beta * self.G_hat_live + (1 - self.ema_beta) * grad_norm
            
            # Maintain buffer for trimmed quantile
            self.grad_norm_buffer.append(grad_norm)
            if len(self.grad_norm_buffer) > self.buffer_size:
                self.grad_norm_buffer.pop(0)
                
            # Use trimmed quantile for robustness if we have enough samples
            if len(self.grad_norm_buffer) >= 10:
                trim_quantile = getattr(self, 'trim_quantile', 0.95)
                G_trimmed = float(np.quantile(self.grad_norm_buffer, trim_quantile))
                # Blend EMA and trimmed quantile
                self.G_hat_live = 0.7 * self.G_hat_live + 0.3 * G_trimmed
        
        # Update D_hat using parameter trajectory
        if 'theta' in obs:
            theta = obs['theta']
            
            # Initialize theta_0 on first observation
            if self.theta_0_live is None:
                self.theta_0_live = theta.copy()
                self.D_hat_live = 0.0
            else:
                # Compute distance from initial parameters
                current_distance = np.linalg.norm(theta - self.theta_0_live)
                
                # Update D_hat using EMA
                if self.D_hat_live is None:
                    self.D_hat_live = current_distance
                else:
                    self.D_hat_live = self.ema_beta * self.D_hat_live + (1 - self.ema_beta) * current_distance
                
                # Clamp by D_cap
                self.D_hat_live = min(self.D_hat_live, self.D_cap)
        
        # Return current snapshot
        return {
            'G_hat': self.G_hat_live or self.finalized_G or 1.0,
            'D_hat': self.D_hat_live or getattr(self, 'D', 1.0),
            'c_hat': self.c_hat or 1.0,
            'C_hat': self.C_hat or 1.0,
        }