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
    
    After collection, it computes the sample complexity N* = ⌈(G·D·√(cC)/γ)²⌉
    
    Attributes:
        grad_norms (List[float]): Collected gradient norms during calibration
        thetas (List[np.ndarray]): Parameter snapshots during calibration  
        quantile (float): Quantile for robust G estimation (default 0.95)
        D_cap (float): Upper bound for hypothesis diameter (default 10.0)
        c_hat (Optional[float]): Estimated lower curvature bound
        C_hat (Optional[float]): Estimated upper curvature bound
    """
    
    def __init__(self, quantile: float = 0.95, D_cap: float = 10.0):
        """
        Initialize the Calibrator.
        
        Args:
            quantile: Quantile for gradient norm estimation (e.g., 0.95 to clip outliers)
            D_cap: Upper bound for hypothesis diameter to prevent extreme values
        """
        self.grad_norms = []
        self.thetas = []
        self.quantile = quantile
        self.D_cap = D_cap
        self.c_hat: Optional[float] = None
        self.C_hat: Optional[float] = None

    def observe(self, grad: np.ndarray, theta: np.ndarray) -> None:
        """
        Record gradient and parameter snapshot during calibration.
        
        Args:
            grad: Gradient vector ∇ℓ_t(w_t)
            theta: Current parameter vector w_t
        """
        self.grad_norms.append(np.linalg.norm(grad))
        self.thetas.append(theta.copy())

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
        
        # Estimate curvature bounds with fallback to conservative defaults
        self.c_hat, self.C_hat = self.estimate_bounds(model)
        
        # Compute sample complexity
        numerator = G_hat * D_hat * np.sqrt(self.c_hat * self.C_hat)
        N_star_raw = (numerator / gamma) ** 2
        
        # Add reasonable bounds to prevent overflow
        N_star = int(np.ceil(min(N_star_raw, 1e6)))  # Cap at 1M
        
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