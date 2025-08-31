# /code/memory_pair/src/lbfgs.py
# Limited Memory BFGS implementation for curvature pair storage and direction computation.
import numpy as np


ABS_CURV_EPS = 1e-10
REL_CURV_EPS = 1e-8


class LimitedMemoryBFGS:
    """Simple L-BFGS helper storing curvature pairs with strong convexity controls."""

    def __init__(self, m_max: int = 10, cfg=None):
        self.m_max = m_max if m_max is not None else 10
        self.S = []  # list of s vectors
        self.Y = []  # list of y vectors
        self.cfg = cfg  # store config for strong convexity parameters

    def add_pair(self, s: np.ndarray, y: np.ndarray) -> tuple[bool, bool]:
        """
        Add curvature pair (s, y) with admission gating and optional damping.
        
        Returns:
            (admitted, damped) - flags indicating if pair was admitted and if damping was applied
        """
        ys = float(y @ s)
        ss = float(s @ s)
        
        # Get admission threshold
        m_t = getattr(self.cfg, "pair_admission_m", 1e-6) if self.cfg else 1e-6
        
        admitted = True
        damped = False
        
        # Check admission condition: s^T y >= m_t * s^T s
        if ys < m_t * ss:
            # Option B: Apply damping
            if ss > 1e-12:
                delta = max(0, m_t * ss - ys) / ss
                y = y + delta * s
                ys = float(y @ s)
                damped = True
            
            # If still not positive after damping, skip the pair
            if ys <= max(ABS_CURV_EPS, REL_CURV_EPS * ss):
                return False, damped
        
        # Final curvature condition check (original requirement)
        if ys <= max(ABS_CURV_EPS, REL_CURV_EPS * ss):
            return False, False
            
        self.S.append(s.astype(float))
        self.Y.append(y.astype(float))
        if len(self.S) > self.m_max:
            self.S.pop(0)
            self.Y.pop(0)
            
        return admitted, damped

    def remove_pair(self, index: int) -> None:
        if 0 <= index < len(self.S):
            self.S.pop(index)
            self.Y.pop(index)

    def direction(self, grad: np.ndarray, calibrator=None) -> np.ndarray:
        """Return approximate -H^{-1} grad using stored pairs with spectrum clamping."""
        if not self.S:
            return -grad
        q = grad.copy()
        alpha = []
        rho = []
        for s, y in reversed(list(zip(self.S, self.Y))):
            r = 1.0 / float(y @ s)
            rho.append(r)
            a = r * (s @ q)
            alpha.append(a)
            q = q - a * y
        
        # Initial H⁻¹ scaling with spectrum clamping
        y_last = self.Y[-1]
        s_last = self.S[-1]
        gamma = float(s_last @ y_last) / float(y_last @ y_last)
        
        # Apply spectrum clamping: clamp gamma to reasonable bounds
        if calibrator:
            c_hat = getattr(calibrator, 'c_hat', 1.0) or 1.0
            C_hat = getattr(calibrator, 'C_hat', 1.0) or 1.0
            
            # More conservative clamping - ensure gamma stays reasonable
            eps = getattr(self.cfg, "hessian_clamp_eps", 1e-12) if self.cfg else 1e-12
            gamma_min = max(eps, 1e-6)  # Conservative lower bound 
            gamma_max = min(1e6, max(1.0, C_hat / c_hat))  # Conservative upper bound
            gamma = float(np.clip(gamma, gamma_min, gamma_max))
        else:
            # Fallback clamping - very conservative
            gamma = float(np.clip(gamma, 1e-6, 1e6))
        
        r = gamma * q
        for s, y, a, r_i in zip(self.S, self.Y, reversed(alpha), reversed(rho)):
            b = r_i * (y @ r)
            r = r + s * (a - b)
        
        # Apply final bounds to direction to prevent extreme values
        direction = -r
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e3:  # Reasonable upper bound for direction
            direction = direction * (1e3 / direction_norm)
            
        return direction

    def last_direction_norm(self, grad: np.ndarray) -> float:
        """Convenience accessor for ||direction(grad)||."""
        d = self.direction(grad)
        return float(np.linalg.norm(d))
