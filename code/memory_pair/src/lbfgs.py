# /code/memory_pair/src/lbfgs.py
# Limited Memory BFGS implementation for curvature pair storage and direction computation.
import numpy as np


class LimitedMemoryBFGS:
    """Simple L-BFGS helper storing curvature pairs."""

    def __init__(self, m_max: int = 10):
        self.m_max = m_max
        self.S = []  # list of s vectors
        self.Y = []  # list of y vectors

    def add_pair(self, s: np.ndarray, y: np.ndarray) -> None:
        # Ensure yᵀs > 0 to keep B⁻¹ positive‑definite (oLBFGS requirement)
        ys = float(y @ s)
        if ys <= 1e-10:
            return  # discard pair
        self.S.append(s.astype(float))
        self.Y.append(y.astype(float))
        if len(self.S) > self.m_max:
            self.S.pop(0)
            self.Y.pop(0)

    def remove_pair(self, index: int) -> None:
        if 0 <= index < len(self.S):
            self.S.pop(index)
            self.Y.pop(index)

    def direction(self, grad: np.ndarray) -> np.ndarray:
        """Return approximate -H^{-1} grad using stored pairs."""
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
        # ----- initial H⁻¹ scaling per Mokhtari & Ribeiro (eq. 19) -----
        y_last = self.Y[-1]
        s_last = self.S[-1]
        gamma = float(s_last @ y_last) / float(y_last @ y_last)
        r = gamma * q
        for s, y, a, r_i in zip(self.S, self.Y, reversed(alpha), reversed(rho)):
            b = r_i * (y @ r)
            r = r + s * (a - b)
        return -r
