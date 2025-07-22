import numpy as np

class LimitedMemoryBFGS:
    """Simple L-BFGS helper storing curvature pairs."""

    def __init__(self, m_max: int = 10):
        self.m_max = m_max
        self.S = []  # list of s vectors
        self.Y = []  # list of y vectors

    def add_pair(self, s: np.ndarray, y: np.ndarray) -> None:
        self.S.append(s.astype(float))
        self.Y.append(y.astype(float))
        if len(self.S) > self.m_max:
            self.S.pop(0)
            self.Y.pop(0)

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
        r = q
        for s, y, a, r_i in zip(self.S, self.Y, reversed(alpha), reversed(rho)):
            b = r_i * (y @ r)
            r = r + s * (a - b)
        return -r
