# code/memory_pair/src/memory_pair.py
# Stream-based Newton method with L-BFGS for online learning and deletion.
# This module implements a memory pair model that supports efficient unlearning.
import logging
import numpy as np

from .lbfgs import LimitedMemoryBFGS
from .odometer import PrivacyOdometer

logger = logging.getLogger(__name__)


class StreamNewtonMemoryPair:
    """Online learner with support for deletion using L-BFGS."""

    def __init__(
        self,
        dim: int,
        lam: float = 1.0,
        lr0: float = 0.5,
        t0: int = 10,
        odometer: PrivacyOdometer | None = None,
    ) -> None:
        self.dim = dim
        self.lam = lam
        self.lr0 = lr0
        self.t0 = t0
        self.t = 0  # iteration counter
        self.theta = np.zeros(dim)
        self.lbfgs = LimitedMemoryBFGS(m_max=10)
        self.odometer = odometer or PrivacyOdometer()

    # -------- utilities ---------
    def _grad_point(self, x: np.ndarray, y: float) -> np.ndarray:
        return (self.theta @ x - y) * x

    def insert(self, x: np.ndarray, y: float) -> float:
        self.t += 1
        g_old = self._grad_point(x, y)
        d = self.lbfgs.direction(g_old)
        lr = self.lr0 * self.t0 / (self.t0 + self.t)  # decaying schedule
        theta_new = self.theta + lr * d
        s = theta_new - self.theta
        self.theta = theta_new
        g_new = self._grad_point(x, y)
        y_vec = g_new - g_old
        self.lbfgs.add_pair(s, y_vec)
        loss = 0.5 * (self.theta @ x - y) ** 2
        logger.debug("step", extra={"loss": float(loss)})
        return loss

    # -------- deletion ---------
    def delete(self, x: np.ndarray, y: float) -> None:
        if not self.lbfgs.S:
            raise RuntimeError("No curvature pairs to use for unlearning")
        g = self._grad_point(x, y)
        clip = 1.0  # or tune
        d = self.lbfgs.direction(g)
        norm_d = np.linalg.norm(d)
        if norm_d > clip:
            d = d * (clip / norm_d)
        self.theta -= d
        self.odometer.consume()
        self.lbfgs.remove_pair(0)
        sigma = self.odometer.noise_scale(np.linalg.norm(d))
        self.theta += np.random.normal(0.0, sigma, size=self.dim)
        logger.debug("delete", extra={"remaining_eps": self.odometer.remaining()})


MemoryPair = StreamNewtonMemoryPair
