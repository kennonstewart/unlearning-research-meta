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
        odometer: PrivacyOdometer | None = None,
    ) -> None:
        self.dim = dim
        self.lam = lam
        self.theta = np.zeros(dim)
        self.lbfgs = LimitedMemoryBFGS(m_max=10)
        self.odometer = odometer or PrivacyOdometer()

    # -------- utilities ---------
    def _grad_point(self, x: np.ndarray, y: float) -> np.ndarray:
        return (self.theta @ x - y) * x

    # -------- learning ---------
    def step(self, x: np.ndarray, y: float) -> float:
        g_old = self._grad_point(x, y)
        d = self.lbfgs.direction(g_old)
        lr = 0.5
        theta_new = self.theta + lr * d
        s = theta_new - self.theta
        self.theta = theta_new
        g_new = self._grad_point(x, y)
        y_vec = g_new - g_old
        self.lbfgs.add_pair(s, y_vec)
        loss = 0.5 * (self.theta @ x - y) ** 2
        logger.debug("step", extra={"loss": float(loss)})
        return loss

    def insert(self, x: np.ndarray, y: float) -> float:
        return self.step(x, y)

    # -------- deletion ---------
    def delete(self, x: np.ndarray, y: float) -> None:
        if not self.lbfgs.S:
            raise RuntimeError("No curvature pairs to use for unlearning")
        g = self._grad_point(x, y)
        d = self.lbfgs.direction(g)
        self.theta -= d
        self.odometer.consume()
        sigma = self.odometer.noise_scale(np.linalg.norm(d))
        self.theta += np.random.normal(0.0, sigma, size=self.dim)
        logger.debug("delete", extra={"remaining_eps": self.odometer.remaining()})

MemoryPair = StreamNewtonMemoryPair
