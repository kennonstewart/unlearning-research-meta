import numpy as np
from .lbfgs import LimitedMemoryBFGS
from .odometer import PrivacyOdometer
from .metrics import regret


class MemoryPair:
    def __init__(self, dim: int, odometer: PrivacyOdometer = PrivacyOdometer()):
        self.theta = np.zeros(dim)
        self.lbfgs = LimitedMemoryBFGS(dim)
        self.odometer = odometer
        # --- NEW ATTRIBUTES ---
        self.cumulative_regret = 0.0
        self.events_seen = 0

    def insert(
        self,
        x: np.ndarray,
        y: float,
        *,
        return_grad: bool = False,
        log_to_odometer: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """
        Inserts a point, updates internal regret and L-BFGS state.
        Returns the prediction made *before* the update.
        If return_grad=True, also returns the pre-update gradient g_old.
        """

        # 1. Prediction before update
        pred = float(self.theta @ x)

        # 2. Update regret counters
        self.cumulative_regret += regret(pred, y)
        self.events_seen += 1

        # 3. Compute gradients and L-BFGS direction
        g_old = (pred - y) * x  # <-- gradient you want
        direction = self.lbfgs.direction(g_old)
        alpha = 1.0
        s = alpha * direction
        theta_new = self.theta + s

        g_new = (float(theta_new @ x) - y) * x
        y_vec = g_new - g_old

        self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

        # Expose for outside consumers
        self.last_grad = g_old

        # Optional: push stats to odometer during bootstrap/warmup
        if log_to_odometer and hasattr(self, "odometer"):
            self.odometer.observe(g_old, self.theta)

        if return_grad:
            return pred, g_old
        return pred

    def delete(self, x: np.ndarray, y: float) -> None:
        if self.odometer:
            self.odometer.spend()

        g = (float(self.theta @ x) - y) * x
        influence = self.lbfgs.direction(g)

        sensitivity = np.linalg.norm(influence)
        sigma = self.odometer.noise_scale(sensitivity)
        noise = np.random.normal(0, sigma, self.theta.shape)

        self.theta = self.theta - influence + noise

    def get_average_regret(self) -> float:
        """Calculates the average regret over all seen events."""
        if self.events_seen == 0:
            return float("inf")
        return self.cumulative_regret / self.events_seen
