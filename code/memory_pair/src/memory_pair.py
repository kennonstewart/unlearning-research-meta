import numpy as np
from .lbfgs import LBFGS
from .odometer import PrivacyOdometer
from .metrics import regret


class StreamNewtonMemoryPair:
    def __init__(self, dim: int, odometer: PrivacyOdometer = None):
        self.theta = np.zeros(dim)
        self.lbfgs = LBFGS(dim)
        self.odometer = odometer
        # --- NEW ATTRIBUTES ---
        self.cumulative_regret = 0.0
        self.events_seen = 0

    def insert(self, x: np.ndarray, y: float) -> float:
        """
        Inserts a point and updates internal regret.
        Returns the prediction made *before* the update.
        """
        # 1. Make prediction
        pred = float(self.theta @ x)

        # 2. Update internal regret state
        self.cumulative_regret += regret(pred, y)
        self.events_seen += 1

        # 3. Perform the L-BFGS update
        g_old = (pred - y) * x
        direction = self.lbfgs.direction(g_old)
        # Assuming a simple learning rate of 1.0 for this example
        alpha = 1.0
        s = alpha * direction
        theta_new = self.theta + s

        g_new = (float(theta_new @ x) - y) * x
        y_vec = g_new - g_old

        self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

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
