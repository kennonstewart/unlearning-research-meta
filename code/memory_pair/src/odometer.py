# /code/memory_pair/src/odometer.py
# Privacy odometer to track budget consumption for deletions.
# This module implements a simple odometer to manage privacy budgets for unlearning operations.
import numpy as np


class PrivacyOdometer:
    """Privacy accountant using principled bounds for (ε, δ)-unlearning."""

    def __init__(
        self,
        *,
        eps_total: float = 1.0,
        delta_total: float = 1e-5,
        deletion_capacity: int,
        L: float = 1.0,
        lambda_: float = 0.1,  # Strong convexity bound
    ):
        assert deletion_capacity > 0, "Deletion capacity must be positive."
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.m = deletion_capacity

        self.L = L
        self.lambda_ = lambda_

        self.eps_spent = 0.0
        self.deletions_count = 0

        self.eps_step = eps_total / self.m
        self.delta_step = delta_total / self.m

        self._sigma2_step = (L / lambda_) ** 2 * (
            2 * np.log(1.25 / self.delta_step) / self.eps_step**2
        )
        self._sigma_step = np.sqrt(self._sigma2_step)

    def spend(self):
        if self.deletions_count >= self.m:
            raise RuntimeError("Exceeded deletion capacity. Retraining required.")
        self.eps_spent += self.eps_step
        self.deletions_count += 1

    def remaining(self):
        return self.eps_total - self.eps_spent

    def noise_scale(self) -> float:
        """Standard deviation of Gaussian noise to inject."""
        return self._sigma_step