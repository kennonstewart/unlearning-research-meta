# /code/memory_pair/src/odometer.py
# Privacy odometer to track budget consumption for deletions.
# This module implements a simple odometer to manage privacy budgets for unlearning operations.
import numpy as np


class PrivacyOdometer:
    """Adaptive Privacy Odometer that estimates deletion capacity from warmup dynamics."""

    def __init__(
        self,
        *,
        eps_total: float = 1.0,
        delta_total: float = 1e-5,
        T: int = 10000,
        gamma: float = 0.5,
        lambda_: float = 0.1,
        delta_b: float = 0.05,
    ):
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta_b = delta_b

        # Dynamic warmup statistics
        self._grad_norms = []
        self._theta_traj = []

        # Filled after warmup
        self.deletion_capacity = None
        self.L = None
        self.D = None
        self.eps_step = None
        self.delta_step = None
        self.sigma_step = None

        self.eps_spent = 0.0
        self.deletions_count = 0

    def observe(self, grad: np.ndarray, theta: np.ndarray):
        """Call during warmup to track model statistics."""
        self._grad_norms.append(np.linalg.norm(grad))
        self._theta_traj.append(np.copy(theta))

    def finalize(self):
        """Run after warmup to compute capacity, noise scale, and accounting."""
        self.L = max(self._grad_norms)
        theta_0 = self._theta_traj[0]
        self.D = max(np.linalg.norm(theta - theta_0) for theta in self._theta_traj)

        m = self._compute_capacity()
        self.deletion_capacity = max(1, m)  # <- guard
        self.eps_step = self.eps_total / self.deletion_capacity
        self.delta_step = self.delta_total / self.deletion_capacity

        self.sigma_step = (
            (self.L / self.lambda_)
            * np.sqrt(2 * np.log(1.25 / self.delta_step))
            / self.eps_step
        )

        print(
            f"[Odometer] Finalized with deletion capacity m = {self.deletion_capacity}"
        )
        print(f"[Odometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(
            f"[Odometer] ε_step = {self.eps_step:.6f}, δ_step = {self.delta_step:.2e}, σ = {self.sigma_step:.4f}"
        )

    def _compute_capacity(self) -> int:
        def regret_bound(m):
            insertion_regret = self.L * self.D * np.sqrt(self.T)
            deletion_regret = (m * self.L / self.lambda_) * np.sqrt(
                (2 * np.log(1.25 * max(m, 1) / self.delta_total) / self.eps_total)
                * (2 * np.log(1 / self.delta_b))
            )
            return (insertion_regret + deletion_regret) / self.T

        # If even m=1 fails, bail out early
        if regret_bound(1) > self.gamma:
            return 0

        lo, hi = 1, self.T
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if regret_bound(mid) <= self.gamma:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def spend(self):
        if self.deletion_capacity is None:
            raise RuntimeError(
                "Deletion capacity is not set. Call `finalize()` before spending."
            )
        if self.deletions_count >= self.deletion_capacity:
            raise RuntimeError(
                f"Deletion capacity {self.deletion_capacity} exceeded. Retraining required."
            )
        self.eps_spent += self.eps_step
        self.deletions_count += 1

    def remaining(self) -> float:
        return self.eps_total - self.eps_spent

    def noise_scale(self) -> float:
        """Standard deviation of Gaussian noise to inject."""
        if self.sigma_step is None:
            raise ValueError("Call `finalize()` after warmup to compute noise scale.")
        return self.sigma_step
