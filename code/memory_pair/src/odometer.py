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
        T: int,
        gamma: float,
        lambda_: float = 0.1,
        delta_B: float = 0.05,
    ):
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta_B = delta_B

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

        self.deletion_capacity = self._compute_capacity()
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
        """Solve for the largest m such that regret stays under γ."""

        def regret_bound(m):
            # Insertion regret
            insertion_regret = self.L * self.D * np.sqrt(self.T)

            # Deletion regret (from Theorem 5.5)
            deletion_regret = (m * self.L / self.lambda_) * np.sqrt(
                (2 * np.log(1.25 * m / self.delta_total) / self.eps_total)
                * (2 * np.log(1 / self.delta_B))
            )

            return (insertion_regret + deletion_regret) / self.T

        # Linear search until bound exceeded
        m = 1
        while regret_bound(m) <= self.gamma and m < self.T:
            m += 1
        return m - 1

    def spend(self):
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
