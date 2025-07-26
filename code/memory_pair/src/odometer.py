# /code/memory_pair/src/odometer.py
# Privacy odometer to track budget consumption for deletions.
# This module implements a simple odometer to manage privacy budgets for unlearning operations.
import numpy as np


class PrivacyOdometer:
    """Track privacy budget consumption for deletions."""

    def __init__(
        self,
        *,
        eps_total: float | None = 1.0,
        delta_total: float = 1e-5,
        max_deletions: int | None = None,
    ):
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.max_deletions = max_deletions
        self.eps_spent = 0.0
        self.deletions_count = 0
        # ε consumed at each deletion
        denom = 2 * (max_deletions if max_deletions else 1)
        self.eps_step = eps_total / denom

        self.delta_step = delta_total / (2 * (max_deletions if max_deletions else 1))
        if eps_total is not None and max_deletions is not None:
            if eps_per_delete is not None:
                raise ValueError(
                    "Specify either eps_per_delete or max_deletions, not both."
                )
            if eps_total < 0 or max_deletions <= 0:
                raise ValueError(
                    "eps_total must be non-negative and max_deletions must be positive."
                )

    def spend(self):
        if self.deletions_count >= self.max_deletions:
            raise RuntimeError(
                f"Deletion capacity of {self.max_deletions} has been exceeded. "
                "Model guarantees are void."
            )
        self.eps_spent += self.eps_per_delete
        self.deletions_count += 1

    def remaining(self) -> float:
        return self.eps_total - self.eps_spent

    def noise_scale(self, sensitivity: float) -> float:
        return sensitivity * (2 * np.log(1.25 / self.delta_step)) ** 0.5 / self.eps_step
