class PrivacyOdometer:
    """Track privacy budget consumption for deletions."""

    def __init__(self, eps_total: float = 1.0, delta_total: float = 1e-5, max_deletions: int = 20):
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.max_deletions = max_deletions
        self.eps_spent = 0.0
        self.deletes = 0
        self.eps_step = eps_total / (2 * max_deletions)
        self.delta_step = delta_total / (2 * max_deletions)

    def consume(self) -> None:
        if self.deletes >= self.max_deletions:
            raise RuntimeError("max_deletions budget exceeded")
        self.deletes += 1
        self.eps_spent += self.eps_step

    def remaining(self) -> float:
        return self.eps_total - self.eps_spent

    def noise_scale(self, sensitivity: float) -> float:
        import numpy as np
        return sensitivity * (2 * np.log(1.25 / self.delta_step)) ** 0.5 / self.eps_step
