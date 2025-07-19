# baselines.py
import numpy as np
from abc import ABC, abstractmethod


class BaseOnlineLearner(ABC):
    """
    Common interface for all online learners used in run.py
    -------------------------------------------------------
    Public API
    ----------
    insert(x: np.ndarray, y: float) -> float
        • Updates the model with a single (x, y) pair.
        • Returns the example’s squared‑error loss (optional for callers).

    Attributes
    ----------
    theta : np.ndarray
        Current parameter vector (read by run.py when computing regret).
    """

    def __init__(self, dim: int):
        self.theta = np.zeros(dim)

    # `insert` is the method the simulation script calls
    def insert(self, x: np.ndarray, y: float) -> float:
        return self.step(x, y)

    # subclasses implement their own update rule here
    @abstractmethod
    def step(self, x: np.ndarray, y: float) -> float: 
        ...


class OnlineSGD(BaseOnlineLearner):
    def __init__(self, dim: int, lr: float = 0.1):
        super().__init__(dim)
        self.lr = lr

    def step(self, x: np.ndarray, y: float) -> float:
        pred = self.theta @ x
        grad = (pred - y) * x
        self.theta -= self.lr * grad
        return 0.5 * (pred - y) ** 2


class AdaGrad(BaseOnlineLearner):
    def __init__(self, dim: int, lr: float = 1.0, eps: float = 1e-8):
        super().__init__(dim)
        self.lr = lr
        self.eps = eps
        self.G = np.zeros(dim)

    def step(self, x: np.ndarray, y: float) -> float:
        pred = self.theta @ x
        grad = (pred - y) * x
        self.G += grad ** 2
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.eps)
        self.theta -= adjusted_lr * grad
        return 0.5 * (pred - y) ** 2


class OnlineNewtonStep(BaseOnlineLearner):
    """
    “Poor‑man’s” Online Newton Step (O‑N‑S) with a full Hessian approximation.
    For high‑dimensional streams you’d normally use a diagonal or sketch, but
    this keeps the maths simple for the baseline.
    """

    def __init__(self, dim: int, lam: float = 1.0):
        super().__init__(dim)
        self.H = lam * np.eye(dim)  # running Hessian estimate

    def step(self, x: np.ndarray, y: float) -> float:
        pred = self.theta @ x
        grad = (pred - y) * x
        self.H += np.outer(x, x)
        # Solve H⁻¹⋅grad without forming the inverse every time
        direction = np.linalg.solve(self.H, grad)
        self.theta -= direction
        return 0.5 * (pred - y) ** 2
