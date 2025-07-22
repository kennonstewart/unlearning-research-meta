import numpy as np


def regret(pred: float, y: float) -> float:
    """Squared loss regret for a single prediction."""
    return 0.5 * float((pred - y) ** 2)


def accuracy(pred: float, y: float) -> float:
    """Simple classification accuracy by rounding the prediction."""
    return float(int(np.rint(pred)) == int(y))
