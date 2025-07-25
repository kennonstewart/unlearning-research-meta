def regret(pred: float, y: float) -> float:
    """Calculate the regret as the squared difference between prediction and true value."""
    return 0.5 * float((pred - y) ** 2)


def abs_error(pred: float, y: float) -> float:
    return abs(pred - y)


def mape(pred: float, y: float, eps: float = 1e-8) -> float:
    return abs(pred - y) / (abs(y) + eps)


def smape(pred: float, y: float, eps: float = 1e-8) -> float:
    return 2 * abs(pred - y) / (abs(pred) + abs(y) + eps)


# If you still want a 0–1 “accuracy” notion:
def tol_accuracy(pred: float, y: float, tau: float = 0.1) -> int:
    return int(abs(pred - y) <= tau)
