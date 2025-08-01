def regret(pred: float, y: float) -> float:
    """Calculate the regret as the squared difference between prediction and true value."""
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return 0.5 * float((pred_scalar - y_scalar) ** 2)


def abs_error(pred: float, y: float) -> float:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return abs(pred_scalar - y_scalar)


def mape(pred: float, y: float, eps: float = 1e-8) -> float:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return abs(pred_scalar - y_scalar) / (abs(y_scalar) + eps)


def smape(pred: float, y: float, eps: float = 1e-8) -> float:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return 2 * abs(pred_scalar - y_scalar) / (abs(pred_scalar) + abs(y_scalar) + eps)


# If you still want a 0–1 “accuracy” notion:
def tol_accuracy(pred: float, y: float, tau: float = 0.1) -> int:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return int(abs(pred_scalar - y_scalar) <= tau)
