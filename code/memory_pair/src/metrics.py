"""Utility metrics for loss and regret calculations."""

from typing import Any


def _to_scalar(val: Any) -> float:
    """Convert tensors/arrays to plain Python floats."""
    return float(val.item() if hasattr(val, "item") else val)


def loss_half_mse(pred: Any, y: Any) -> float:
    """Half mean squared error loss.

    Converts inputs to scalars to remain agnostic to the array/tensor type.
    """
    pred_scalar = _to_scalar(pred)
    y_scalar = _to_scalar(y)
    return 0.5 * (pred_scalar - y_scalar) ** 2


def regret(prediction: Any, target: Any) -> float:
    """Alias for :func:`loss_half_mse` for backward compatibility."""
    return loss_half_mse(prediction, target)

