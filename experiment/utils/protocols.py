"""
Protocol interfaces for unifying accountant and model APIs.
Eliminates isinstance checks and provides type safety.
"""

from typing import Protocol, Dict, Any, Tuple, Optional
import numpy as np


class Accountant(Protocol):
    """Protocol for privacy accountants (RDP and legacy)."""
    
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """Finalize accountant with calibration statistics."""
        ...
    
    def spend(self, sensitivity: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """Spend privacy budget for one operation."""
        ...
    
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get noise scale for current operation."""
        ...
    
    def over_budget(self) -> bool:
        """Check if privacy budget is exhausted."""
        ...
    
    def metrics(self) -> Dict[str, Any]:
        """Get current privacy metrics for logging."""
        ...


class OnlineUnlearner(Protocol):
    """Protocol for online unlearning algorithms."""
    
    def calibrate_step(self, x: np.ndarray, y: Any) -> float:
        """Perform one calibration step, return prediction."""
        ...
    
    def insert(self, x: np.ndarray, y: Any, return_grad: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """Insert new data point, optionally return gradient."""
        ...
    
    def delete(self, x: np.ndarray, y: Any) -> None:
        """Delete data point."""
        ...
    
    @property
    def last_grad(self) -> Optional[np.ndarray]:
        """Get gradient from last operation."""
        ...
    
    @property
    def cumulative_regret(self) -> float:
        """Get current cumulative regret."""
        ...


# Adapter classes to wrap existing implementations

class AccountantAdapter:
    """Adapter to make existing accountants conform to Protocol."""
    
    def __init__(self, accountant):
        self._accountant = accountant
    
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        return self._accountant.finalize_with(stats, T_estimate)
    
    def spend(self, sensitivity: Optional[float] = None, sigma: Optional[float] = None) -> None:
        return self._accountant.spend(sensitivity, sigma)
    
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        return self._accountant.noise_scale(sensitivity)
    
    def over_budget(self) -> bool:
        return self._accountant.over_budget()
    
    def metrics(self) -> Dict[str, Any]:
        """Extract metrics from accountant."""
        from metrics_utils import get_privacy_metrics
        
        # Create a mock model object with this accountant
        class MockModel:
            def __init__(self, odometer):
                self.odometer = odometer
        
        mock = MockModel(self._accountant)
        return get_privacy_metrics(mock)
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped accountant."""
        return getattr(self._accountant, name)


class ModelAdapter:
    """Adapter to make existing models conform to Protocol."""
    
    def __init__(self, model):
        self._model = model
    
    def calibrate_step(self, x: np.ndarray, y: Any) -> float:
        return self._model.calibrate_step(x, y)
    
    def insert(self, x: np.ndarray, y: Any, return_grad: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        try:
            return self._model.insert(x, y, return_grad=return_grad)
        except TypeError:
            # Fallback for models that don't support return_grad
            pred = self._model.insert(x, y)
            grad = self.last_grad if return_grad else None
            return pred, grad
    
    def delete(self, x: np.ndarray, y: Any) -> None:
        return self._model.delete(x, y)
    
    @property
    def last_grad(self) -> Optional[np.ndarray]:
        return getattr(self._model, "last_grad", None)
    
    @property
    def cumulative_regret(self) -> float:
        return getattr(self._model, "cumulative_regret", 0.0)
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped model."""
        return getattr(self._model, name)