from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    from ..odometer import PrivacyOdometer
except ImportError:
    from odometer import PrivacyOdometer


class Adapter:
    """Adapter for PrivacyOdometer to conform to Accountant protocol."""
    
    def __init__(self, **kwargs):
        """Initialize with PrivacyOdometer parameters."""
        self.odometer = PrivacyOdometer(**kwargs)
    
    def finalize(self, stats: Dict[str, float], T_estimate: int) -> None:
        """Finalize using calibration statistics."""
        self.odometer.finalize_with(stats, T_estimate)
    
    def ready(self) -> bool:
        """Check if odometer is ready for deletions."""
        return self.odometer.ready_to_delete
    
    def pre_delete(self, sensitivity: float) -> Tuple[bool, float | None, str | None]:
        """Check if deletion is allowed and return noise scale."""
        if not self.ready():
            return False, None, "privacy_gate"
        
        if self.odometer.deletions_count >= self.odometer.deletion_capacity:
            return False, None, "privacy_gate"
        
        try:
            sigma = self.odometer.noise_scale(sensitivity)
            if sigma is None or np.isnan(sigma) or np.isinf(sigma):
                return False, None, "privacy_gate"
            return True, sigma, None
        except Exception:
            return False, None, "privacy_gate"
    
    def spend(self, sensitivity: float, sigma: float) -> None:
        """Spend privacy budget."""
        self.odometer.spend()
    
    def metrics(self) -> Dict[str, Any]:
        """Get current privacy metrics."""
        return {
            "accountant": "eps_delta",
            "m_capacity": self.odometer.deletion_capacity,
            "m_used": self.odometer.deletions_count,
            "eps_spent": self.odometer.eps_spent,
            "eps_remaining": self.odometer.eps_total - self.odometer.eps_spent,
            "delta_total": self.odometer.delta_total,
            "sigma_step": getattr(self.odometer, 'sigma_step', None),
        }