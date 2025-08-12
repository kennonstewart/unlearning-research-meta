from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    from ..odometer import ZCDPOdometer
except ImportError:
    from odometer import ZCDPOdometer


class Adapter:
    """Adapter for ZCDPOdometer to conform to Accountant protocol."""
    
    def __init__(self, **kwargs):
        """Initialize with ZCDPOdometer parameters."""
        self.odometer = ZCDPOdometer(**kwargs)
    
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
            sigma = self.odometer.sigma_step
            if sigma is None or np.isnan(sigma) or np.isinf(sigma):
                return False, None, "privacy_gate"
            
            # Check if we have enough rho budget for this deletion
            rho_cost = self.odometer.rho_cost_gaussian(sensitivity, sigma)
            if self.odometer.rho_spent + rho_cost > self.odometer.rho_total:
                return False, None, "privacy_gate"
            
            return True, sigma, None
        except Exception:
            return False, None, "privacy_gate"
    
    def spend(self, sensitivity: float, sigma: float) -> None:
        """Spend privacy budget."""
        self.odometer.spend(sensitivity, sigma)
    
    def metrics(self) -> Dict[str, Any]:
        """Get current privacy metrics."""
        return {
            "accountant": "zcdp",
            "m_capacity": self.odometer.deletion_capacity,
            "m_used": self.odometer.deletions_count,
            "rho_spent": self.odometer.rho_spent,
            "rho_remaining": self.odometer.rho_total - self.odometer.rho_spent,
            "delta_total": self.odometer.delta_total,
            "sigma_step": getattr(self.odometer, 'sigma_step', None),
        }