"""
Zero-concentrated differential privacy (zCDP) accountant adapter.

Wraps the existing ZCDPOdometer to conform to the unified Accountant interface,
providing zCDP accounting with joint m-sigma optimization.
"""
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
        # Filter kwargs for ZCDPOdometer
        odometer_kwargs = {
            'rho_total': kwargs.get('rho_total', 1.0),
            'delta_total': kwargs.get('delta_total', 1e-5),
            'T': kwargs.get('T', 10000),
            'gamma': kwargs.get('gamma', 0.5),
            'lambda_': kwargs.get('lambda_', 0.1),
            'delta_b': kwargs.get('delta_b', 0.05),
            'm_max': kwargs.get('m_max', None),
        }
        # Sanitize m_max
        if odometer_kwargs['m_max'] is None:
            odometer_kwargs['m_max'] = 10
        self.odometer = ZCDPOdometer(**odometer_kwargs)
    
    def finalize(self, stats: Dict[str, float], T_estimate: int) -> None:
        """Finalize using calibration statistics."""
        self.odometer.finalize_with(stats, T_estimate)
    
    def ready(self) -> bool:
        """Check if odometer is ready for deletions."""
        return self.odometer.ready_to_delete
    
    def pre_delete(self, sensitivity: float) -> Tuple[bool, Optional[float], Optional[str]]:
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
        sigma = getattr(self.odometer, 'sigma_step', None)
        return {
            "accountant": "zcdp",
            "m_capacity": int(self.odometer.deletion_capacity),
            "m_used": int(self.odometer.deletions_count),
            "rho_spent": float(self.odometer.rho_spent),
            "rho_remaining": float(self.odometer.rho_total - self.odometer.rho_spent),
            "delta_total": float(self.odometer.delta_total),
            "sigma_step": float(sigma) if sigma is not None else None,
            # Standard fields that don't apply to zCDP
            "eps_spent": None,
            "eps_remaining": None,
        }