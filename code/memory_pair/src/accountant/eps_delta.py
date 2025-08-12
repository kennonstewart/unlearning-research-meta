"""
Epsilon-delta differential privacy accountant adapter.

Wraps the existing PrivacyOdometer to conform to the unified Accountant interface,
providing eps-delta DP accounting with fixed privacy budget allocation.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import logging

try:
    from ..odometer import PrivacyOdometer
except ImportError:
    from odometer import PrivacyOdometer


class Adapter:
    """Adapter for PrivacyOdometer to conform to Accountant protocol."""
    
    def __init__(self, **kwargs):
        """Initialize with PrivacyOdometer parameters."""
        # Filter kwargs for PrivacyOdometer
        odometer_kwargs = {
            'eps_total': kwargs.get('eps_total', 1.0),
            'delta_total': kwargs.get('delta_total', 1e-5),
            'T': kwargs.get('T', 10000),
            'gamma': kwargs.get('gamma', 0.5),
            'lambda_': kwargs.get('lambda_', 0.1),
            'delta_b': kwargs.get('delta_b', 0.05),
        }
        self.odometer = PrivacyOdometer(**odometer_kwargs)
    
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
            # Note: PrivacyOdometer.noise_scale ignores sensitivity parameter
            sigma = self.odometer.noise_scale(sensitivity)
            if sigma is None or np.isnan(sigma) or np.isinf(sigma):
                return False, None, "privacy_gate"
            return True, sigma, None
        except Exception as e:
            logging.debug(f"Exception in pre_delete: {e}")
            return False, None, "privacy_gate"
    
    def spend(self, sensitivity: float, sigma: float) -> None:
        """Spend privacy budget.
        
        Note: sensitivity and sigma parameters are ignored for eps-delta accounting.
        The PrivacyOdometer tracks budget independently of per-deletion parameters.
        """
        self.odometer.spend()
    
    def metrics(self) -> Dict[str, Any]:
        """Get current privacy metrics."""
        sigma = getattr(self.odometer, 'sigma_step', None)
        return {
            "accountant": "eps_delta",
            "m_capacity": int(self.odometer.deletion_capacity),
            "m_used": int(self.odometer.deletions_count),
            "eps_spent": float(self.odometer.eps_spent),
            "eps_remaining": float(self.odometer.eps_total - self.odometer.eps_spent),
            "delta_total": float(self.odometer.delta_total),
            "sigma_step": float(sigma) if sigma is not None else None,
            # Standard fields that don't apply to eps-delta
            "rho_spent": None,
            "rho_remaining": None,
        }