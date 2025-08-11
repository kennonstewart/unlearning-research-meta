#!/usr/bin/env python3
"""
Privacy accounting strategies for adaptive capacity odometer.

This module implements the strategy pattern for privacy accounting, allowing
runtime selection between different privacy accounting methods:
- zCDP (zero-Concentrated Differential Privacy)
- (ε,δ)-DP (epsilon-delta Differential Privacy)
- RelaxedAccountant (placeholder for experimental methods)

Each accountant provides:
1. Budget tracking and spending
2. Noise scale computation for given sensitivity
3. Capacity estimation given regret constraints
4. Dynamic recalibration support
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import warnings


class PrivacyAccountant(ABC):
    """
    Abstract base class for privacy accounting strategies.
    
    Defines the interface that all privacy accountants must implement
    for use with the adaptive capacity odometer.
    """
    
    @abstractmethod
    def __init__(self, budget_params: Dict[str, float], **kwargs):
        """
        Initialize accountant with budget parameters.
        
        Args:
            budget_params: Dictionary with budget parameters (eps_total, delta_total, rho_total, etc.)
            **kwargs: Additional accountant-specific parameters
        """
        pass
    
    @abstractmethod
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """
        Finalize accountant using calibration statistics.
        
        Args:
            stats: Calibration statistics (G, D, c, C)
            T_estimate: Estimated total number of events
        """
        pass
    
    @abstractmethod
    def spend(self, sensitivity: float, **kwargs) -> None:
        """
        Consume privacy budget for one deletion operation.
        
        Args:
            sensitivity: L2 sensitivity of the deletion operation
            **kwargs: Additional parameters (sigma, etc.)
        """
        pass
    
    @abstractmethod
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """
        Get noise scale for given sensitivity.
        
        Args:
            sensitivity: L2 sensitivity (may be unused depending on strategy)
            
        Returns:
            Standard deviation for Gaussian noise
        """
        pass
    
    @abstractmethod
    def remaining_capacity(self) -> int:
        """
        Get remaining deletion capacity.
        
        Returns:
            Number of deletions remaining before budget exhaustion
        """
        pass
    
    @abstractmethod
    def get_accountant_type(self) -> str:
        """
        Get string identifier for this accountant type.
        
        Returns:
            Accountant type string for logging/plotting
        """
        pass
    
    @abstractmethod
    def get_budget_summary(self) -> Dict[str, float]:
        """
        Get current budget usage summary.
        
        Returns:
            Dictionary with budget usage metrics
        """
        pass
    
    def supports_recalibration(self) -> bool:
        """Check if this accountant supports mid-stream recalibration."""
        return False
    
    def recalibrate_with(self, new_stats: Dict[str, Any], remaining_T: int) -> None:
        """
        Recalibrate accountant with updated statistics.
        
        Args:
            new_stats: Updated calibration statistics
            remaining_T: Remaining events to process
        """
        if not self.supports_recalibration():
            raise NotImplementedError(f"{self.get_accountant_type()} does not support recalibration")


class ZCDPAccountant(PrivacyAccountant):
    """
    zCDP-based privacy accountant.
    
    Uses zero-Concentrated Differential Privacy accounting with linear
    composition. Wraps the existing ZCDPOdometer implementation.
    """
    
    def __init__(self, budget_params: Dict[str, float], **kwargs):
        """
        Initialize zCDP accountant.
        
        Args:
            budget_params: Must contain 'rho_total' and 'delta_total'
            **kwargs: Additional parameters for ZCDPOdometer
        """
        # Import here to avoid circular imports
        from .odometer import ZCDPOdometer
        
        self.rho_total = budget_params['rho_total']
        self.delta_total = budget_params['delta_total']
        
        # Create underlying ZCDPOdometer
        self._odometer = ZCDPOdometer(
            rho_total=self.rho_total,
            delta_total=self.delta_total,
            **kwargs
        )
        
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """Finalize using ZCDPOdometer's joint optimization."""
        self._odometer.finalize_with(stats, T_estimate)
        
    def spend(self, sensitivity: float, **kwargs) -> None:
        """Spend budget using actual sensitivity and fixed noise scale."""
        sigma = kwargs.get('sigma', self._odometer.noise_scale())
        self._odometer.spend(sensitivity, sigma)
        
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get noise scale from underlying odometer."""
        return self._odometer.noise_scale(sensitivity)
        
    def remaining_capacity(self) -> int:
        """Get remaining deletion capacity."""
        return max(0, self._odometer.deletion_capacity - self._odometer.deletions_count)
        
    def get_accountant_type(self) -> str:
        """Return accountant type identifier."""
        return "zCDP"
        
    def get_budget_summary(self) -> Dict[str, float]:
        """Get zCDP budget summary."""
        return {
            'type': 'zCDP',
            'rho_total': self._odometer.rho_total,
            'rho_spent': self._odometer.rho_spent,
            'rho_remaining': self._odometer.rho_total - self._odometer.rho_spent,
            'delta_total': self._odometer.delta_total,
            'deletions_count': self._odometer.deletions_count,
            'deletion_capacity': self._odometer.deletion_capacity,
            'current_eps_delta': self._odometer.to_eps_delta(self._odometer.delta_total)
        }
        
    def supports_recalibration(self) -> bool:
        """zCDP accountant supports recalibration."""
        return True
        
    def recalibrate_with(self, new_stats: Dict[str, Any], remaining_T: int) -> None:
        """Recalibrate using ZCDPOdometer's recalibration."""
        self._odometer.recalibrate_with(new_stats, remaining_T)


class EpsDeltaAccountant(PrivacyAccountant):
    """
    (ε,δ)-DP privacy accountant.
    
    Uses traditional epsilon-delta differential privacy accounting
    with uniform budget allocation. Wraps the existing PrivacyOdometer.
    """
    
    def __init__(self, budget_params: Dict[str, float], **kwargs):
        """
        Initialize (ε,δ)-DP accountant.
        
        Args:
            budget_params: Must contain 'eps_total' and 'delta_total'
            **kwargs: Additional parameters for PrivacyOdometer
        """
        # Import here to avoid circular imports
        from .odometer import PrivacyOdometer
        
        self.eps_total = budget_params['eps_total']
        self.delta_total = budget_params['delta_total']
        
        # Create underlying PrivacyOdometer
        self._odometer = PrivacyOdometer(
            eps_total=self.eps_total,
            delta_total=self.delta_total,
            **kwargs
        )
        
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """Finalize using PrivacyOdometer's capacity calculation."""
        self._odometer.finalize_with(stats, T_estimate)
        
    def spend(self, sensitivity: float, **kwargs) -> None:
        """Spend uniform budget per deletion."""
        self._odometer.spend()
        
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get fixed noise scale from underlying odometer."""
        return self._odometer.noise_scale(sensitivity)
        
    def remaining_capacity(self) -> int:
        """Get remaining deletion capacity."""
        return max(0, self._odometer.deletion_capacity - self._odometer.deletions_count)
        
    def get_accountant_type(self) -> str:
        """Return accountant type identifier."""
        return "eps_delta"
        
    def get_budget_summary(self) -> Dict[str, float]:
        """Get (ε,δ)-DP budget summary."""
        return {
            'type': 'eps_delta',
            'eps_total': self._odometer.eps_total,
            'eps_spent': self._odometer.eps_spent,
            'eps_remaining': self._odometer.remaining(),
            'delta_total': self._odometer.delta_total,
            'deletions_count': self._odometer.deletions_count,
            'deletion_capacity': self._odometer.deletion_capacity,
            'eps_step': self._odometer.eps_step,
            'delta_step': self._odometer.delta_step
        }


class RelaxedAccountant(PrivacyAccountant):
    """
    Relaxed privacy accountant (placeholder for experimental methods).
    
    This is a placeholder implementation for experimental privacy accounting
    methods that may use relaxed constraints or alternative composition theorems.
    Currently implements a simple variant of (ε,δ)-DP with looser bounds.
    """
    
    def __init__(self, budget_params: Dict[str, float], relaxation_factor: float = 0.5, **kwargs):
        """
        Initialize relaxed accountant.
        
        Args:
            budget_params: Must contain 'eps_total' and 'delta_total'
            relaxation_factor: Factor to relax privacy constraints (0.5 = 50% relaxation)
            **kwargs: Additional parameters
        """
        self.eps_total = budget_params['eps_total']
        self.delta_total = budget_params['delta_total']
        self.relaxation_factor = relaxation_factor
        
        # Apply relaxation to budget (increase effective budget)
        self.effective_eps = self.eps_total / max(relaxation_factor, 0.1)
        self.effective_delta = self.delta_total / max(relaxation_factor, 0.1)
        
        # Initialize with relaxed budget parameters
        relaxed_params = {
            'eps_total': self.effective_eps,
            'delta_total': self.effective_delta
        }
        
        # Use EpsDeltaAccountant as base with relaxed parameters
        self._base_accountant = EpsDeltaAccountant(relaxed_params, **kwargs)
        
        warnings.warn(
            f"RelaxedAccountant is experimental and may not provide formal privacy guarantees. "
            f"Using relaxation_factor={relaxation_factor}",
            UserWarning
        )
        
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """Finalize using base accountant with relaxed parameters."""
        self._base_accountant.finalize_with(stats, T_estimate)
        
    def spend(self, sensitivity: float, **kwargs) -> None:
        """Spend budget using base accountant."""
        self._base_accountant.spend(sensitivity, **kwargs)
        
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get noise scale with relaxation applied."""
        base_scale = self._base_accountant.noise_scale(sensitivity)
        # Reduce noise by relaxation factor (more relaxed = less noise)
        return base_scale * self.relaxation_factor
        
    def remaining_capacity(self) -> int:
        """Get remaining capacity from base accountant."""
        return self._base_accountant.remaining_capacity()
        
    def get_accountant_type(self) -> str:
        """Return accountant type identifier."""
        return f"relaxed_{self.relaxation_factor}"
        
    def get_budget_summary(self) -> Dict[str, float]:
        """Get relaxed budget summary."""
        base_summary = self._base_accountant.get_budget_summary()
        base_summary.update({
            'type': f'relaxed_{self.relaxation_factor}',
            'original_eps_total': self.eps_total,
            'original_delta_total': self.delta_total,
            'relaxation_factor': self.relaxation_factor,
            'effective_noise_scale': self.noise_scale()
        })
        return base_summary


def create_accountant(accountant_type: str, budget_params: Dict[str, float], **kwargs) -> PrivacyAccountant:
    """
    Factory function to create privacy accountants.
    
    Args:
        accountant_type: Type of accountant ('zCDP', 'eps_delta', 'relaxed')
        budget_params: Budget parameters dict
        **kwargs: Additional accountant-specific parameters
        
    Returns:
        Configured privacy accountant
        
    Raises:
        ValueError: If accountant_type is not recognized
    """
    accountant_type = accountant_type.lower()
    
    if accountant_type == 'zcdp':
        if 'rho_total' not in budget_params:
            raise ValueError("zCDP accountant requires 'rho_total' in budget_params")
        return ZCDPAccountant(budget_params, **kwargs)
        
    elif accountant_type == 'eps_delta':
        if 'eps_total' not in budget_params:
            raise ValueError("eps_delta accountant requires 'eps_total' in budget_params")
        return EpsDeltaAccountant(budget_params, **kwargs)
        
    elif accountant_type == 'relaxed':
        if 'eps_total' not in budget_params:
            raise ValueError("relaxed accountant requires 'eps_total' in budget_params")
        return RelaxedAccountant(budget_params, **kwargs)
        
    else:
        raise ValueError(f"Unknown accountant type: {accountant_type}. "
                        f"Supported types: 'zCDP', 'eps_delta', 'relaxed'")


# Utility functions for budget conversion
def eps_delta_to_zcdp(eps: float, delta: float) -> float:
    """
    Convert (ε,δ)-DP parameters to approximate zCDP parameter ρ.
    
    Uses the conversion: ρ ≈ ε²/(2*log(1/δ))
    
    Args:
        eps: Epsilon parameter
        delta: Delta parameter
        
    Returns:
        Approximate zCDP parameter ρ
    """
    if delta <= 0:
        return float('inf')
    return eps**2 / (2 * math.log(1 / delta))


def zcdp_to_eps_delta(rho: float, delta: float) -> float:
    """
    Convert zCDP parameter ρ to (ε,δ)-DP.
    
    Args:
        rho: zCDP parameter
        delta: Target failure probability
        
    Returns:
        Corresponding ε value for (ε, δ)-DP
    """
    if delta <= 0:
        return float("inf")
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))