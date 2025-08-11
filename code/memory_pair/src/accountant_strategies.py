"""
Accountant strategy interface and implementations for Milestone 5.

This module provides a unified interface for different privacy accounting methods:
- ZCDPAccountant: Zero-Concentrated Differential Privacy accounting
- EpsDeltaAccountant: Traditional (ε, δ)-DP accounting  
- RelaxedAccountant: Experimental relaxed mode for less conservative noise

Each accountant implements the same interface but uses different underlying
privacy accounting mechanisms and noise calibration strategies.
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Protocol, Union
from abc import ABC, abstractmethod

# Import existing odometer implementations
try:
    from .odometer import PrivacyOdometer, ZCDPOdometer, rho_to_epsilon
except ImportError:
    from odometer import PrivacyOdometer, ZCDPOdometer, rho_to_epsilon


class AccountantStrategy(Protocol):
    """Protocol interface for privacy accountants."""
    
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
    
    @property
    def accountant_type(self) -> str:
        """Get the type identifier for this accountant."""
        ...
    
    @property
    def deletion_capacity(self) -> int:
        """Get maximum number of deletions allowed."""
        ...
    
    @property
    def deletions_count(self) -> int:
        """Get number of deletions performed so far."""
        ...


class BaseAccountantStrategy(ABC):
    """Base class for accountant strategy implementations."""
    
    def __init__(self, 
                 eps_total: float = 1.0,
                 delta_total: float = 1e-5,
                 T: int = 10000,
                 gamma: float = 0.5,
                 lambda_: float = 0.1,
                 delta_b: float = 0.05,
                 **kwargs):
        """Initialize base accountant parameters."""
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta_b = delta_b
        self._extra_kwargs = kwargs
        
        # Will be set by concrete implementations
        self._underlying_odometer = None
        self._finalized = False
    
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """Finalize accountant with calibration statistics."""
        if self._underlying_odometer is None:
            raise RuntimeError("Underlying odometer not initialized")
        
        self._underlying_odometer.finalize_with(stats, T_estimate)
        self._finalized = True
    
    def spend(self, sensitivity: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """Spend privacy budget for one operation."""
        if not self._finalized:
            raise RuntimeError("Accountant not finalized")
        
        # Delegate to underlying odometer with appropriate parameters
        self._spend_impl(sensitivity, sigma)
    
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get noise scale for current operation."""
        if not self._finalized:
            raise RuntimeError("Accountant not finalized")
        return self._underlying_odometer.noise_scale(sensitivity)
    
    def over_budget(self) -> bool:
        """Check if privacy budget is exhausted."""
        if not self._finalized:
            return False
        return self._underlying_odometer.over_budget()
    
    @property
    def deletion_capacity(self) -> int:
        """Get maximum number of deletions allowed."""
        if self._underlying_odometer is None:
            return 0
        return self._underlying_odometer.deletion_capacity
    
    @property
    def deletions_count(self) -> int:
        """Get number of deletions performed so far."""
        if self._underlying_odometer is None:
            return 0
        return self._underlying_odometer.deletions_count
    
    @abstractmethod
    def _spend_impl(self, sensitivity: Optional[float], sigma: Optional[float]) -> None:
        """Implementation-specific spending logic."""
        pass
    
    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """Get current privacy metrics for logging."""
        pass


class ZCDPAccountant(BaseAccountantStrategy):
    """Zero-Concentrated Differential Privacy accountant strategy."""
    
    def __init__(self, 
                 eps_total: float = 1.0,
                 delta_total: float = 1e-5,
                 **kwargs):
        """Initialize zCDP accountant."""
        super().__init__(eps_total=eps_total, delta_total=delta_total, **kwargs)
        
        # Convert (ε, δ)-DP to zCDP: ρ ≈ ε²/(2*log(1/δ))
        rho_total = eps_total**2 / (2 * math.log(1 / delta_total))
        
        self._underlying_odometer = ZCDPOdometer(
            rho_total=rho_total,
            delta_total=delta_total,
            T=self.T,
            gamma=self.gamma,
            lambda_=self.lambda_,
            delta_b=self.delta_b,
            **self._extra_kwargs
        )
    
    @property
    def accountant_type(self) -> str:
        return "zcdp"
    
    def _spend_impl(self, sensitivity: Optional[float], sigma: Optional[float]) -> None:
        """Spend zCDP budget using actual sensitivity and noise scale."""
        if sensitivity is None or sigma is None:
            raise ValueError("zCDP accountant requires both sensitivity and sigma for spending")
        self._underlying_odometer.spend(sensitivity, sigma)
    
    def metrics(self) -> Dict[str, Any]:
        """Get zCDP-specific privacy metrics."""
        if not self._finalized:
            return {"accountant_type": self.accountant_type}
        
        return {
            "accountant_type": self.accountant_type,
            "rho_total": self._underlying_odometer.rho_total,
            "rho_spent": self._underlying_odometer.rho_spent,
            "rho_remaining": self._underlying_odometer.rho_total - self._underlying_odometer.rho_spent,
            "eps_converted": self._underlying_odometer.to_eps_delta(self.delta_total),
            "delta_total": self.delta_total,
            "sigma_step_theory": self._underlying_odometer.sigma_step,
            "deletion_capacity": self.deletion_capacity,
            "deletions_count": self.deletions_count,
            "capacity_remaining": max(0, self.deletion_capacity - self.deletions_count),
        }


class EpsDeltaAccountant(BaseAccountantStrategy):
    """Traditional (ε, δ)-DP accountant strategy."""
    
    def __init__(self, **kwargs):
        """Initialize (ε, δ)-DP accountant."""
        super().__init__(**kwargs)
        
        self._underlying_odometer = PrivacyOdometer(
            eps_total=self.eps_total,
            delta_total=self.delta_total,
            T=self.T,
            gamma=self.gamma,
            lambda_=self.lambda_,
            delta_b=self.delta_b,
        )
    
    @property
    def accountant_type(self) -> str:
        return "eps_delta"
    
    def _spend_impl(self, sensitivity: Optional[float], sigma: Optional[float]) -> None:
        """Spend (ε, δ)-DP budget using uniform per-step allocation."""
        # Traditional approach: uniform budget allocation
        self._underlying_odometer.spend()
    
    def metrics(self) -> Dict[str, Any]:
        """Get (ε, δ)-DP specific privacy metrics."""
        if not self._finalized:
            return {"accountant_type": self.accountant_type}
        
        return {
            "accountant_type": self.accountant_type,
            "eps_total": self.eps_total,
            "eps_spent": self._underlying_odometer.eps_spent,
            "eps_remaining": self._underlying_odometer.remaining(),
            "eps_step_theory": self._underlying_odometer.eps_step,
            "delta_total": self.delta_total,
            "delta_step_theory": self._underlying_odometer.delta_step,
            "sigma_step_theory": self._underlying_odometer.sigma_step,
            "deletion_capacity": self.deletion_capacity,
            "deletions_count": self.deletions_count,
            "capacity_remaining": max(0, self.deletion_capacity - self.deletions_count),
        }


class RelaxedAccountant(BaseAccountantStrategy):
    """Experimental relaxed privacy accountant with less conservative noise scaling."""
    
    def __init__(self, 
                 relaxation_factor: float = 0.8,
                 **kwargs):
        """Initialize relaxed accountant.
        
        Args:
            relaxation_factor: Factor to reduce noise (0.8 = 20% less noise)
        """
        super().__init__(**kwargs)
        
        self.relaxation_factor = max(0.1, min(1.0, relaxation_factor))  # Clamp to reasonable range
        
        # Use zCDP as base but with relaxed noise scaling
        rho_total = self.eps_total**2 / (2 * math.log(1 / self.delta_total))
        
        self._underlying_odometer = ZCDPOdometer(
            rho_total=rho_total,
            delta_total=self.delta_total,
            T=self.T,
            gamma=self.gamma,
            lambda_=self.lambda_,
            delta_b=self.delta_b,
            **self._extra_kwargs
        )
        
        # Store original noise scale for relaxed computation
        self._base_sigma = None
    
    @property
    def accountant_type(self) -> str:
        return "relaxed"
    
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """Finalize with relaxed noise scaling."""
        super().finalize_with(stats, T_estimate)
        self._base_sigma = self._underlying_odometer.sigma_step
        # Apply relaxation factor to reduce noise
        self._underlying_odometer.sigma_step = self._base_sigma * self.relaxation_factor
    
    def _spend_impl(self, sensitivity: Optional[float], sigma: Optional[float]) -> None:
        """Spend budget using relaxed noise scale."""
        # Use relaxed sigma for spending, but actual sensitivity
        if sensitivity is None:
            raise ValueError("Relaxed accountant requires sensitivity for spending")
        
        relaxed_sigma = sigma * self.relaxation_factor if sigma else self.noise_scale()
        self._underlying_odometer.spend(sensitivity, relaxed_sigma)
    
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get relaxed noise scale."""
        if not self._finalized:
            raise RuntimeError("Accountant not finalized")
        # Return the relaxed noise scale
        return self._underlying_odometer.sigma_step
    
    def metrics(self) -> Dict[str, Any]:
        """Get relaxed accountant specific privacy metrics."""
        if not self._finalized:
            return {"accountant_type": self.accountant_type}
        
        # Get base metrics from zCDP
        base_metrics = {
            "accountant_type": self.accountant_type,
            "rho_total": self._underlying_odometer.rho_total,
            "rho_spent": self._underlying_odometer.rho_spent,
            "rho_remaining": self._underlying_odometer.rho_total - self._underlying_odometer.rho_spent,
            "eps_converted": self._underlying_odometer.to_eps_delta(self.delta_total),
            "delta_total": self.delta_total,
            "sigma_step_theory": self._underlying_odometer.sigma_step,  # Relaxed sigma
            "sigma_step_base": self._base_sigma,  # Original unrelaxed sigma
            "relaxation_factor": self.relaxation_factor,
            "deletion_capacity": self.deletion_capacity,
            "deletions_count": self.deletions_count,
            "capacity_remaining": max(0, self.deletion_capacity - self.deletions_count),
        }
        
        return base_metrics


def create_accountant_strategy(accountant_type: str, **kwargs) -> AccountantStrategy:
    """Factory function to create accountant strategies.
    
    Args:
        accountant_type: One of "zcdp", "eps_delta", "relaxed", "default", "legacy"
        **kwargs: Parameters for the accountant
    
    Returns:
        Accountant strategy instance
    """
    # Handle legacy names
    if accountant_type in ["default", "legacy"]:
        accountant_type = "eps_delta"
    elif accountant_type == "rdp":
        accountant_type = "zcdp"
    
    if accountant_type == "zcdp":
        return ZCDPAccountant(**kwargs)
    elif accountant_type == "eps_delta":
        return EpsDeltaAccountant(**kwargs)
    elif accountant_type == "relaxed":
        return RelaxedAccountant(**kwargs)
    else:
        raise ValueError(f"Unknown accountant type: {accountant_type}")


# Backward compatibility: create a protocol adapter for existing AccountantAdapter
class StrategyAccountantAdapter:
    """Adapter to make AccountantStrategy work with existing Protocol interface."""
    
    def __init__(self, strategy: AccountantStrategy):
        self._strategy = strategy
    
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        return self._strategy.finalize_with(stats, T_estimate)
    
    def spend(self, sensitivity: Optional[float] = None, sigma: Optional[float] = None) -> None:
        return self._strategy.spend(sensitivity, sigma)
    
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        return self._strategy.noise_scale(sensitivity)
    
    def over_budget(self) -> bool:
        return self._strategy.over_budget()
    
    def metrics(self) -> Dict[str, Any]:
        return self._strategy.metrics()
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped strategy."""
        return getattr(self._strategy, name)