#!/usr/bin/env python3
"""
Adaptive Capacity Odometer with Alternative Privacy Accounting.

This module implements an adaptive odometer that can switch between different
privacy accounting strategies (zCDP, (ε,δ)-DP, relaxed) and provides:

1. Runtime selection of privacy accounting strategy
2. Dynamic capacity adjustment based on observed statistics
3. Integration with pathwise comparator statistics for drift adjustment
4. Comprehensive logging and metrics collection
5. Noise calibration integration consistent with Theorem 5.4

The adaptive odometer maintains compatibility with existing odometer interfaces
while providing enhanced functionality for experimental comparison of different
privacy accounting methods.
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import warnings

from .privacy_accountants import (
    PrivacyAccountant, 
    create_accountant,
    ZCDPAccountant,
    EpsDeltaAccountant, 
    RelaxedAccountant
)


@dataclass
class DeletionEvent:
    """Record of a single deletion event for logging and analysis."""
    event_id: int
    accountant_type: str
    sensitivity: float
    noise_scale: float
    budget_spent: Dict[str, float]
    remaining_capacity: int
    gradient_magnitude: Optional[float] = None
    pathwise_drift: Optional[float] = None


class AdaptiveCapacityOdometer:
    """
    Adaptive Privacy Odometer with configurable accounting strategies.
    
    This odometer provides a unified interface for different privacy accounting
    methods while supporting dynamic capacity adjustment and comprehensive
    logging for experimental comparison.
    
    Key features:
    - Runtime selection of privacy accounting strategy
    - Dynamic recalibration based on observed gradient statistics
    - Integration with pathwise comparator drift detection
    - Detailed logging of deletion events and budget usage
    - Support for matched-seed experiments across accountant types
    
    Attributes:
        accountant (PrivacyAccountant): Current privacy accounting strategy
        deletion_events (List[DeletionEvent]): Log of all deletion events
        recalibration_enabled (bool): Whether to perform dynamic recalibration
        recalibration_interval (int): Number of deletions between recalibrations
        drift_threshold (float): Threshold for pathwise drift adjustment
        comparator (Optional): Pathwise comparator for drift detection
    """
    
    def __init__(
        self,
        accountant_type: str = "zCDP",
        budget_params: Optional[Dict[str, float]] = None,
        recalibration_enabled: bool = True,
        recalibration_interval: int = 10,
        drift_threshold: float = 0.2,
        **accountant_kwargs
    ):
        """
        Initialize adaptive capacity odometer.
        
        Args:
            accountant_type: Type of privacy accountant ('zCDP', 'eps_delta', 'relaxed')
            budget_params: Budget parameters for the accountant
            recalibration_enabled: Whether to enable dynamic recalibration
            recalibration_interval: Number of deletions between recalibrations
            drift_threshold: Threshold for significant pathwise drift
            **accountant_kwargs: Additional parameters for the accountant
        """
        # Set default budget parameters if not provided
        if budget_params is None:
            if accountant_type.lower() == 'zcdp':
                budget_params = {'rho_total': 1.0, 'delta_total': 1e-5}
            else:
                budget_params = {'eps_total': 1.0, 'delta_total': 1e-5}
        
        # Create privacy accountant
        self.accountant = create_accountant(accountant_type, budget_params, **accountant_kwargs)
        
        # Adaptive recalibration settings
        self.recalibration_enabled = recalibration_enabled
        self.recalibration_interval = recalibration_interval
        self.drift_threshold = drift_threshold
        
        # Event logging
        self.deletion_events: List[DeletionEvent] = []
        self.event_counter = 0
        
        # Pathwise comparator (optional)
        self.comparator: Optional[Any] = None
        
        # Cached statistics for recalibration
        self._last_stats: Optional[Dict[str, Any]] = None
        self._last_T_estimate: Optional[int] = None
        
        # State tracking
        self.ready_to_delete = False
        self._finalized = False
        
        print(f"[AdaptiveOdometer] Initialized with {self.accountant.get_accountant_type()} accountant")
        if recalibration_enabled:
            print(f"[AdaptiveOdometer] Recalibration enabled: interval={recalibration_interval}, drift_threshold={drift_threshold}")
    
    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """
        Finalize odometer configuration using calibration statistics.
        
        Args:
            stats: Dictionary containing calibration results (G, D, c, C)
            T_estimate: Estimated total number of events
        """
        # Store stats for potential recalibration
        self._last_stats = stats.copy()
        self._last_T_estimate = T_estimate
        
        # Finalize the underlying accountant
        self.accountant.finalize_with(stats, T_estimate)
        
        self.ready_to_delete = True
        self._finalized = True
        
        # Log initial capacity
        initial_summary = self.accountant.get_budget_summary()
        print(f"[AdaptiveOdometer] Finalized with initial capacity: {initial_summary['deletion_capacity']}")
        print(f"[AdaptiveOdometer] Budget summary: {initial_summary}")
    
    def set_comparator(self, comparator: Any) -> None:
        """
        Set pathwise comparator for drift-aware recalibration.
        
        Args:
            comparator: Pathwise comparator implementing get_oracle_metrics()
        """
        self.comparator = comparator
        print(f"[AdaptiveOdometer] Pathwise comparator attached for drift detection")
    
    def spend(
        self, 
        sensitivity: float, 
        gradient_magnitude: Optional[float] = None,
        **kwargs
    ) -> DeletionEvent:
        """
        Spend privacy budget for one deletion operation with comprehensive logging.
        
        Args:
            sensitivity: L2 sensitivity of the deletion operation
            gradient_magnitude: Magnitude of gradient for this deletion (optional)
            **kwargs: Additional parameters for the accountant
            
        Returns:
            DeletionEvent record for logging and analysis
            
        Raises:
            RuntimeError: If odometer is not finalized or capacity is exhausted
        """
        if not self.ready_to_delete:
            raise RuntimeError(
                "Odometer not finalized. Call finalize_with() before spending."
            )
        
        # Check if we need to recalibrate before spending
        if self._should_recalibrate():
            self._perform_recalibration()
        
        # Get noise scale before spending
        noise_scale = self.accountant.noise_scale(sensitivity)
        
        # Record pre-spend state
        pre_spend_summary = self.accountant.get_budget_summary()
        pre_spend_capacity = self.accountant.remaining_capacity()
        
        # Spend budget through accountant
        self.accountant.spend(sensitivity, sigma=noise_scale, **kwargs)
        
        # Record post-spend state
        post_spend_summary = self.accountant.get_budget_summary()
        remaining_capacity = self.accountant.remaining_capacity()
        
        # Compute budget spent for this deletion
        budget_spent = self._compute_budget_spent(pre_spend_summary, post_spend_summary)
        
        # Get pathwise drift if comparator available
        pathwise_drift = None
        if self.comparator is not None:
            try:
                oracle_metrics = self.comparator.get_oracle_metrics()
                pathwise_drift = oracle_metrics.get('P_T_est', None)
            except Exception as e:
                print(f"[AdaptiveOdometer] Warning: Could not get pathwise drift: {e}")
        
        # Create deletion event record
        event = DeletionEvent(
            event_id=self.event_counter,
            accountant_type=self.accountant.get_accountant_type(),
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            budget_spent=budget_spent,
            remaining_capacity=remaining_capacity,
            gradient_magnitude=gradient_magnitude,
            pathwise_drift=pathwise_drift
        )
        
        # Log the event
        self.deletion_events.append(event)
        self.event_counter += 1
        
        print(f"[AdaptiveOdometer] Deletion {self.event_counter}: "
              f"sensitivity={sensitivity:.4f}, noise_scale={noise_scale:.4f}, "
              f"remaining_capacity={remaining_capacity}")
        
        return event
    
    def _should_recalibrate(self) -> bool:
        """Determine if recalibration should be performed."""
        if not self.recalibration_enabled:
            return False
        
        if not self.accountant.supports_recalibration():
            return False
        
        # Check recalibration interval
        if len(self.deletion_events) % self.recalibration_interval != 0:
            return False
        
        # Check for significant drift if comparator available
        if self.comparator is not None:
            try:
                oracle_metrics = self.comparator.get_oracle_metrics()
                current_drift = oracle_metrics.get('P_T_est', 0.0)
                
                # Compare with drift from previous recalibration
                if len(self.deletion_events) >= self.recalibration_interval:
                    prev_event = self.deletion_events[-self.recalibration_interval]
                    prev_drift = prev_event.pathwise_drift or 0.0
                    drift_change = abs(current_drift - prev_drift)
                    
                    if drift_change > self.drift_threshold:
                        print(f"[AdaptiveOdometer] Significant drift detected: {drift_change:.4f} > {self.drift_threshold}")
                        return True
            except Exception as e:
                print(f"[AdaptiveOdometer] Warning: Could not check drift: {e}")
        
        # Recalibrate based on interval if no drift detection
        return True
    
    def _perform_recalibration(self) -> None:
        """Perform dynamic recalibration of the accountant."""
        if self._last_stats is None or self._last_T_estimate is None:
            print("[AdaptiveOdometer] Warning: No cached stats for recalibration")
            return
        
        try:
            # Update statistics with observed gradient magnitudes if available
            updated_stats = self._update_stats_with_observations()
            
            # Estimate remaining T based on current progress
            remaining_T = max(100, self._last_T_estimate - len(self.deletion_events))
            
            print(f"[AdaptiveOdometer] Performing recalibration at deletion {len(self.deletion_events)}")
            print(f"[AdaptiveOdometer] Remaining T estimate: {remaining_T}")
            
            # Recalibrate the accountant
            self.accountant.recalibrate_with(updated_stats, remaining_T)
            
            # Log recalibration result
            post_recalib_summary = self.accountant.get_budget_summary()
            print(f"[AdaptiveOdometer] Recalibration complete: new capacity = {post_recalib_summary.get('deletion_capacity', 'N/A')}")
            
        except Exception as e:
            print(f"[AdaptiveOdometer] Warning: Recalibration failed: {e}")
    
    def _update_stats_with_observations(self) -> Dict[str, Any]:
        """Update calibration statistics with observed data."""
        if self._last_stats is None:
            return {}
        
        updated_stats = self._last_stats.copy()
        
        # Update gradient bound with observed gradient magnitudes
        if any(event.gradient_magnitude is not None for event in self.deletion_events):
            observed_gradients = [
                event.gradient_magnitude for event in self.deletion_events 
                if event.gradient_magnitude is not None
            ]
            if observed_gradients:
                # Use 95th percentile of observed gradients as updated bound
                updated_G = float(np.quantile(observed_gradients, 0.95))
                updated_stats['G'] = max(updated_stats.get('G', 1.0), updated_G)
                print(f"[AdaptiveOdometer] Updated gradient bound: G = {updated_stats['G']:.4f}")
        
        # Update sensitivity bound with observed sensitivities
        observed_sensitivities = [event.sensitivity for event in self.deletion_events]
        if observed_sensitivities:
            updated_sens = float(np.quantile(observed_sensitivities, 0.95))
            # Update gradient bound if no explicit gradient observations
            if 'G' in updated_stats:
                updated_stats['G'] = max(updated_stats['G'], updated_sens)
            print(f"[AdaptiveOdometer] Updated sensitivity bound: {updated_sens:.4f}")
        
        return updated_stats
    
    def _compute_budget_spent(
        self, 
        pre_summary: Dict[str, float], 
        post_summary: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute budget spent for this deletion."""
        budget_spent = {}
        
        if 'eps_spent' in pre_summary and 'eps_spent' in post_summary:
            budget_spent['eps_spent'] = post_summary['eps_spent'] - pre_summary['eps_spent']
        
        if 'rho_spent' in pre_summary and 'rho_spent' in post_summary:
            budget_spent['rho_spent'] = post_summary['rho_spent'] - pre_summary['rho_spent']
        
        return budget_spent
    
    def remaining_capacity(self) -> int:
        """Get remaining deletion capacity."""
        return self.accountant.remaining_capacity()
    
    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """Get noise scale for given sensitivity."""
        return self.accountant.noise_scale(sensitivity)
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget and usage summary."""
        base_summary = self.accountant.get_budget_summary()
        
        # Add adaptive odometer specific metrics
        base_summary.update({
            'total_deletions': len(self.deletion_events),
            'recalibration_enabled': self.recalibration_enabled,
            'recalibrations_performed': len(self.deletion_events) // self.recalibration_interval if self.recalibration_interval > 0 else 0
        })
        
        return base_summary
    
    def get_deletion_events(self) -> List[DeletionEvent]:
        """Get list of all deletion events for analysis."""
        return self.deletion_events.copy()
    
    def export_metrics_for_plotting(self) -> Dict[str, Any]:
        """
        Export metrics in format suitable for plotting and comparison.
        
        Returns:
            Dictionary with metrics for plotting regret vs accountant type
        """
        if not self.deletion_events:
            return {
                'accountant_type': self.accountant.get_accountant_type(),
                'deletion_count': 0,
                'budget_usage': [],
                'noise_scales': [],
                'sensitivities': [],
                'pathwise_drift': []
            }
        
        return {
            'accountant_type': self.accountant.get_accountant_type(),
            'deletion_count': len(self.deletion_events),
            'budget_usage': [
                sum(event.budget_spent.values()) if event.budget_spent else 0 
                for event in self.deletion_events
            ],
            'noise_scales': [event.noise_scale for event in self.deletion_events],
            'sensitivities': [event.sensitivity for event in self.deletion_events],
            'pathwise_drift': [
                event.pathwise_drift if event.pathwise_drift is not None else 0.0 
                for event in self.deletion_events
            ],
            'gradient_magnitudes': [
                event.gradient_magnitude if event.gradient_magnitude is not None else 0.0
                for event in self.deletion_events
            ]
        }
    
    def switch_accountant(
        self, 
        new_accountant_type: str, 
        new_budget_params: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Switch to a different privacy accountant (experimental feature).
        
        Args:
            new_accountant_type: Type of new accountant
            new_budget_params: Budget parameters for new accountant
            **kwargs: Additional parameters for new accountant
            
        Note:
            This is an experimental feature. Switching accountants mid-stream
            may not provide formal privacy guarantees.
        """
        warnings.warn(
            "Switching accountants mid-stream is experimental and may not provide "
            "formal privacy guarantees. Use for experimental comparison only.",
            UserWarning
        )
        
        # Create new accountant
        if new_budget_params is None:
            # Use remaining budget from current accountant
            current_summary = self.accountant.get_budget_summary()
            if new_accountant_type.lower() == 'zcdp':
                new_budget_params = {
                    'rho_total': current_summary.get('rho_remaining', 1.0),
                    'delta_total': current_summary.get('delta_total', 1e-5)
                }
            else:
                new_budget_params = {
                    'eps_total': current_summary.get('eps_remaining', 1.0),
                    'delta_total': current_summary.get('delta_total', 1e-5)
                }
        
        # Create and finalize new accountant
        new_accountant = create_accountant(new_accountant_type, new_budget_params, **kwargs)
        
        if self._finalized and self._last_stats is not None and self._last_T_estimate is not None:
            # Adjust T estimate for remaining work
            remaining_T = max(100, self._last_T_estimate - len(self.deletion_events))
            new_accountant.finalize_with(self._last_stats, remaining_T)
        
        # Switch accountants
        old_type = self.accountant.get_accountant_type()
        self.accountant = new_accountant
        
        print(f"[AdaptiveOdometer] Switched from {old_type} to {self.accountant.get_accountant_type()}")
        print(f"[AdaptiveOdometer] New budget summary: {self.accountant.get_budget_summary()}")


# Legacy compatibility functions
def create_legacy_odometer(accountant_type: str = "zCDP", **kwargs) -> AdaptiveCapacityOdometer:
    """
    Create an adaptive odometer with legacy-compatible interface.
    
    Args:
        accountant_type: Type of privacy accountant to use
        **kwargs: Additional parameters for odometer/accountant
        
    Returns:
        AdaptiveCapacityOdometer configured for legacy compatibility
    """
    return AdaptiveCapacityOdometer(
        accountant_type=accountant_type,
        recalibration_enabled=False,  # Disable for legacy compatibility
        **kwargs
    )