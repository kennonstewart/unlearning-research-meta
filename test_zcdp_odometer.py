#!/usr/bin/env python3
"""
Unit tests for ZCDPOdometer refactoring.

Tests cover the requirements from the problem statement:
1. Budget monotonicity: after each `spend`, `rho_spent` increases and never exceeds `rho_total`
2. Noise calibration: for a fixed `sens=1`, `rho_step=0.25` ⇒ expected `sigma≈1.4142`
3. Capacity calculation: with `rho_total=1`, `rho_step=0.1` ⇒ capacity `m≥9`
"""

import sys
import os
import numpy as np
import math
import warnings

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

from code.memory_pair.src.odometer import ZCDPOdometer, RDPOdometer, rho_to_epsilon


def test_budget_monotonicity():
    """Test that rho_spent increases after each spend and never exceeds rho_total."""
    print("\n=== Testing Budget Monotonicity ===")
    
    odometer = ZCDPOdometer(rho_total=1.0, delta_total=1e-5)
    
    # Finalize with simple stats
    stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
    odometer.finalize_with(stats, T_estimate=1000)
    
    assert odometer.ready_to_delete
    assert odometer.rho_spent == 0.0
    print(f"✅ Initial rho_spent: {odometer.rho_spent}")
    
    # Test spending budget
    initial_spent = odometer.rho_spent
    sensitivity = 0.5
    sigma = odometer.noise_scale()
    
    # Perform several spends
    for i in range(min(3, odometer.deletion_capacity)):
        prev_spent = odometer.rho_spent
        odometer.spend(sensitivity, sigma)
        
        # Check monotonicity
        assert odometer.rho_spent > prev_spent, f"rho_spent should increase: {prev_spent} -> {odometer.rho_spent}"
        assert odometer.rho_spent <= odometer.rho_total, f"rho_spent should not exceed rho_total: {odometer.rho_spent} > {odometer.rho_total}"
        
        expected_cost = odometer.rho_cost_gaussian(sensitivity, sigma)
        print(f"✅ Spend {i+1}: rho_spent = {odometer.rho_spent:.6f}, cost = {expected_cost:.6f}")
    
    print("✅ Budget monotonicity test passed")


def test_noise_calibration():
    """Test noise calibration: for sens=1, rho_step=0.25 ⇒ expected sigma≈1.4142."""
    print("\n=== Testing Noise Calibration ===")
    
    # For rho_step = sens²/(2σ²) = 0.25
    # With sens = 1: 0.25 = 1/(2σ²) ⇒ σ² = 2 ⇒ σ = √2 ≈ 1.4142
    sens = 1.0
    rho_step = 0.25
    expected_sigma = math.sqrt(2.0)  # ≈ 1.4142
    
    # Calculate sigma from the formula
    # rho_step = sens²/(2σ²) ⇒ σ = sens/√(2*rho_step)
    calculated_sigma = sens / math.sqrt(2 * rho_step)
    
    print(f"✅ Expected σ: {expected_sigma:.6f}")
    print(f"✅ Calculated σ: {calculated_sigma:.6f}")
    assert abs(calculated_sigma - expected_sigma) < 1e-6
    
    # Test the rho_cost_gaussian function
    odometer = ZCDPOdometer()
    calculated_rho = odometer.rho_cost_gaussian(sens, calculated_sigma)
    print(f"✅ Calculated ρ cost: {calculated_rho:.6f}")
    assert abs(calculated_rho - rho_step) < 1e-6
    
    print("✅ Noise calibration test passed")


def test_capacity_calculation():
    """Test capacity calculation: with rho_total=1, rho_step=0.1 ⇒ capacity m≥9."""
    print("\n=== Testing Capacity Calculation ===")
    
    # With rho_total = 1 and rho_step = 0.1 per deletion
    # Maximum capacity = rho_total / rho_step = 1 / 0.1 = 10 deletions
    # So we should get m ≥ 9 (accounting for regret constraints)
    rho_total = 1.0
    rho_step = 0.1
    expected_min_capacity = 9
    
    # Create odometer with generous regret constraint to allow high capacity
    odometer = ZCDPOdometer(
        rho_total=rho_total, 
        delta_total=1e-5,
        gamma=2.0,  # Large gamma to allow high capacity
        lambda_=1.0,  # Higher lambda for smaller regret terms
        T=1000
    )
    
    # Set up stats to achieve desired rho_step
    # rho_step = sens_bound²/(2σ²) = 0.1
    # If we want σ to be reasonable (say σ=1), then sens_bound = √(2*0.1*1²) = √0.2 ≈ 0.447
    sens_bound = math.sqrt(2 * rho_step * 1.0)  # σ=1
    
    stats = {
        "G": sens_bound,  # This will become L, and sens_bound = L/λ = L/1.0 = L
        "D": 0.1,  # Small D to minimize insertion regret
        "c": 1.0,
        "C": 1.0
    }
    
    odometer.finalize_with(stats, T_estimate=1000)
    
    print(f"✅ rho_total: {rho_total}")
    print(f"✅ Target rho_step: {rho_step}")
    print(f"✅ Expected min capacity: {expected_min_capacity}")
    print(f"✅ Actual capacity: {odometer.deletion_capacity}")
    print(f"✅ Computed σ: {odometer.sigma_step:.6f}")
    
    # Verify the capacity calculation
    actual_rho_step = rho_total / odometer.deletion_capacity
    print(f"✅ Actual rho_step: {actual_rho_step:.6f}")
    
    assert odometer.deletion_capacity >= expected_min_capacity, f"Capacity {odometer.deletion_capacity} should be >= {expected_min_capacity}"
    
    print("✅ Capacity calculation test passed")


def test_rho_to_epsilon_conversion():
    """Test the rho_to_epsilon utility function."""
    print("\n=== Testing rho_to_epsilon Conversion ===")
    
    # Test cases
    test_cases = [
        (0.5, 1e-5),
        (1.0, 1e-6),
        (2.0, 1e-5),
    ]
    
    for rho, delta in test_cases:
        epsilon = rho_to_epsilon(rho, delta)
        expected = rho + 2 * math.sqrt(rho * math.log(1 / delta))
        
        print(f"✅ ρ={rho}, δ={delta} ⇒ ε={epsilon:.6f} (expected: {expected:.6f})")
        assert abs(epsilon - expected) < 1e-10
    
    # Test edge case: delta = 0 should return inf
    assert rho_to_epsilon(1.0, 0.0) == float("inf")
    print("✅ Edge case δ=0 ⇒ ε=∞")
    
    print("✅ rho_to_epsilon conversion test passed")


def test_backward_compatibility():
    """Test that RDPOdometer backward compatibility shim works."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test that RDPOdometer issues a deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        odometer = RDPOdometer(eps_total=1.0, delta_total=1e-5)
        
        # Check that a deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "RDPOdometer is deprecated" in str(w[0].message)
        print("✅ Deprecation warning issued correctly")
    
    # Test that it inherits from ZCDPOdometer
    assert isinstance(odometer, ZCDPOdometer)
    print("✅ RDPOdometer inherits from ZCDPOdometer")
    
    # Test basic functionality
    stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
    odometer.finalize_with(stats, T_estimate=1000)
    assert odometer.ready_to_delete
    print("✅ Basic functionality works through inheritance")
    
    print("✅ Backward compatibility test passed")


def test_zcdp_spend_budget_check():
    """Test that spend() correctly checks budget overflow."""
    print("\n=== Testing zCDP Budget Overflow Check ===")
    
    # Create odometer with small budget
    odometer = ZCDPOdometer(rho_total=0.1, delta_total=1e-5, gamma=5.0)
    
    stats = {"G": 1.0, "D": 0.1, "c": 1.0, "C": 1.0}
    odometer.finalize_with(stats, T_estimate=1000)
    
    # Try to spend more than the total budget
    large_sensitivity = 10.0  # Large sensitivity
    small_sigma = 0.1  # Small sigma
    
    try:
        odometer.spend(large_sensitivity, small_sigma)
        assert False, "Should have raised RuntimeError for budget overflow"
    except RuntimeError as e:
        assert "zCDP budget exceeded" in str(e)
        print(f"✅ Budget overflow correctly detected: {str(e)}")
    
    print("✅ zCDP budget overflow test passed")


if __name__ == "__main__":
    test_budget_monotonicity()
    test_noise_calibration()
    test_capacity_calculation()
    test_rho_to_epsilon_conversion()
    test_backward_compatibility()
    test_zcdp_spend_budget_check()
    print("\n🎉 All ZCDPOdometer tests passed!")