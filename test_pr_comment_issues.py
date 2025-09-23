"""
Tests for the specific issues mentioned in the PR comments to ensure they are fixed.
"""

import sys
import os
sys.path.append('./code')

import numpy as np
from code.memory_pair.src.comparators import StaticOracle
from code.memory_pair.src.metrics import loss_half_mse


def test_no_double_counting_stats():
    """
    Test that sufficient statistics are not double-counted.
    After N inserts, check xtx, xty equal the sum of outer products and cross terms 
    from the logged samples (no double counts).
    """
    lambda_reg = 0.1
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    
    # Initial calibration with known data
    initial_data = [(np.array([1.0, 0.0]), 1.0)]
    oracle.calibrate_with_initial_data(initial_data)
    
    # Add more samples
    test_samples = [
        (np.array([0.0, 1.0]), 0.5),
        (np.array([0.5, 0.5]), 0.75),
        (np.array([0.3, 0.7]), 0.6),
    ]
    
    current_theta = np.array([0.4, 0.3])
    
    # Track all samples for manual computation
    all_samples = initial_data + test_samples
    
    # Add samples via update_regret_accounting
    for x, y in test_samples:
        oracle.update_regret_accounting(x, y, current_theta)
    
    # Manually compute expected statistics
    expected_xtx = np.zeros((2, 2))
    expected_xty = np.zeros(2)
    
    for x, y in all_samples:
        expected_xtx += np.outer(x, x)
        expected_xty += x * y
    
    # Check that oracle statistics match expected (no double counting)
    assert np.allclose(oracle.xtx, expected_xtx, atol=1e-12), \
        f"xtx should match expected without double counting. Got {oracle.xtx}, expected {expected_xtx}"
    assert np.allclose(oracle.xty, expected_xty, atol=1e-12), \
        f"xty should match expected without double counting. Got {oracle.xty}, expected {expected_xty}"
    
    print("✓ No double counting stats test passed")


def test_single_source_regret():
    """
    Test that regret_increment == loss_curr_reg - loss_comp_reg for random mini-batch.
    Ensure there's no second code path that overwrites the regret calculation.
    """
    lambda_reg = 0.15
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    
    # Test multiple random samples
    np.random.seed(42)
    for _ in range(5):
        x = np.random.randn(2)
        y = np.random.randn()
        current_theta = np.random.randn(2)
        
        # Get regret from oracle
        result = oracle.update_regret_accounting(x, y, current_theta)
        regret_increment = result["regret_increment"]
        
        # Manually compute using the same _regret_terms method
        regret_terms = oracle._regret_terms(x, y, current_theta, oracle.w_star_fixed, lambda_reg)
        expected_regret = regret_terms["regret_inc"]
        
        # Should match exactly - no second code path
        assert abs(regret_increment - expected_regret) < 1e-12, \
            f"Regret should come from single source. Got {regret_increment}, expected {expected_regret}"
    
    print("✓ Single source regret test passed")


def test_events_counter_once_per_insert():
    """
    Test that events_seen increments once per insert event and refresh cadence 
    hits exactly on the configured period.
    """
    lambda_reg = 0.1
    
    # Test with refresh period 3
    class Config:
        oracle_refresh_period = 3
    
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=Config())
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    
    current_theta = np.array([0.4, 0.3])
    initial_events = oracle.events_seen
    initial_refreshes = oracle.oracle_refreshes
    
    # Add several samples and track events_seen
    test_samples = [
        (np.array([0.0, 1.0]), 0.5),    # Event 1 - no refresh
        (np.array([0.5, 0.5]), 0.75),   # Event 2 - no refresh  
        (np.array([0.3, 0.7]), 0.6),    # Event 3 - should refresh
        (np.array([0.1, 0.9]), 0.4),    # Event 4 - no refresh
        (np.array([0.7, 0.3]), 0.8),    # Event 5 - no refresh
        (np.array([0.2, 0.8]), 0.3),    # Event 6 - should refresh
    ]
    
    expected_events = initial_events
    expected_refreshes = initial_refreshes
    
    for i, (x, y) in enumerate(test_samples):
        oracle.update_regret_accounting(x, y, current_theta)
        expected_events += 1
        
        # Check events_seen increments by exactly 1
        assert oracle.events_seen == expected_events, \
            f"events_seen should increment by 1 per event. Expected {expected_events}, got {oracle.events_seen}"
        
        # Check refresh logic: should refresh on events 3 and 6 (every 3rd event)
        if (expected_events % 3) == 0:
            expected_refreshes += 1
            
        assert oracle.oracle_refreshes == expected_refreshes, \
            f"Oracle should refresh every 3 events. Expected {expected_refreshes} refreshes, got {oracle.oracle_refreshes}"
    
    print("✓ Events counter test passed")


def test_oracle_refresh_step_tracking():
    """Test that oracle_refresh_step is correctly updated when refresh happens."""
    lambda_reg = 0.1
    
    class Config:
        oracle_refresh_period = 2
        
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=Config())
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    
    current_theta = np.array([0.4, 0.3])
    
    # First event - should refresh (events_seen goes from 0 to 1, then 1 to 2)
    oracle.update_regret_accounting(np.array([0.0, 1.0]), 0.5, current_theta)
    # events_seen = 1, no refresh yet
    
    oracle.update_regret_accounting(np.array([0.5, 0.5]), 0.75, current_theta)
    # events_seen = 2, should refresh, oracle_refresh_step should be 2
    
    assert oracle.oracle_refresh_step == 2, \
        f"oracle_refresh_step should be updated to current events_seen on refresh. Expected 2, got {oracle.oracle_refresh_step}"
    
    print("✓ Oracle refresh step tracking test passed")


def test_unified_regret_terms():
    """Test that all comparators use the same _regret_terms implementation."""
    from code.memory_pair.src.comparators import RollingOracle, OracleState
    
    lambda_reg = 0.1
    x = np.array([0.5, 0.5])
    y = 0.7
    theta_curr = np.array([0.8, 0.3])
    theta_comp = np.array([0.6, 0.4])
    
    # Test StaticOracle
    static_oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    static_result = static_oracle._regret_terms(x, y, theta_curr, theta_comp, lambda_reg)
    
    # Test RollingOracle
    rolling_oracle = RollingOracle(dim=2, lambda_reg=lambda_reg)
    rolling_result = rolling_oracle._regret_terms(x, y, theta_curr, theta_comp, lambda_reg)
    
    # Should get identical results
    assert abs(static_result["regret_inc"] - rolling_result["regret_inc"]) < 1e-12, \
        "Static and rolling oracles should use identical _regret_terms implementation"
    assert abs(static_result["loss_curr_reg"] - rolling_result["loss_curr_reg"]) < 1e-12, \
        "Static and rolling oracles should compute identical current loss"
    assert abs(static_result["loss_comp_reg"] - rolling_result["loss_comp_reg"]) < 1e-12, \
        "Static and rolling oracles should compute identical comparator loss"
    
    print("✓ Unified regret terms test passed")


if __name__ == "__main__":
    print("Running tests for specific PR comment issues...")
    
    test_no_double_counting_stats()
    test_single_source_regret() 
    test_events_counter_once_per_insert()
    test_oracle_refresh_step_tracking()
    test_unified_regret_terms()
    
    print("\n🎉 All PR comment issue tests passed!")