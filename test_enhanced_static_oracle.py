"""
Test enhanced StaticOracle with delete support and refresh periods.
"""

import sys
import os
sys.path.append('./code')

import numpy as np
from code.memory_pair.src.comparators import StaticOracle
from code.memory_pair.src.metrics import loss_half_mse


def test_static_oracle_erm_correctness():
    """Test that StaticOracleERM equals closed-form ridge solution on the full prefix."""
    lambda_reg = 0.1
    
    # Create oracle
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    
    # Test data
    data = [
        (np.array([1.0, 0.0]), 1.0),
        (np.array([0.0, 1.0]), 0.5),
        (np.array([0.5, 0.5]), 0.75),
    ]
    
    # Calibrate oracle with initial data
    oracle.calibrate_with_initial_data(data)
    
    # Add more data via update_regret_accounting
    x_new = np.array([0.3, 0.7])
    y_new = 0.6
    current_theta = np.array([0.5, 0.3])
    
    oracle.update_regret_accounting(x_new, y_new, current_theta)
    
    # Manually compute the closed-form solution
    X = np.array([d[0] for d in data] + [x_new])
    Y = np.array([d[1] for d in data] + [y_new])
    
    XtX = X.T @ X
    XtY = X.T @ Y
    A = XtX + lambda_reg * np.eye(2)
    w_star_expected = np.linalg.solve(A, XtY)
    
    # Compare with oracle solution
    assert np.allclose(oracle.w_star_fixed, w_star_expected, atol=1e-10), \
        "StaticOracle should match closed-form ridge solution"
    
    print("✓ Static ERM correctness test passed")


def test_oracle_refresh_period():
    """Test oracle refresh period configuration."""
    lambda_reg = 0.1
    
    # Create config object with refresh period
    class Config:
        oracle_refresh_period = 3
    
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=Config())
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    
    current_theta = np.array([0.5, 0.3])
    
    # Add samples and check refresh behavior
    initial_w_star = oracle.w_star_fixed.copy()
    initial_refreshes = oracle.oracle_refreshes
    
    # Add 2 samples - should not refresh yet
    oracle.update_regret_accounting(np.array([0.1, 0.2]), 0.5, current_theta)
    oracle.update_regret_accounting(np.array([0.2, 0.3]), 0.6, current_theta)
    
    assert oracle.oracle_refreshes == initial_refreshes, \
        "Oracle should not refresh before period"
    
    # Add 3rd sample - should trigger refresh
    oracle.update_regret_accounting(np.array([0.3, 0.4]), 0.7, current_theta)
    
    assert oracle.oracle_refreshes == initial_refreshes + 1, \
        "Oracle should refresh after period"
    assert not np.allclose(oracle.w_star_fixed, initial_w_star), \
        "Oracle weights should update after refresh"
    
    print("✓ Oracle refresh period test passed")


def test_delete_correctness():
    """Test that delete operations correctly update statistics."""
    lambda_reg = 0.1
    
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    
    # Initial data
    data = [
        (np.array([1.0, 0.0]), 1.0),
        (np.array([0.0, 1.0]), 0.5),
        (np.array([0.5, 0.5]), 0.75),
    ]
    
    oracle.calibrate_with_initial_data(data)
    current_theta = np.array([0.5, 0.3])
    
    # Add a sample
    x_add = np.array([0.3, 0.7])
    y_add = 0.6
    oracle.update_regret_accounting(x_add, y_add, current_theta)
    
    # Get oracle weights with all data
    w_star_full = oracle.w_star_fixed.copy()
    
    # Remove the added sample
    success = oracle.process_delete_event(x_add, y_add)
    assert success, "Delete should succeed for recently added sample"
    
    # Oracle should now match what we had before adding the sample
    oracle_after_calibration = StaticOracle(dim=2, lambda_reg=lambda_reg)
    oracle_after_calibration.calibrate_with_initial_data(data)
    
    assert np.allclose(oracle.w_star_fixed, oracle_after_calibration.w_star_fixed, atol=1e-10), \
        "Oracle after delete should match state before add"
    
    # Try to delete a non-existent sample
    success = oracle.process_delete_event(np.array([99.0, 99.0]), 99.0)
    assert not success, "Delete should fail for non-existent sample"
    
    print("✓ Delete correctness test passed")


def test_erm_solver_options():
    """Test different ERM solver options."""
    lambda_reg = 0.1
    
    solvers = ['chol', 'svd', 'default']
    solutions = []
    
    data = [
        (np.array([1.0, 0.0]), 1.0),
        (np.array([0.0, 1.0]), 0.5),
        (np.array([0.5, 0.5]), 0.75),
    ]
    
    for solver in solvers:
        class Config:
            erm_solver = solver
            
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=Config())
        oracle.calibrate_with_initial_data(data)
        solutions.append(oracle.w_star_fixed.copy())
    
    # All solvers should give the same result (within numerical precision)
    for i in range(1, len(solutions)):
        assert np.allclose(solutions[0], solutions[i], atol=1e-8), \
            f"Solver {solvers[i]} should give same result as {solvers[0]}"
    
    print("✓ ERM solver options test passed")


def test_logging_fields():
    """Test that required logging fields are present."""
    lambda_reg = 0.1
    
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    
    metrics = oracle.get_oracle_metrics()
    
    # Check required fields
    required_fields = [
        "comparator_type", 
        "oracle_refresh_step", 
        "oracle_refreshes",
        "regret_static",
        "is_calibrated",
        "events_seen"
    ]
    
    for field in required_fields:
        assert field in metrics, f"Required field {field} missing from metrics"
    
    # Check specific values
    assert metrics["comparator_type"] == "static_oracle_erm_fullprefix", \
        "Comparator type should be theory-faithful name"
    
    print("✓ Logging fields test passed")


if __name__ == "__main__":
    print("Running enhanced StaticOracle tests...")
    
    test_static_oracle_erm_correctness()
    test_oracle_refresh_period()
    test_delete_correctness()
    test_erm_solver_options()
    test_logging_fields()
    
    print("\n🎉 All enhanced StaticOracle tests passed!")