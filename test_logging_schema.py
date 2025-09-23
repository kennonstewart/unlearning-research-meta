"""
Test logging and schema semantics for regret calculation.
"""

import sys
import os
sys.path.append('./code')

import numpy as np
from code.memory_pair.src.memory_pair import MemoryPair, Phase
from code.memory_pair.src.comparators import StaticOracle
from code.memory_pair.src.metrics import loss_half_mse


def test_schema_invariants():
    """Test that schema invariants are maintained."""
    lambda_reg = 0.1
    
    # Create MemoryPair with oracle
    mp = MemoryPair(dim=2)
    mp.lambda_reg = lambda_reg
    mp.phase = Phase.LEARNING
    
    # Create and attach oracle
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    mp.oracle = oracle
    
    # Perform an insert
    x = np.array([0.5, 0.5])
    y = 0.7
    pred = mp.insert(x, y)
    
    # Get metrics
    metrics = mp.get_metrics_dict()
    
    # Check required fields are present
    required_fields = [
        "regret_increment",
        "cum_regret", 
        "comparator_type",
        "oracle_refresh_step",
        "oracle_refreshes",
        "oracle_objective"
    ]
    
    for field in required_fields:
        assert field in metrics, f"Required field {field} missing from metrics"
        assert metrics[field] is not None, f"Field {field} should not be None"
        # Only check finite-ness for numeric fields
        if field not in ["comparator_type"]:
            assert np.isfinite(float(metrics[field])), f"Field {field} should be finite"
    
    # Check specific values
    assert metrics["comparator_type"] == "static_oracle_erm_fullprefix", \
        "Comparator type should be theory-faithful name"
    
    print("✓ Schema invariants test passed")


def test_oracle_objective_calculation():
    """Test that oracle_objective equals comparator's regularized loss."""
    lambda_reg = 0.1
    
    # Create MemoryPair with oracle
    mp = MemoryPair(dim=2)
    mp.lambda_reg = lambda_reg
    mp.phase = Phase.LEARNING
    
    # Create and attach oracle
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    mp.oracle = oracle
    
    # Perform an insert
    x = np.array([0.5, 0.5])
    y = 0.7
    pred = mp.insert(x, y)
    
    # Get metrics
    metrics = mp.get_metrics_dict()
    
    # Manually compute expected oracle_objective
    oracle_params = oracle.get_current_oracle()
    pred_comp = float(oracle_params @ x)
    expected_oracle_objective = (
        loss_half_mse(pred_comp, y) + 
        0.5 * lambda_reg * float(oracle_params @ oracle_params)
    )
    
    # Check that oracle_objective matches
    assert abs(metrics["oracle_objective"] - expected_oracle_objective) < 1e-12, \
        "oracle_objective should equal comparator's regularized loss"
    
    # Verify reconstruction: loss_curr_reg = regret_increment + oracle_objective
    pred_curr = float(mp.theta @ x)
    expected_loss_curr = (
        loss_half_mse(pred_curr, y) + 
        0.5 * lambda_reg * float(mp.theta @ mp.theta)
    )
    
    reconstructed_loss_curr = metrics["regret_increment"] + metrics["oracle_objective"]
    assert abs(reconstructed_loss_curr - expected_loss_curr) < 1e-12, \
        "Should be able to reconstruct learner loss from regret + oracle_objective"
    
    print("✓ Oracle objective calculation test passed")


def test_zero_proxy_oracle_objective():
    """Test oracle_objective calculation for zero proxy case."""
    lambda_reg = 0.1
    
    # Create MemoryPair without oracle (zero proxy)
    mp = MemoryPair(dim=2)
    mp.lambda_reg = lambda_reg
    mp.phase = Phase.LEARNING
    mp.oracle = None  # No oracle - zero proxy
    
    # Perform an insert
    x = np.array([0.5, 0.5])
    y = 0.7
    pred = mp.insert(x, y)
    
    # Get metrics
    metrics = mp.get_metrics_dict()
    
    # Check that comparator_type is zero_proxy
    assert metrics["comparator_type"] == "zero_proxy", \
        "Should be labeled as zero_proxy when oracle disabled"
    
    # Check that oracle_objective equals zero prediction loss (no regularization for zero weights)
    expected_oracle_objective = loss_half_mse(0.0, y)
    assert abs(metrics["oracle_objective"] - expected_oracle_objective) < 1e-12, \
        "oracle_objective should equal zero prediction loss for zero proxy"
    
    # Verify reconstruction for regularized learner loss
    pred_curr = float(mp.theta @ x)
    expected_loss_curr = (
        loss_half_mse(pred_curr, y) + 
        0.5 * lambda_reg * float(mp.theta @ mp.theta)
    )
    
    reconstructed_loss_curr = metrics["regret_increment"] + metrics["oracle_objective"]
    assert abs(reconstructed_loss_curr - expected_loss_curr) < 1e-12, \
        "Should be able to reconstruct regularized learner loss even in zero proxy case"
    
    print("✓ Zero proxy oracle objective test passed")


def test_metrics_comprehensive_coverage():
    """Test that metrics include all expected fields."""
    lambda_reg = 0.1
    
    # Create MemoryPair with oracle
    mp = MemoryPair(dim=2)
    mp.lambda_reg = lambda_reg
    mp.phase = Phase.LEARNING
    
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
    oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
    mp.oracle = oracle
    
    # Perform an insert
    x = np.array([0.5, 0.5])
    y = 0.7
    pred = mp.insert(x, y)
    
    # Get metrics
    metrics = mp.get_metrics_dict()
    
    # Check core fields
    expected_fields = [
        # Core regret metrics
        "regret_increment", "cum_regret", "static_regret_increment", 
        "path_regret_increment", "noise_regret_increment", "noise_regret_cum",
        "cum_regret_with_noise",
        
        # Model state
        "events_seen", "inserts_seen", "deletes_seen", "phase", 
        "ready_to_predict", "N_star", "N_gamma",
        
        # Regularization
        "lambda_reg", "lambda_raw",
        
        # Comparator/oracle specific
        "comparator_type", "oracle_refresh_step", "oracle_refreshes",
        "oracle_objective", "regret_static", "is_calibrated",
    ]
    
    for field in expected_fields:
        assert field in metrics, f"Expected field {field} missing from metrics"
    
    print("✓ Comprehensive metrics coverage test passed")


if __name__ == "__main__":
    print("Running logging and schema tests...")
    
    test_schema_invariants()
    test_oracle_objective_calculation()
    test_zero_proxy_oracle_objective()
    test_metrics_comprehensive_coverage()
    
    print("\n🎉 All logging and schema tests passed!")