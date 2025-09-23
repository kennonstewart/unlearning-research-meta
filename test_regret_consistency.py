"""
Test regret calculation consistency and theory-faithful comparator implementation.
"""

import sys
import os
sys.path.append('./code')

import numpy as np
import pytest
from code.memory_pair.src.memory_pair import MemoryPair, Phase
from code.memory_pair.src.comparators import StaticOracle, RollingOracle, OracleState
from code.memory_pair.src.metrics import loss_half_mse


class TestRegretConsistency:
    """Test suite for regret calculation consistency requirements."""

    def test_loss_parity_unit(self):
        """Test loss parity: regret_increment == loss_curr_reg - loss_comp_reg"""
        lambda_reg = 0.1
        
        # Test data
        x = np.array([1.0, 2.0])
        y = 1.5
        theta_curr = np.array([0.5, 0.3])
        theta_comp = np.array([0.4, 0.2])
        
        # Create MemoryPair instance to test _regret_terms
        mp = MemoryPair(dim=2)
        mp.lambda_reg = lambda_reg
        
        # Use the new _regret_terms method
        regret_terms = mp._regret_terms(x, y, theta_curr, theta_comp, lambda_reg)
        
        # Manual computation for verification
        pred_curr = float(theta_curr @ x)
        pred_comp = float(theta_comp @ x)
        
        loss_curr_expected = loss_half_mse(pred_curr, y) + 0.5 * lambda_reg * float(theta_curr @ theta_curr)
        loss_comp_expected = loss_half_mse(pred_comp, y) + 0.5 * lambda_reg * float(theta_comp @ theta_comp)
        regret_expected = loss_curr_expected - loss_comp_expected
        
        # Assert loss parity
        assert abs(regret_terms["regret_inc"] - regret_expected) < 1e-10, \
            "regret_increment should equal loss_curr_reg - loss_comp_reg"
        assert abs(regret_terms["loss_curr_reg"] - loss_curr_expected) < 1e-10, \
            "loss_curr_reg should match manual computation"
        assert abs(regret_terms["loss_comp_reg"] - loss_comp_expected) < 1e-10, \
            "loss_comp_reg should match manual computation"
        
        # Verify reconstruction: loss_curr_reg == regret_increment + oracle_objective
        reconstructed_curr = regret_terms["regret_inc"] + regret_terms["loss_comp_reg"]
        assert abs(reconstructed_curr - regret_terms["loss_curr_reg"]) < 1e-10, \
            "Should be able to reconstruct learner loss from regret + oracle loss"

    def test_static_oracle_consistency(self):
        """Test that StaticOracle uses consistent regularized loss calculation"""
        lambda_reg = 0.2
        
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        
        # Calibrate oracle
        calibration_data = [
            (np.array([1.0, 0.0]), 1.0),
            (np.array([0.0, 1.0]), 0.5),
        ]
        oracle.calibrate_with_initial_data(calibration_data)
        
        # Test point
        x = np.array([0.5, 0.5])
        y = 0.7
        current_theta = np.array([0.8, 0.3])
        
        # Get regret from oracle
        result = oracle.update_regret_accounting(x, y, current_theta)
        
        # Manually compute using same methodology
        regret_terms = oracle._regret_terms(x, y, current_theta, oracle.w_star_fixed, lambda_reg)
        
        # Should match exactly
        assert abs(result["regret_increment"] - regret_terms["regret_inc"]) < 1e-12, \
            "StaticOracle should use consistent _regret_terms calculation"

    def test_rolling_oracle_consistency(self):
        """Test that RollingOracle uses consistent regularized loss calculation"""
        lambda_reg = 0.15
        
        rolling = RollingOracle(dim=2, lambda_reg=lambda_reg, window_W=5)
        rolling.w_star_first = np.array([0.5, 0.2])
        rolling.oracle_state = OracleState(
            w_star=np.array([0.6, 0.25]), 
            last_refresh_step=0, 
            last_objective=0.0, 
            window_size=5
        )
        
        # Test point
        x = np.array([0.5, 0.5])
        y = 0.7
        current_theta = np.array([0.8, 0.3])
        
        # Get regret from oracle
        result = rolling.update_regret_accounting(x, y, current_theta)
        
        # Manually compute using same methodology
        regret_terms = rolling._regret_terms(x, y, current_theta, rolling.oracle_state.w_star, lambda_reg)
        static_terms = rolling._regret_terms(x, y, current_theta, rolling.w_star_first, lambda_reg)
        
        # Should match exactly
        assert abs(result["regret_increment"] - regret_terms["regret_inc"]) < 1e-12, \
            "RollingOracle dynamic regret should use consistent _regret_terms calculation"
        assert abs(result["static_increment"] - static_terms["regret_inc"]) < 1e-12, \
            "RollingOracle static regret should use consistent _regret_terms calculation"

    def test_fallback_regret_consistency(self):
        """Test that fallback regret (oracle disabled) uses consistent regularized loss"""
        lambda_reg = 0.1
        
        # Create MemoryPair without oracle
        mp = MemoryPair(dim=2)
        mp.lambda_reg = lambda_reg
        mp.phase = Phase.LEARNING
        mp.oracle = None  # No oracle - should use fallback
        
        x = np.array([0.5, 0.5])
        y = 0.7
        
        # Insert to trigger regret calculation
        pred = mp.insert(x, y)
        
        # Manually compute expected fallback regret
        theta_comp = np.zeros(mp.theta.shape)  # Zero baseline
        regret_terms = mp._regret_terms(x, y, mp.theta, theta_comp, lambda_reg)
        
        # Should match
        assert abs(mp.regret_increment - regret_terms["regret_inc"]) < 1e-12, \
            "Fallback regret should use consistent _regret_terms with zero comparator"

    def test_lambda_consistency(self):
        """Test that same λ is used for both learner and comparator"""
        lambda_reg = 0.25
        
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
        
        x = np.array([0.3, 0.7])
        y = 0.9
        current_theta = np.array([0.6, 0.4])
        
        # Get regret calculation
        result = oracle.update_regret_accounting(x, y, current_theta)
        
        # Check that oracle's lambda_reg is being used consistently
        regret_terms = oracle._regret_terms(x, y, current_theta, oracle.w_star_fixed, oracle.lambda_reg)
        
        assert abs(result["regret_increment"] - regret_terms["regret_inc"]) < 1e-12, \
            "Oracle should use its own lambda_reg consistently"
        
        # Verify regularization is actually being applied (not zero)
        regret_terms_no_reg = oracle._regret_terms(x, y, current_theta, oracle.w_star_fixed, 0.0)
        assert abs(regret_terms["regret_inc"] - regret_terms_no_reg["regret_inc"]) > 1e-6, \
            "Regularization should make a measurable difference in regret calculation"

    def test_fallback_zero_proxy(self):
        """Test that zero proxy fallback uses consistent regularized loss and correct labeling"""
        lambda_reg = 0.1
        
        # Create MemoryPair without oracle (fallback case)
        mp = MemoryPair(dim=2)
        mp.lambda_reg = lambda_reg
        mp.phase = Phase.LEARNING
        mp.oracle = None  # No oracle - should use fallback
        
        x = np.array([0.5, 0.5])
        y = 0.7
        
        # Insert to trigger regret calculation
        pred = mp.insert(x, y)
        
        # Check comparator metrics
        metrics = mp.get_comparator_metrics()
        assert metrics["comparator_type"] == "zero_proxy", \
            "Fallback should be labeled as zero_proxy"
        
        # Manually verify the regret calculation uses same regularized objective
        theta_comp = np.zeros(mp.theta.shape)  # Zero baseline
        regret_terms = mp._regret_terms(x, y, mp.theta, theta_comp, lambda_reg)
        
        assert abs(mp.regret_increment - regret_terms["regret_inc"]) < 1e-12, \
            "Fallback regret should use consistent _regret_terms with zero comparator"


if __name__ == "__main__":
    # Run tests
    test_suite = TestRegretConsistency()
    
    print("Running regret consistency tests...")
    
    test_suite.test_loss_parity_unit()
    print("✓ Loss parity test passed")
    
    test_suite.test_static_oracle_consistency()
    print("✓ Static oracle consistency test passed")
    
    test_suite.test_rolling_oracle_consistency()
    print("✓ Rolling oracle consistency test passed")
    
    test_suite.test_fallback_regret_consistency()
    print("✓ Fallback regret consistency test passed")
    
    test_suite.test_lambda_consistency()
    print("✓ Lambda consistency test passed")
    
    test_suite.test_fallback_zero_proxy()
    print("✓ Fallback zero proxy test passed")
    
    print("\n🎉 All regret consistency tests passed!")