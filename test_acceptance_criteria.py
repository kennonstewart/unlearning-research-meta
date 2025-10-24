"""
Comprehensive acceptance tests for regret calculation consistency and theory-faithful comparator.
"""

import sys
import os
sys.path.append('./code')

import numpy as np
from code.memory_pair.src.memory_pair import MemoryPair
from code.memory_pair.src.comparators import StaticOracle
from code.memory_pair.src.metrics import loss_half_mse


class TestAcceptanceCriteria:
    """Test suite for all acceptance criteria from the requirements."""

    def test_acceptance_1_loss_parity(self):
        """
        Acceptance Test 1: Loss parity test (unit)
        For a tiny synthetic dataset, compute one event by hand. Assert:
        - regret_increment == loss_curr_reg - loss_comp_reg
        - oracle_objective == loss_comp_reg  
        - Reconstructed loss_curr_reg == regret_increment + oracle_objective (to 1e-10)
        """
        lambda_reg = 0.1
        
        # Create MemoryPair with oracle - provide default constants
        mp = MemoryPair(dim=2, G=1.0, D=1.0, c=1.0, C=1.0)
        mp.lambda_reg = lambda_reg
        
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
        mp.oracle = oracle
        
        # Test event
        x = np.array([0.5, 0.5])
        y = 0.7
        pred = mp.insert(x, y)
        
        # Get metrics and manual computation
        metrics = mp.get_metrics_dict()
        
        # Manual computation of losses
        pred_curr = float(mp.theta @ x)
        pred_comp = float(oracle.w_star_fixed @ x)
        
        loss_curr_reg = loss_half_mse(pred_curr, y) + 0.5 * lambda_reg * float(mp.theta @ mp.theta)
        loss_comp_reg = loss_half_mse(pred_comp, y) + 0.5 * lambda_reg * float(oracle.w_star_fixed @ oracle.w_star_fixed)
        
        # Test assertions
        assert abs(metrics["regret_increment"] - (loss_curr_reg - loss_comp_reg)) < 1e-10, \
            "regret_increment should equal loss_curr_reg - loss_comp_reg"
        assert abs(metrics["oracle_objective"] - loss_comp_reg) < 1e-10, \
            "oracle_objective should equal loss_comp_reg"
        
        reconstructed_curr = metrics["regret_increment"] + metrics["oracle_objective"]
        assert abs(reconstructed_curr - loss_curr_reg) < 1e-10, \
            "Reconstructed loss_curr_reg should equal regret_increment + oracle_objective"
        
        print("✓ Acceptance Test 1: Loss parity test passed")

    def test_acceptance_2_static_erm_correctness(self):
        """
        Acceptance Test 2: Static ERM correctness (unit)
        With no drift and no deletes: StaticOracleERM equals closed-form ridge solution 
        on the full prefix at every refresh.
        """
        lambda_reg = 0.2
        
        # Create oracle with refresh period 1 (refresh every event)
        class Config:
            oracle_refresh_period = 1
            
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=Config())
        oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
        
        # Accumulate data and check ERM solution at each step
        data = [
            (np.array([1.0, 0.0]), 1.0),  # Initial calibration data
            (np.array([0.0, 1.0]), 0.5),
            (np.array([0.5, 0.5]), 0.75),
            (np.array([0.3, 0.7]), 0.6),
        ]
        
        current_theta = np.array([0.4, 0.3])
        accumulated_data = [data[0]]  # Start with calibration data
        
        for i in range(1, len(data)):
            x, y = data[i]
            accumulated_data.append((x, y))
            
            # Update oracle
            oracle.update_regret_accounting(x, y, current_theta)
            
            # Compute expected closed-form solution
            X = np.array([d[0] for d in accumulated_data])
            Y = np.array([d[1] for d in accumulated_data])
            XtX = X.T @ X
            XtY = X.T @ Y
            A = XtX + lambda_reg * np.eye(2)
            w_star_expected = np.linalg.solve(A, XtY)
            
            # Compare
            assert np.allclose(oracle.w_star_fixed, w_star_expected, atol=1e-10), \
                f"Oracle solution should match closed-form at step {i}"
        
        print("✓ Acceptance Test 2: Static ERM correctness test passed")

    def test_acceptance_3_delete_correctness(self):
        """
        Acceptance Test 3: Delete correctness (integration)
        After a delete event that removes sample i:
        - Next refresh yields θ* equal to ridge ERM on the remaining samples only
        - comparator_type='static_oracle_erm_fullprefix' and oracle_refresh_step increments on delete
        """
        lambda_reg = 0.1
        
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        
        # Initial data
        data = [
            (np.array([1.0, 0.0]), 1.0),
            (np.array([0.0, 1.0]), 0.5),
            (np.array([0.5, 0.5]), 0.75),
        ]
        oracle.calibrate_with_initial_data(data)
        
        current_theta = np.array([0.4, 0.3])
        
        # Add a sample
        x_add = np.array([0.3, 0.7])
        y_add = 0.6
        oracle.update_regret_accounting(x_add, y_add, current_theta)
        
        initial_refreshes = oracle.oracle_refreshes
        
        # Delete the added sample
        success = oracle.process_delete_event(x_add, y_add)
        assert success, "Delete should succeed"
        
        # Check that refresh happened on delete
        assert oracle.oracle_refreshes > initial_refreshes, \
            "Oracle should refresh on delete"
        
        # Check that oracle matches expected solution with remaining data
        expected_oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        expected_oracle.calibrate_with_initial_data(data)  # Only original data
        
        assert np.allclose(oracle.w_star_fixed, expected_oracle.w_star_fixed, atol=1e-10), \
            "Oracle after delete should match ERM on remaining samples"
        
        # Check comparator type
        metrics = oracle.get_oracle_metrics()
        assert metrics["comparator_type"] == "static_oracle_erm_fullprefix", \
            "Comparator type should be theory-faithful name"
        
        print("✓ Acceptance Test 3: Delete correctness test passed")

    def test_acceptance_4_fallback_parity(self):
        """
        Acceptance Test 4: Parity under fallback (unit)
        With enable_oracle=False (zero proxy):
        - Regret uses the same L_λ on both sides
        - comparator_type='zero_proxy', oracle_objective equals comparator loss with θ*=0
        """
        lambda_reg = 0.15
        
        # Create MemoryPair without oracle - provide default constants
        mp = MemoryPair(dim=2, G=1.0, D=1.0, c=1.0, C=1.0)
        mp.lambda_reg = lambda_reg
        mp.oracle = None  # Zero proxy
        
        # Perform insert
        x = np.array([0.4, 0.6])
        y = 0.8
        pred = mp.insert(x, y)
        
        # Get metrics
        metrics = mp.get_metrics_dict()
        
        # Check comparator type
        assert metrics["comparator_type"] == "zero_proxy", \
            "Should be labeled as zero_proxy"
        
        # Check that regret uses same L_λ on both sides
        theta_comp = np.zeros(mp.theta.shape)
        regret_terms = mp._regret_terms(x, y, mp.theta, theta_comp, lambda_reg)
        
        assert abs(metrics["regret_increment"] - regret_terms["regret_inc"]) < 1e-12, \
            "Fallback should use same L_λ via _regret_terms"
        
        # Check oracle_objective equals zero predictor loss
        expected_oracle_objective = loss_half_mse(0.0, y)  # No regularization for zero weights
        assert abs(metrics["oracle_objective"] - expected_oracle_objective) < 1e-12, \
            "oracle_objective should equal zero predictor loss"
        
        print("✓ Acceptance Test 4: Fallback parity test passed")

    def test_acceptance_5_schema_invariants(self):
        """
        Acceptance Test 5: Schema invariants (integration)
        - No new columns appear in analytics.fact_event (we check comprehensive metrics)
        - oracle_objective populated on all insert events; regret_increment present and finite
        - cum_regret_with_noise - cum_regret == noise_regret_cum ± tolerance
        """
        lambda_reg = 0.1
        
        # Test with oracle enabled - provide default constants
        mp = MemoryPair(dim=2, G=1.0, D=1.0, c=1.0, C=1.0)
        mp.lambda_reg = lambda_reg
        
        oracle = StaticOracle(dim=2, lambda_reg=lambda_reg)
        oracle.calibrate_with_initial_data([(np.array([1.0, 0.0]), 1.0)])
        mp.oracle = oracle
        
        # Perform insert
        x = np.array([0.3, 0.7])
        y = 0.9
        pred = mp.insert(x, y)
        
        # Get metrics (simulates analytics.fact_event row)
        metrics = mp.get_metrics_dict()
        
        # Check required fields are present and valid
        assert "oracle_objective" in metrics, "oracle_objective should be present"
        assert "regret_increment" in metrics, "regret_increment should be present"
        assert np.isfinite(metrics["oracle_objective"]), "oracle_objective should be finite"
        assert np.isfinite(metrics["regret_increment"]), "regret_increment should be finite"
        
        # Check noise accounting invariant
        cum_regret = metrics["cum_regret"]
        noise_regret_cum = metrics["noise_regret_cum"]
        cum_regret_with_noise = metrics["cum_regret_with_noise"]
        
        difference = abs(cum_regret_with_noise - cum_regret - noise_regret_cum)
        assert difference < 1e-10, \
            "cum_regret_with_noise - cum_regret should equal noise_regret_cum"
        
        print("✓ Acceptance Test 5: Schema invariants test passed")


def run_all_acceptance_tests():
    """Run all acceptance tests from the requirements."""
    print("Running all acceptance tests from the requirements...\n")
    
    test_suite = TestAcceptanceCriteria()
    
    test_suite.test_acceptance_1_loss_parity()
    test_suite.test_acceptance_2_static_erm_correctness()
    test_suite.test_acceptance_3_delete_correctness()
    test_suite.test_acceptance_4_fallback_parity()
    test_suite.test_acceptance_5_schema_invariants()
    
    print("\n🎉 All acceptance tests passed!")
    print("✅ Regret calculation consistency and theory-faithful comparator implementation complete!")


if __name__ == "__main__":
    run_all_acceptance_tests()