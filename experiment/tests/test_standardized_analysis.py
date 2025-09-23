#!/usr/bin/env python3
"""
Test the standardized analysis functions for experiment notebooks.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.standardized_analysis import (
    compute_theory_bound_tracking,
    validate_stepsize_policy,
    check_privacy_odometer_sanity,
    audit_seed_stability,
    enhance_claim_check_export,
    run_all_standardized_analyses
)


def test_theory_bound_tracking():
    """Test theory bound tracking with mock data."""
    # Mock DuckDB connection
    class MockConnection:
        def execute(self, query, params=None):
            # Return mock DataFrame
            return MockResult({
                'event_id': [1, 2, 3, 4, 5],
                'cum_regret': [0.1, 0.2, 0.3, 0.4, 0.5],
                'P_T_true': [0.05, 0.1, 0.15, 0.2, 0.25],
                'lambda_est': [0.01, 0.01, 0.01, 0.01, 0.01],
                'G_hat': [1.0, 1.0, 1.0, 1.0, 1.0]
            })
    
    class MockResult:
        def __init__(self, data):
            self.data = data
        def df(self):
            return pd.DataFrame(self.data)
    
    con = MockConnection()
    runs_df = pd.DataFrame({'grid_id': ['test_grid'], 'seed': [1]})
    
    results = compute_theory_bound_tracking(con, runs_df)
    
    assert 'test_grid_1' in results
    assert results['test_grid_1']['status'] == 'success'
    assert results['test_grid_1']['theory_ratio_final'] is not None
    print("Theory bound tracking test passed!")


def test_stepsize_policy_validation():
    """Test stepsize policy validation with mock data."""
    class MockConnection:
        def execute(self, query, params=None):
            return MockResult({
                'event_id': [1, 2, 3, 4, 5],
                'eta_t': [0.1, 0.05, 0.033, 0.025, 0.02],
                'base_eta_t': [0.1, 0.1, 0.1, 0.1, 0.1],
                'sc_active': [False, False, False, False, False],
                'lambda_est': [0.01, 0.01, 0.01, 0.01, 0.01],
                'ST_running': [1.0, 4.0, 9.0, 16.0, 25.0],
                'D_bound': [1.0, 1.0, 1.0, 1.0, 1.0]
            })
    
    class MockResult:
        def __init__(self, data):
            self.data = data
        def df(self):
            return pd.DataFrame(self.data)
    
    con = MockConnection()
    runs_df = pd.DataFrame({'grid_id': ['test_grid'], 'seed': [1]})
    
    results = validate_stepsize_policy(con, runs_df)
    
    assert 'test_grid_1' in results
    assert 'stepsize_policy_mape' in results['test_grid_1']
    print("Stepsize policy validation test passed!")


def test_privacy_odometer_sanity():
    """Test privacy and odometer sanity checks with mock data."""
    class MockConnection:
        def execute(self, query, params=None):
            return MockResult({
                'm_used': [5],
                'm_capacity': [10],
                'rho_spent': [0.8],
                'rho_total': [1.0],
                'sigma_step': [0.1],
                'cum_regret': [0.5],
                'cum_regret_with_noise': [0.52],
                'noise_regret_cum': [0.02]
            })
    
    class MockResult:
        def __init__(self, data):
            self.data = data
        def df(self):
            return pd.DataFrame(self.data)
    
    con = MockConnection()
    runs_df = pd.DataFrame({'grid_id': ['test_grid'], 'seed': [1]})
    
    results = check_privacy_odometer_sanity(con, runs_df)
    
    assert 'test_grid_1' in results
    assert results['test_grid_1']['privacy_odometer_status'] == 'pass'
    print("Privacy odometer sanity test passed!")


def test_seed_stability_audit():
    """Test seed stability audit with mock data."""
    class MockConnection:
        def execute(self, query, params=None):
            return MockResult({
                'seed': [1, 2, 3],
                'final_cum_regret': [0.5, 0.6, 0.55],
                'final_P_T_true': [0.25, 0.3, 0.28]
            })
    
    class MockResult:
        def __init__(self, data):
            self.data = data
        def df(self):
            return pd.DataFrame(self.data)
    
    con = MockConnection()
    
    results = audit_seed_stability(con, ['test_grid'])
    
    assert 'test_grid' in results
    assert results['test_grid']['status'] == 'success'
    assert 'cum_regret_stats' in results['test_grid']
    print("Seed stability audit test passed!")


def test_enhance_claim_check_export():
    """Test enhanced claim check export."""
    base_summary = [
        {'grid_id': 'test_grid', 'seed': 1, 'final_cum_regret': 0.5}
    ]
    
    theory_results = {
        'test_grid_1': {
            'theory_ratio_median_tail': 1.2,
            'theory_ratio_final': 1.1,
            'status': 'success'
        }
    }
    
    stepsize_results = {
        'test_grid_1': {
            'stepsize_policy_mape': 15.0,
            'stepsize_policy_status': 'pass',
            'stepsize_policy_type': 'adagrad'
        }
    }
    
    privacy_results = {
        'test_grid_1': {
            'privacy_odometer_status': 'pass',
            'checks': {}
        }
    }
    
    stability_results = {
        'test_grid': {
            'status': 'success',
            'high_variability_flag': False
        }
    }
    
    enhanced = enhance_claim_check_export(
        base_summary, theory_results, stepsize_results, 
        privacy_results, stability_results
    )
    
    assert len(enhanced) == 1
    assert 'theory_ratio_final' in enhanced[0]
    assert 'stepsize_policy_mape' in enhanced[0]
    assert 'privacy_odometer_status' in enhanced[0]
    print("Enhanced claim check export test passed!")


if __name__ == "__main__":
    test_theory_bound_tracking()
    test_stepsize_policy_validation()
    test_privacy_odometer_sanity()
    test_seed_stability_audit()
    test_enhance_claim_check_export()
    print("All standardized analysis tests passed!")