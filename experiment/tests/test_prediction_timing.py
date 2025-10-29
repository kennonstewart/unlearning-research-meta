#!/usr/bin/env python3
"""
Test that predictions are computed before model updates in online learning.

This test validates the fix for the prediction timing error where predictions
were incorrectly computed after model updates instead of before.
"""

import os
import sys
import tempfile
import numpy as np

# Add parent directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CODE_DIR = os.path.join(_REPO_ROOT, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from experiment.config import ExperimentConfig
from memory_pair.src.memory_pair import MemoryPair


def test_insert_prediction_before_update():
    """
    Test that model.insert() returns the prediction computed BEFORE the model update.
    
    This test:
    1. Creates a model with known parameters
    2. Records the prediction that would be made with current parameters
    3. Calls model.insert() and verifies the returned prediction matches the pre-update prediction
    4. Verifies that the model parameters have changed after the insert
    """
    # Create a simple model
    dim = 5
    model = MemoryPair(dim=dim, G=1.0, D=1.0, c=1.0, C=1.0)
    
    # Set initial parameters to something non-zero for easier testing
    model.theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    initial_theta = model.theta.copy()
    
    # Create a test data point
    x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    y = 1.5
    
    # Compute what the prediction SHOULD be before the update
    expected_prediction = float(initial_theta @ x)
    
    # Call insert and get the returned prediction
    returned_prediction = model.insert(x, y)
    
    # The returned prediction should match the pre-update prediction
    assert abs(returned_prediction - expected_prediction) < 1e-10, \
        f"Prediction should be computed before update. Expected {expected_prediction}, got {returned_prediction}"
    
    # Verify that the model parameters have changed
    assert not np.allclose(model.theta, initial_theta), \
        "Model parameters should have changed after insert"
    
    # Verify that a post-update prediction would be different
    post_update_prediction = float(model.theta @ x)
    assert abs(post_update_prediction - expected_prediction) > 1e-10, \
        "Post-update prediction should be different from pre-update prediction"
    
    print("✅ Insert prediction timing test passed")


def test_run_script_uses_correct_prediction():
    """
    Test that the run.py script executes without errors with the prediction fix.
    
    This is a smoke test that verifies the fixed code in run_single_experiment 
    executes correctly. The unit test above validates the actual prediction timing.
    """
    from experiment.run import run_single_experiment
    
    # Create a minimal parameter dict
    params = {
        "target_G": 1.0,
        "target_D": 1.0,
        "target_c": 1.0,
        "target_C": 1.0,
        "target_lambda": 0.5,
        "target_PT": 10.0,
        "target_ST": 100.0,
        "rho_total": 1.0,
        "max_events": 10,  # Just a few events for testing
        "dim": 5,
        "grid_id": "test_prediction_timing",
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run a single experiment - this will exercise the fixed code
        results = run_single_experiment(params, seed=42, output_dir=tmpdir)
        
        # Verify basic results
        assert results is not None, "Results should not be None"
        assert "seed" in results, "Results should contain seed"
        assert "events_processed" in results, "Results should contain events_processed"
        assert results["events_processed"] == 10, "Should have processed 10 events"
        
        print(f"✅ Integration test passed - processed {results['events_processed']} events")


if __name__ == "__main__":
    test_insert_prediction_before_update()
    test_run_script_uses_correct_prediction()
    print("\n✅ All prediction timing tests passed!")
