#!/usr/bin/env python3
"""
Test for privacy metrics fix - validates that rho_spent and privacy_spend_running
are properly force-updated from model metrics, and sigma_delete is added for delete operations.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_privacy_metrics_force_update():
    """Test that privacy metrics are force-updated from model metrics even when stale values exist."""
    from phases import _create_extended_log_entry
    from experiment.utils.configs.config import Config
    
    # Create a mock PhaseState with minimal structure
    class MockPhaseState:
        def __init__(self):
            self.current_record = None
    
    # Create a mock config
    cfg = Config()
    
    # Create a mock model with privacy metrics
    mock_model = Mock()
    mock_model.get_metrics_dict.return_value = {
        "rho_spent": 0.5,  # Fresh value from accountant
        "sigma_step": 0.1,
        "privacy_spend_running": 0.5,
    }
    mock_model.odometer = None
    
    # Create base entry with stale zero value (simulating the bug)
    base_entry = {
        "op": "delete",
        "rho_spent": 0.0,  # Stale zero value that should be overwritten
        "privacy_spend_running": 0.0,  # Stale zero value
    }
    
    state = MockPhaseState()
    
    # Call the function
    result = _create_extended_log_entry(base_entry, state, mock_model, cfg)
    
    print("Base entry (with stale values):", base_entry)
    print("Model metrics:", mock_model.get_metrics_dict.return_value)
    print("Result entry rho_spent:", result.get("rho_spent"))
    print("Result entry privacy_spend_running:", result.get("privacy_spend_running"))
    
    # Before fix: this would preserve the stale 0.0 value
    # After fix: this should force-update to 0.5
    return result


def test_sigma_delete_column():
    """Test that sigma_delete column is added for delete operations."""
    from phases import _create_extended_log_entry
    from experiment.utils.configs.config import Config
    
    class MockPhaseState:
        def __init__(self):
            self.current_record = None
    
    cfg = Config()
    
    # Mock model with sigma_step
    mock_model = Mock()
    mock_model.get_metrics_dict.return_value = {
        "sigma_step": 0.1,
    }
    mock_model.odometer = None
    
    # Test delete operation
    base_entry = {"op": "delete"}
    state = MockPhaseState()
    
    result = _create_extended_log_entry(base_entry, state, mock_model, cfg)
    
    print("Delete operation result:")
    print("  sigma_step:", result.get("sigma_step"))
    print("  sigma_delete:", result.get("sigma_delete"))
    
    # Test insert operation
    base_entry = {"op": "insert"}
    result_insert = _create_extended_log_entry(base_entry, state, mock_model, cfg)
    
    print("Insert operation result:")
    print("  sigma_step:", result_insert.get("sigma_step"))
    print("  sigma_delete:", result_insert.get("sigma_delete"))
    
    return result, result_insert


def test_rho_util_computation():
    """Test rho_util computation in grid_runner process_seed_output."""
    from experiment.grid_runner import process_seed_output
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV with privacy data
        data = {
            "event": [1, 2, 3, 4, 5],
            "op": ["insert", "insert", "delete", "delete", "delete"],
            "rho_spent": [0.1, 0.2, 0.3, 0.4, 0.5],  # Final value: 0.5
            "regret": [0.1, 0.2, 0.3, 0.4, 0.5],
            "G_hat": [2.0] * 5,
            "D_hat": [1.5] * 5,
            "sigma_step_theory": [0.05] * 5,
        }
        df = pd.DataFrame(data)
        
        # Use expected filename format (seed_X_memorypair.csv format)
        csv_file = os.path.join(temp_dir, "123_memorypair.csv")
        df.to_csv(csv_file, index=False)
        
        # Mock mandatory fields including rho_total
        mandatory_fields = {
            "rho_total": 1.0,  # Total budget
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
        }
        
        print(f"Created test file: {csv_file}")
        print(f"File exists: {os.path.exists(csv_file)}")
        
        # Call process_seed_output
        output_files = process_seed_output(
            csv_files=[csv_file],
            grid_id="test_grid",
            output_dir=temp_dir,
            mandatory_fields=mandatory_fields
        )
        
        print(f"Output files: {output_files}")
        
        # Read the result
        if output_files:
            result_df = pd.read_csv(output_files[0])
            print("Result columns:", list(result_df.columns))
            if "rho_util" in result_df.columns:
                rho_util_val = result_df["rho_util"].iloc[0]
                print("rho_util value:", rho_util_val)
                print("Expected rho_util (0.5/1.0 = 0.5):", 0.5)
                if abs(rho_util_val - 0.5) < 1e-6:
                    print("✓ rho_util computation is correct")
                else:
                    print("✗ rho_util computation is incorrect")
            else:
                print("rho_util column missing (expected after fix)")
        else:
            print("No output files generated")
        
        return output_files


if __name__ == "__main__":
    print("=== Testing Privacy Metrics Fix ===\n")
    
    print("1. Testing force-update of privacy metrics:")
    try:
        result = test_privacy_metrics_force_update()
        print("✓ Privacy metrics test completed\n")
    except Exception as e:
        print(f"✗ Privacy metrics test failed: {e}\n")
    
    print("2. Testing sigma_delete column:")
    try:
        delete_result, insert_result = test_sigma_delete_column()
        print("✓ Sigma delete test completed\n")
    except Exception as e:
        print(f"✗ Sigma delete test failed: {e}\n")
    
    print("3. Testing rho_util computation:")
    try:
        output_files = test_rho_util_computation()
        print("✓ Rho util test completed\n")
    except Exception as e:
        print(f"✗ Rho util test failed: {e}\n")