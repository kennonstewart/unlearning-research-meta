#!/usr/bin/env python3
"""
Test script to validate γ-adherence checking functionality.
Tests the new columns and pass/fail logic.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))

from experiment.grid_runner import process_seed_output

def create_mock_csv_with_regret(path: str, num_events: int = 10, avg_regret_final: float = 0.5):
    """Create a mock CSV file with specific regret pattern for testing γ-adherence"""
    events = []
    
    # Create events with target final average regret
    for i in range(num_events):
        op = "insert" if i < num_events // 2 else "delete"
        
        # Set regret to achieve desired final average
        regret = avg_regret_final + np.random.normal(0, 0.1)
        
        event = {
            'event': i + 1000,
            'op': op,
            'regret': regret,
            'avg_regret': avg_regret_final,  # Explicit avg_regret column
            'cum_regret': (i + 1) * avg_regret_final,  # Cumulative regret
            'regret_increment': regret,  # For decomposition tests
            'acc': 0.1 + i * 0.02,
            'eps_spent': i * 0.01,
            'capacity_remaining': 1.0 - (i * 0.01),
            'G_hat': 2.5,
            'D_hat': 1.8,
            'sigma_step_theory': 0.05
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    df.to_csv(path, index=False)
    return path

def test_gamma_adherence_pass():
    """Test γ-adherence checking when regret is below threshold (should pass)"""
    print("Testing γ-adherence pass scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with low regret (should pass γ=1.0 threshold)
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        create_mock_csv_with_regret(csv_path, num_events=10, avg_regret_final=0.8)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set gamma_bar = 1.0, so regret = 0.8 should pass
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path], "test_grid", output_dir, mandatory_fields)
        
        # Verify output file was created
        assert len(result) == 1, f"Expected 1 output file, got {len(result)}"
        output_file = result[0]
        assert os.path.exists(output_file), f"Output file {output_file} does not exist"
        
        # Read and validate results
        df = pd.read_csv(output_file)
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        row = df.iloc[0]
        
        # Check new γ-adherence columns exist
        required_cols = [
            'avg_regret_final', 'gamma_bar_threshold', 'gamma_split_threshold',
            'gamma_insert_threshold', 'gamma_delete_threshold', 'gamma_pass_overall',
            'gamma_pass_insert', 'gamma_pass_delete', 'gamma_error'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Validate values
        assert abs(row['avg_regret_final'] - 0.8) < 0.01, f"avg_regret_final should be ~0.8, got {row['avg_regret_final']}"
        assert row['gamma_bar_threshold'] == 1.0, f"gamma_bar_threshold should be 1.0, got {row['gamma_bar_threshold']}"
        assert row['gamma_split_threshold'] == 0.5, f"gamma_split_threshold should be 0.5, got {row['gamma_split_threshold']}"
        assert row['gamma_insert_threshold'] == 0.5, f"gamma_insert_threshold should be 0.5, got {row['gamma_insert_threshold']}"
        assert row['gamma_delete_threshold'] == 0.5, f"gamma_delete_threshold should be 0.5, got {row['gamma_delete_threshold']}"
        assert row['gamma_pass_overall'] == True, f"gamma_pass_overall should be True, got {row['gamma_pass_overall']}"
        assert row['gamma_error'] == 0.0, f"gamma_error should be 0.0, got {row['gamma_error']}"
        
        # Should not have AT-γ in blocked_reason
        blocked_reason = str(row['blocked_reason'])
        assert 'AT-γ' not in blocked_reason, f"Should not have AT-γ in blocked_reason for passing case, got: {blocked_reason}"
        
        print("✅ γ-adherence pass scenario validated")

def test_gamma_adherence_fail():
    """Test γ-adherence checking when regret exceeds threshold (should fail)"""
    print("Testing γ-adherence fail scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with high regret (should fail γ=1.0 threshold)
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        create_mock_csv_with_regret(csv_path, num_events=10, avg_regret_final=1.5)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set gamma_bar = 1.0, so regret = 1.5 should fail
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path], "test_grid", output_dir, mandatory_fields)
        
        # Read and validate results
        df = pd.read_csv(result[0])
        row = df.iloc[0]
        
        # Validate values for failing case
        assert abs(row['avg_regret_final'] - 1.5) < 0.01, f"avg_regret_final should be ~1.5, got {row['avg_regret_final']}"
        assert row['gamma_pass_overall'] == False, f"gamma_pass_overall should be False, got {row['gamma_pass_overall']}"
        assert row['gamma_error'] == 0.5, f"gamma_error should be 0.5 (1.5/1.0-1), got {row['gamma_error']}"
        
        # Should have AT-γ in blocked_reason
        blocked_reason = str(row['blocked_reason'])
        assert 'AT-γ' in blocked_reason, f"Should have AT-γ in blocked_reason for failing case, got: {blocked_reason}"
        assert '1.5' in blocked_reason and '1' in blocked_reason, f"Blocked reason should contain regret and threshold values, got: {blocked_reason}"
        
        print("✅ γ-adherence fail scenario validated")

def test_gamma_adherence_robust_computation():
    """Test robust computation of avg_regret_final with different column patterns"""
    print("Testing robust avg_regret_final computation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test case 1: No avg_regret column, fallback to regret mean
        csv_path1 = os.path.join(temp_dir, "1_memorypair.csv")
        events = []
        regret_values = [0.2, 0.4, 0.6, 0.8]  # mean = 0.5
        for i, regret in enumerate(regret_values):
            events.append({
                'event': i + 1000,
                'op': 'insert' if i < 2 else 'delete',
                'regret': regret,
                'acc': 0.1,
                'G_hat': 2.5,
                'D_hat': 1.8,
                'sigma_step_theory': 0.05
            })
        pd.DataFrame(events).to_csv(csv_path1, index=False)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path1], "test_grid", output_dir, mandatory_fields)
        df = pd.read_csv(result[0])
        row = df.iloc[0]
        
        # Should compute mean of regret column
        assert abs(row['avg_regret_final'] - 0.5) < 0.01, f"Should compute regret mean=0.5, got {row['avg_regret_final']}"
        
        print("✅ Robust computation scenarios validated")

def test_gamma_adherence_columns_always_present():
    """Test that γ-adherence columns are always present even with missing data"""
    print("Testing γ-adherence columns always present...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create minimal CSV with missing gamma fields
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        events = [{'event': 1000, 'op': 'insert', 'regret': 0.5, 'acc': 0.1}]
        pd.DataFrame(events).to_csv(csv_path, index=False)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Don't provide gamma fields
        mandatory_fields = {
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path], "test_grid", output_dir, mandatory_fields)
        df = pd.read_csv(result[0])
        
        # All γ columns should still exist
        required_cols = [
            'avg_regret_final', 'gamma_bar_threshold', 'gamma_split_threshold',
            'gamma_insert_threshold', 'gamma_delete_threshold', 'gamma_pass_overall',
            'gamma_pass_insert', 'gamma_pass_delete', 'gamma_error'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Values should be reasonable defaults (NaN for thresholds, False for pass)
        row = df.iloc[0]
        assert row['avg_regret_final'] == 0.5, f"Should still compute avg_regret_final"
        assert np.isnan(row['gamma_bar_threshold']), f"gamma_bar_threshold should be NaN when not provided"
        assert row['gamma_pass_overall'] == False, f"gamma_pass_overall should be False when thresholds missing"
        
        print("✅ Columns always present validation passed")

if __name__ == "__main__":
    print("Testing γ-adherence checking functionality...")
    try:
        test_gamma_adherence_pass()
        test_gamma_adherence_fail() 
        test_gamma_adherence_robust_computation()
        test_gamma_adherence_columns_always_present()
        print("\n✅ All γ-adherence tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()