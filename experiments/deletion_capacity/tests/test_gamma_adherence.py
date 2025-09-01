#!/usr/bin/env python3
"""
Test script to validate gamma-adherence (pass/fail) processing.
Tests the new γ-adherence check functionality in process_seed_output.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))

from agents.grid_runner import process_seed_output

def create_mock_csv_with_regret(path: str, avg_regret_value: float, num_events: int = 10):
    """Create a mock CSV file with specific average regret"""
    events = []
    for i in range(num_events):
        op = "insert" if i < num_events // 2 else "delete"
        # Set regret values to achieve the desired average
        regret = avg_regret_value + (i - num_events//2) * 0.1  # small variation around target
        event = {
            'event': i + 1000,
            'op': op,
            'regret': regret,
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

def test_gamma_pass():
    """Test gamma-adherence check when it should PASS"""
    print("Testing gamma-adherence PASS scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with low regret (should pass)
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        create_mock_csv_with_regret(csv_path, avg_regret_value=0.5, num_events=10)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set gamma_bar = 1.0, so avg_regret = 0.5 should pass
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path], "test_grid", output_dir, mandatory_fields)
        
        # Check results
        assert len(result) == 1, f"Expected 1 output file, got {len(result)}"
        df = pd.read_csv(result[0])
        
        # Verify gamma columns exist
        required_cols = [
            'avg_regret_final', 'gamma_bar_threshold', 'gamma_pass_overall',
            'gamma_error', 'gamma_insert_threshold', 'gamma_delete_threshold'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Verify values
        row = df.iloc[0]
        print(f"  avg_regret_final: {row['avg_regret_final']}")
        print(f"  gamma_bar_threshold: {row['gamma_bar_threshold']}")
        print(f"  gamma_pass_overall: {row['gamma_pass_overall']}")
        print(f"  gamma_error: {row['gamma_error']}")
        print(f"  blocked_reason: '{row['blocked_reason']}'")
        
        # Assertions for PASS scenario
        assert abs(row['avg_regret_final'] - 0.5) < 0.1, f"Expected avg_regret_final ~0.5, got {row['avg_regret_final']}"
        assert row['gamma_bar_threshold'] == 1.0, f"Expected gamma_bar_threshold=1.0, got {row['gamma_bar_threshold']}"
        assert row['gamma_pass_overall'] == True, f"Expected gamma_pass_overall=True, got {row['gamma_pass_overall']}"
        assert row['gamma_error'] == 0.0, f"Expected gamma_error=0.0 for passing case, got {row['gamma_error']}"
        assert "AT-γ" not in str(row['blocked_reason']), f"Expected no AT-γ in blocked_reason for passing case, got: '{row['blocked_reason']}'"
        
        print("  ✅ PASS scenario verified correctly")

def test_gamma_fail():
    """Test gamma-adherence check when it should FAIL"""
    print("\nTesting gamma-adherence FAIL scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with high regret (should fail)
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        create_mock_csv_with_regret(csv_path, avg_regret_value=1.5, num_events=10)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set gamma_bar = 1.0, so avg_regret = 1.5 should fail
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path], "test_grid", output_dir, mandatory_fields)
        
        # Check results
        assert len(result) == 1, f"Expected 1 output file, got {len(result)}"
        df = pd.read_csv(result[0])
        
        # Verify values for FAIL scenario
        row = df.iloc[0]
        print(f"  avg_regret_final: {row['avg_regret_final']}")
        print(f"  gamma_bar_threshold: {row['gamma_bar_threshold']}")
        print(f"  gamma_pass_overall: {row['gamma_pass_overall']}")
        print(f"  gamma_error: {row['gamma_error']}")
        print(f"  blocked_reason: '{row['blocked_reason']}'")
        
        # Assertions for FAIL scenario
        assert abs(row['avg_regret_final'] - 1.5) < 0.1, f"Expected avg_regret_final ~1.5, got {row['avg_regret_final']}"
        assert row['gamma_bar_threshold'] == 1.0, f"Expected gamma_bar_threshold=1.0, got {row['gamma_bar_threshold']}"
        assert row['gamma_pass_overall'] == False, f"Expected gamma_pass_overall=False, got {row['gamma_pass_overall']}"
        assert row['gamma_error'] > 0.0, f"Expected gamma_error>0.0 for failing case, got {row['gamma_error']}"
        expected_error = 1.5 / 1.0 - 1.0  # 0.5
        assert abs(row['gamma_error'] - expected_error) < 0.1, f"Expected gamma_error~{expected_error}, got {row['gamma_error']}"
        assert "AT-γ" in str(row['blocked_reason']), f"Expected AT-γ in blocked_reason for failing case, got: '{row['blocked_reason']}'"
        
        print("  ✅ FAIL scenario verified correctly")

def test_robust_regret_computation():
    """Test the robust regret computation fallback logic"""
    print("\nTesting robust regret computation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV without avg_regret column but with regret column
        events = []
        regret_values = [0.8, 0.9, 1.0, 1.1, 1.2]  # avg = 1.0
        for i, regret in enumerate(regret_values):
            event = {
                'event': i + 1000,
                'op': 'insert' if i < 3 else 'delete',
                'regret': regret,  # Per-event regret values
                'acc': 0.1 + i * 0.02,
            }
            events.append(event)
        
        df = pd.DataFrame(events)
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        df.to_csv(csv_path, index=False)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        mandatory_fields = {
            'gamma_bar': 0.9,  # Threshold below average regret
            'gamma_split': 0.6,
        }
        
        result = process_seed_output([csv_path], "test_grid", output_dir, mandatory_fields)
        
        # Check results
        df_result = pd.read_csv(result[0])
        row = df_result.iloc[0]
        
        print(f"  avg_regret_final: {row['avg_regret_final']} (expected ~1.0)")
        print(f"  gamma_pass_overall: {row['gamma_pass_overall']} (expected False)")
        
        # Should use fallback to df["regret"].mean() = 1.0
        assert abs(row['avg_regret_final'] - 1.0) < 0.01, f"Expected avg_regret_final=1.0, got {row['avg_regret_final']}"
        assert row['gamma_pass_overall'] == False, "Should fail since 1.0 > 0.9"
        
        print("  ✅ Robust computation verified correctly")

if __name__ == "__main__":
    print("Testing gamma-adherence functionality...")
    try:
        test_gamma_pass()
        test_gamma_fail()
        test_robust_regret_computation()
        print("\n✅ All gamma-adherence tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()