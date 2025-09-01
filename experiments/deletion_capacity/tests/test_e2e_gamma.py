#!/usr/bin/env python3
"""
End-to-end test for gamma-adherence check.
Tests integration with different gamma_bar values to verify pass/fail behavior.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))

from agents.grid_runner import process_seed_output

def create_realistic_csv(path: str, regret_pattern: str, num_events: int = 20):
    """Create a more realistic CSV file with different regret patterns"""
    events = []
    
    for i in range(num_events):
        op = "insert" if i < num_events // 2 else "delete"
        
        if regret_pattern == "low":
            # Low regret that should pass with gamma_bar=1.0
            base_regret = 0.3
            regret = base_regret + np.random.normal(0, 0.1)
        elif regret_pattern == "medium":
            # Medium regret that might pass/fail depending on gamma_bar
            base_regret = 0.8
            regret = base_regret + np.random.normal(0, 0.2)
        elif regret_pattern == "high":
            # High regret that should fail with gamma_bar=1.0
            base_regret = 1.5
            regret = base_regret + np.random.normal(0, 0.1)
        else:
            regret = 1.0
            
        event = {
            'event': i + 1000,
            'op': op,
            'regret': max(0, regret),  # Ensure non-negative
            'acc': 0.1 + i * 0.02,
            'eps_spent': i * 0.01,
            'capacity_remaining': 1.0 - (i * 0.01),
            'G_hat': 2.5,
            'D_hat': 1.8,
            'sigma_step_theory': 0.05,
            # Add some theory fields for AT checks
            'P_T_true': 0.1,
            'ST_running': 0.2,
            'g_norm': 1.0,
            'privacy_spend_running': 0.5,
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    df.to_csv(path, index=False)
    return df['regret'].mean()  # Return actual average for verification

def test_grid_scenario():
    """Test multiple scenarios with different gamma_bar values"""
    print("Testing grid scenario with different gamma_bar values...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test scenarios: (regret_pattern, gamma_bar, expected_pass)
        scenarios = [
            ("low", 1.0, True),      # Low regret (~0.3) vs gamma_bar=1.0 → should pass
            ("low", 0.2, False),     # Low regret (~0.3) vs gamma_bar=0.2 → should fail  
            ("medium", 1.0, True),   # Medium regret (~0.8) vs gamma_bar=1.0 → should pass
            ("medium", 0.5, False),  # Medium regret (~0.8) vs gamma_bar=0.5 → should fail
            ("high", 1.0, False),    # High regret (~1.5) vs gamma_bar=1.0 → should fail
            ("high", 2.0, True),     # High regret (~1.5) vs gamma_bar=2.0 → should pass
        ]
        
        for i, (pattern, gamma_bar, expected_pass) in enumerate(scenarios):
            print(f"\n  Scenario {i+1}: {pattern} regret vs gamma_bar={gamma_bar} (expect {'PASS' if expected_pass else 'FAIL'})")
            
            # Create CSV with specific regret pattern
            csv_path = os.path.join(temp_dir, f"scenario_{i}_seed_0.csv")
            actual_avg_regret = create_realistic_csv(csv_path, pattern, num_events=20)
            
            output_dir = os.path.join(temp_dir, f"output_{i}")
            os.makedirs(output_dir, exist_ok=True)
            
            mandatory_fields = {
                'gamma_bar': gamma_bar,
                'gamma_split': 0.5,
                'G_hat': 2.5,
                'D_hat': 1.8,
                'sigma_step_theory': 0.05,
                'target_G': 2.0,
                'target_PT': 0.1,
                'target_ST': 0.2,
                'rho_total': 1.0,
            }
            
            result = process_seed_output([csv_path], f"test_grid_{i}", output_dir, mandatory_fields)
            
            # Verify results
            assert len(result) == 1, f"Expected 1 output file, got {len(result)}"
            df = pd.read_csv(result[0])
            row = df.iloc[0]
            
            print(f"    actual_avg_regret: {actual_avg_regret:.3f}")
            print(f"    avg_regret_final: {row['avg_regret_final']:.3f}")
            print(f"    gamma_bar_threshold: {row['gamma_bar_threshold']}")
            print(f"    gamma_pass_overall: {row['gamma_pass_overall']}")
            print(f"    gamma_error: {row['gamma_error']:.3f}")
            print(f"    blocked_reason: '{row['blocked_reason']}'")
            
            # Verify the pass/fail logic
            actual_pass = row['gamma_pass_overall']
            if actual_pass != expected_pass:
                print(f"    ⚠️  Expected {expected_pass}, got {actual_pass}")
                # This might happen due to randomness, so let's check if it's reasonable
                should_pass = actual_avg_regret <= gamma_bar
                if should_pass == actual_pass:
                    print(f"    ✅ Pass/fail is correct based on actual regret")
                else:
                    raise AssertionError(f"Pass/fail logic incorrect: avg_regret={actual_avg_regret:.3f}, gamma_bar={gamma_bar}, should_pass={should_pass}, actual_pass={actual_pass}")
            else:
                print(f"    ✅ Pass/fail behavior correct")
            
            # Verify blocked_reason consistency
            if actual_pass:
                assert "AT-γ" not in str(row['blocked_reason']), f"Should not have AT-γ in blocked_reason for passing case"
            else:
                assert "AT-γ" in str(row['blocked_reason']), f"Should have AT-γ in blocked_reason for failing case"

def test_decomposition():
    """Test the insert/delete decomposition functionality"""
    print("\n\nTesting insert/delete decomposition...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CSV with different regret for insert vs delete operations
        events = []
        num_events = 20
        for i in range(num_events):
            if i < num_events // 2:
                op = "insert"
                regret = 0.3  # Low regret for inserts
            else:
                op = "delete"
                regret = 1.2  # High regret for deletes
                
            event = {
                'event': i + 1000,
                'op': op,
                'regret': regret,
                'acc': 0.1 + i * 0.02,
            }
            events.append(event)
        
        df = pd.DataFrame(events)
        csv_path = os.path.join(temp_dir, "0_decomp_test.csv")  # Include seed number
        df.to_csv(csv_path, index=False)
        
        output_dir = os.path.join(temp_dir, "output_decomp")
        os.makedirs(output_dir, exist_ok=True)
        
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.6,  # 60% to inserts, 40% to deletes
        }
        
        result = process_seed_output([csv_path], "decomp_grid", output_dir, mandatory_fields)
        
        # Verify decomposition
        df_result = pd.read_csv(result[0])
        row = df_result.iloc[0]
        
        print(f"  avg_regret_insert_only: {row['avg_regret_insert_only']:.3f} (expected ~0.3)")
        print(f"  avg_regret_delete_only: {row['avg_regret_delete_only']:.3f} (expected ~1.2)")
        print(f"  gamma_insert_threshold: {row['gamma_insert_threshold']:.3f} (expected 0.6)")
        print(f"  gamma_delete_threshold: {row['gamma_delete_threshold']:.3f} (expected 0.4)")
        print(f"  gamma_pass_insert: {row['gamma_pass_insert']} (expected True)")
        print(f"  gamma_pass_delete: {row['gamma_pass_delete']} (expected False)")
        
        # Verify values
        assert abs(row['avg_regret_insert_only'] - 0.3) < 0.01, "Insert regret should be ~0.3"
        assert abs(row['avg_regret_delete_only'] - 1.2) < 0.01, "Delete regret should be ~1.2"
        assert abs(row['gamma_insert_threshold'] - 0.6) < 0.01, "Insert threshold should be gamma_bar * gamma_split = 1.0 * 0.6"
        assert abs(row['gamma_delete_threshold'] - 0.4) < 0.01, "Delete threshold should be gamma_bar * (1-gamma_split) = 1.0 * 0.4"
        assert row['gamma_pass_insert'] == True, "Insert should pass (0.3 <= 0.6)"
        assert row['gamma_pass_delete'] == False, "Delete should fail (1.2 > 0.4)"
        
        print("  ✅ Decomposition logic verified correctly")

if __name__ == "__main__":
    print("Testing gamma-adherence end-to-end functionality...")
    np.random.seed(42)  # For reproducible results
    try:
        test_grid_scenario()
        test_decomposition()
        print("\n✅ All end-to-end gamma-adherence tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()