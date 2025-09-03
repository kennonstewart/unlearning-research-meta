#!/usr/bin/env python3
"""
Test script to verify γ-adherence functionality with realistic grid runner scenarios.
This simulates the actual use case with parameter combinations.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))

from experiment.grid_runner import process_seed_output

def create_realistic_mock_csv(path: str, regret_pattern: str = "increasing"):
    """Create a realistic mock CSV that simulates actual experiment output"""
    num_events = 20
    events = []
    
    for i in range(num_events):
        # Alternate between insert and delete operations  
        op = "insert" if i < num_events * 0.6 else "delete"
        
        # Create realistic regret patterns
        if regret_pattern == "increasing":
            regret = 0.1 + i * 0.05  # Steadily increasing regret
            avg_regret = (0.1 + regret) / 2 * (i + 1) / (i + 1)  # Simple average
            cum_regret = sum(0.1 + j * 0.05 for j in range(i + 1))
        elif regret_pattern == "decreasing": 
            regret = 1.0 - i * 0.03  # Decreasing regret
            avg_regret = regret
            cum_regret = sum(1.0 - j * 0.03 for j in range(i + 1))
        else:  # stable
            regret = 0.5
            avg_regret = 0.5
            cum_regret = 0.5 * (i + 1)
        
        event = {
            'event': i,
            'op': op,
            'regret': max(regret, 0.01),  # Ensure positive regret
            'avg_regret': max(avg_regret, 0.01),
            'cum_regret': cum_regret,
            'regret_increment': max(regret, 0.01),
            'acc': 0.8 + i * 0.005,
            'g_norm': 2.0 + np.random.normal(0, 0.1),
            'clip_applied': 0 if i < 15 else 1,  # Some clipping later
            'P_T_true': i * 0.1,
            'ST_running': i * 0.2 + 1.0,
            'S_scalar': i * 0.2 + 1.0,
            'privacy_spend_running': i * 0.01,
            'eps_spent': i * 0.01,
            'capacity_remaining': 1.0 - (i * 0.01),
            'G_hat': 2.5,
            'D_hat': 1.8,
            'sigma_step_theory': 0.05,
            'c_hat': 1.2,
            'C_hat': 5.0,
            'lambda_est': 0.1,
            'N_star_live': 100,
            'm_theory_live': 50
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    df.to_csv(path, index=False)
    return path

def test_realistic_gamma_scenario():
    """Test γ-adherence with realistic experiment parameters"""
    print("Testing realistic γ-adherence scenario...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple seeds with different regret patterns
        scenarios = [
            ("0_memorypair.csv", "stable"),      # Should pass with avg_regret ≈ 0.5
            ("1_memorypair.csv", "increasing"),  # May fail depending on gamma_bar
            ("2_memorypair.csv", "decreasing"),  # Should pass
        ]
        
        csv_files = []
        for filename, pattern in scenarios:
            csv_path = os.path.join(temp_dir, filename)
            create_realistic_mock_csv(csv_path, pattern)
            csv_files.append(csv_path)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test scenario 1: Strict gamma threshold (gamma_bar = 0.6)
        print("  Testing strict γ̄ = 0.6 scenario...")
        mandatory_fields_strict = {
            'gamma_bar': 0.6,
            'gamma_split': 0.7,  # 70% for insert, 30% for delete
            'target_G': 2.5,
            'target_PT': 2.0,
            'target_ST': 4.0,
            'rho_total': 0.2,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output(csv_files, "test_strict", output_dir, mandatory_fields_strict)
        
        # Verify all output files created
        assert len(result) == 3, f"Expected 3 output files, got {len(result)}"
        
        # Check each result
        pass_count = 0
        fail_count = 0
        
        for i, output_file in enumerate(result):
            df = pd.read_csv(output_file)
            row = df.iloc[0]
            
            print(f"    Seed {i}: avg_regret_final={row['avg_regret_final']:.3f}, "
                  f"gamma_pass_overall={row['gamma_pass_overall']}, "
                  f"blocked_reason='{row['blocked_reason']}'")
            
            # Verify γ columns exist and have correct values
            assert row['gamma_bar_threshold'] == 0.6
            assert row['gamma_split_threshold'] == 0.7
            assert abs(row['gamma_insert_threshold'] - 0.42) < 0.01  # 0.6 * 0.7
            assert abs(row['gamma_delete_threshold'] - 0.18) < 0.01  # 0.6 * 0.3
            
            # Check pass/fail logic
            if row['gamma_pass_overall']:
                pass_count += 1
                assert row['avg_regret_final'] <= 0.6, f"Should pass: {row['avg_regret_final']} <= 0.6"
                assert 'AT-γ' not in str(row['blocked_reason']), "Passing runs should not have AT-γ"
            else:
                fail_count += 1
                assert row['avg_regret_final'] > 0.6, f"Should fail: {row['avg_regret_final']} > 0.6"
                assert 'AT-γ' in str(row['blocked_reason']), "Failing runs should have AT-γ"
                
        print(f"    Results: {pass_count} passed, {fail_count} failed")
        
        # Test scenario 2: Lenient gamma threshold (gamma_bar = 2.0)
        print("  Testing lenient γ̄ = 2.0 scenario...")
        
        # Clear output directory
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
            
        mandatory_fields_lenient = mandatory_fields_strict.copy()
        mandatory_fields_lenient['gamma_bar'] = 2.0
        
        result = process_seed_output(csv_files, "test_lenient", output_dir, mandatory_fields_lenient)
        
        # With lenient threshold, most should pass
        pass_count_lenient = 0
        for output_file in result:
            df = pd.read_csv(output_file)
            row = df.iloc[0]
            if row['gamma_pass_overall']:
                pass_count_lenient += 1
                
        print(f"    Lenient results: {pass_count_lenient} passed, {3-pass_count_lenient} failed")
        
        # Should have more passes with lenient threshold
        assert pass_count_lenient >= pass_count, "Lenient threshold should have more passes"
        
        print("  ✅ Realistic scenarios validated")

def test_gamma_edge_cases():
    """Test edge cases for γ-adherence checking"""
    print("Testing γ-adherence edge cases...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Edge case 1: Exactly equal to threshold
        csv_path = os.path.join(temp_dir, "0_memorypair.csv")
        events = []
        for i in range(5):
            events.append({
                'event': i,
                'op': 'insert',
                'regret': 1.0,  # Exactly 1.0
                'avg_regret': 1.0,
                'acc': 0.8,
                'G_hat': 2.5,
                'D_hat': 1.8,
                'sigma_step_theory': 0.05
            })
        pd.DataFrame(events).to_csv(csv_path, index=False)
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test with gamma_bar = 1.0 (exactly equal)
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output([csv_path], "test_edge", output_dir, mandatory_fields)
        df = pd.read_csv(result[0])
        row = df.iloc[0]
        
        # Equal should pass (≤ condition)
        assert row['gamma_pass_overall'] == True, "Equal values should pass (≤ condition)"
        assert row['gamma_error'] == 0.0, "Error should be 0 for exact match"
        assert 'AT-γ' not in str(row['blocked_reason']), "Equal case should not block"
        
        print("  ✅ Edge cases validated")

if __name__ == "__main__":
    print("Testing γ-adherence with realistic scenarios...")
    try:
        test_realistic_gamma_scenario()
        test_gamma_edge_cases()
        print("\n✅ All realistic γ-adherence tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()