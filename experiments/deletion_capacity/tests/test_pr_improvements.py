#!/usr/bin/env python3
"""
Unit tests for PR improvements addressing follow-ups from Experiment A.
Tests privacy spend instrumentation, gamma-regret acceptance checks, and ST source disambiguation.
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
sys.path.append('..')
from agents.grid_runner import process_seed_output


def test_privacy_spend_instrumentation():
    """Test privacy spend source selection and transparency columns."""
    print("Testing privacy spend instrumentation improvements...")
    
    # Create sample CSV data with different privacy spend columns
    event_data = {
        'event': list(range(10)),
        'op': ['insert', 'delete'] * 5,
        'regret': [0.1, 0.2, 0.15, 0.25, 0.12, 0.18, 0.14, 0.22, 0.16, 0.20],
        'acc': [0.95, 0.94, 0.95, 0.93, 0.94, 0.95, 0.94, 0.93, 0.95, 0.94],
        'rho_spent': [0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],  # Odometer source
        'privacy_spend_running': [0.12, 0.17, 0.22, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62],  # Stream source
        'privacy_spend': [0.11, 0.16, 0.21, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61],  # Fallback source
        'P_T_true': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        'ST_running': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        'S_scalar': [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],  # Should be avoided in favor of ST_running
        'g_norm': [0.8, 0.9, 0.85, 0.95, 0.82, 0.88, 0.84, 0.92, 0.86, 0.90],
        'clip_applied': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    }
    
    # Create temporary CSV file with proper naming for seed extraction
    with tempfile.NamedTemporaryFile(mode='w', suffix='_001_test.csv', delete=False) as f:
        df = pd.DataFrame(event_data)
        df.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        # Set up mandatory fields with targets and gamma parameters
        mandatory_fields = {
            'target_G': 1.0,
            'target_PT': 1.9,
            'target_ST': 2.9,
            'rho_total': 1.0,
            'gamma_bar': 0.3,  # Overall gamma budget
            'gamma_split': 0.7  # 70% for inserts, 30% for deletes
        }
        
        # Process the CSV with seed output
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = process_seed_output([csv_file], 'test_grid', temp_dir, mandatory_fields)
            
            assert len(processed_files) == 1, "Should process one seed file"
            
            # Read the processed results
            result_df = pd.read_csv(processed_files[0])
            row = result_df.iloc[0]
            
            # Test privacy spend source selection and transparency
            # Should prefer rho_spent (odometer) for final value: 0.6
            assert row['privacy_spend_odometer_final'] == 0.6, f"Expected odometer final=0.6, got {row['privacy_spend_odometer_final']}"
            # Should store stream source separately: 0.62
            assert row['privacy_spend_stream_final'] == 0.62, f"Expected stream final=0.62, got {row['privacy_spend_stream_final']}"
            # rho_spent_final should be alias of odometer (backward compatibility)
            assert row['rho_spent_final'] == 0.6, f"Expected rho_spent_final=0.6, got {row['rho_spent_final']}"
            
            print("✓ Privacy spend source selection works correctly")
            
    finally:
        os.unlink(csv_file)


def test_gamma_regret_acceptance():
    """Test gamma-regret acceptance checks and op-wise regret computation."""
    print("Testing gamma-regret acceptance checks...")
    
    # Create sample data with insert/delete operations
    event_data = {
        'event': list(range(10)),
        'op': ['insert'] * 6 + ['delete'] * 4,  # 6 inserts, 4 deletes
        'regret': [0.1, 0.05, 0.08, 0.12, 0.07, 0.09,   # Insert regrets (avg = 0.085)
                   0.2, 0.15, 0.18, 0.22],              # Delete regrets (avg = 0.1875)
        'acc': [0.95] * 10,
        'P_T_true': [1.0] * 10,
        'ST_running': [2.0] * 10,
        'g_norm': [0.8] * 10,
        'clip_applied': [0] * 10
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_002_test.csv', delete=False) as f:
        df = pd.DataFrame(event_data)
        df.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        # Test case 1: All regrets within budget
        mandatory_fields = {
            'target_G': 1.0,
            'target_PT': 1.0,
            'target_ST': 2.0,
            'rho_total': 1.0,
            'gamma_bar': 0.25,  # Overall budget should pass (avg regret = 0.1175)
            'gamma_split': 0.6   # 60% for inserts (0.15), 40% for deletes (0.1)
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = process_seed_output([csv_file], 'test_grid', temp_dir, mandatory_fields)
            result_df = pd.read_csv(processed_files[0])
            row = result_df.iloc[0]
            
            # Check op-wise regret computation
            expected_avg_regret_insert = 0.085  # (0.1+0.05+0.08+0.12+0.07+0.09)/6
            expected_avg_regret_delete = 0.1875  # (0.2+0.15+0.18+0.22)/4
            expected_gamma_ins = 0.25 * 0.6  # 0.15
            expected_gamma_del = 0.25 * 0.4  # 0.1
            
            assert abs(row['avg_regret_insert'] - expected_avg_regret_insert) < 1e-6, f"Expected insert regret {expected_avg_regret_insert}, got {row['avg_regret_insert']}"
            assert abs(row['avg_regret_delete'] - expected_avg_regret_delete) < 1e-6, f"Expected delete regret {expected_avg_regret_delete}, got {row['avg_regret_delete']}"
            assert abs(row['gamma_ins'] - expected_gamma_ins) < 1e-6, f"Expected gamma_ins {expected_gamma_ins}, got {row['gamma_ins']}"
            assert abs(row['gamma_del'] - expected_gamma_del) < 1e-6, f"Expected gamma_del {expected_gamma_del}, got {row['gamma_del']}"
            
            # Check acceptance flags - inserts should pass (0.085 <= 0.15), deletes should fail (0.1875 > 0.1)
            assert row['AT_gamma_overall'] == True, f"Overall gamma acceptance should pass"
            assert row['AT_gamma_insert'] == True, f"Insert gamma acceptance should pass (0.085 <= 0.15)"
            assert row['AT_gamma_delete'] == False, f"Delete gamma acceptance should fail (0.1875 > 0.1)"
            
            print("✓ Gamma-regret acceptance checks work correctly")
            
    finally:
        os.unlink(csv_file)


def test_st_source_disambiguation():
    """Test ST source selection preference for ST_running over S_scalar."""
    print("Testing ST source disambiguation...")
    
    # Create data with both ST_running and S_scalar
    event_data = {
        'event': list(range(5)),
        'op': ['insert'] * 5,
        'regret': [0.1] * 5,
        'acc': [0.95] * 5,
        'P_T_true': [1.0] * 5,
        'ST_running': [3.0, 3.1, 3.2, 3.3, 3.4],  # Should be preferred
        'S_scalar': [2.5, 2.6, 2.7, 2.8, 2.9],    # Should be ignored when ST_running available
        'g_norm': [0.8] * 5,
        'clip_applied': [0] * 5
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_003_test.csv', delete=False) as f:
        df = pd.DataFrame(event_data)
        df.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        mandatory_fields = {
            'target_G': 1.0,
            'target_PT': 1.0,
            'target_ST': 3.4,  # Should match ST_running final value
            'gamma_bar': 0.5,
            'gamma_split': 0.5
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = process_seed_output([csv_file], 'test_grid', temp_dir, mandatory_fields)
            result_df = pd.read_csv(processed_files[0])
            row = result_df.iloc[0]
            
            # Should use ST_running (3.4) not S_scalar (2.9)
            assert row['ST_final'] == 3.4, f"Expected ST_final=3.4 from ST_running, got {row['ST_final']}"
            
            print("✓ ST source disambiguation works correctly")
            
    finally:
        os.unlink(csv_file)


def test_fallback_with_missing_odometer():
    """Test fallback behavior when odometer source is not available."""
    print("Testing privacy spend fallback behavior...")
    
    # Create data without rho_spent column (no odometer)
    event_data = {
        'event': list(range(5)),
        'op': ['insert'] * 5,
        'regret': [0.1] * 5,
        'acc': [0.95] * 5,
        'privacy_spend_running': [0.1, 0.2, 0.3, 0.4, 0.5],  # Stream source only
        'privacy_spend': [0.09, 0.19, 0.29, 0.39, 0.49],     # Legacy fallback
        'P_T_true': [1.0] * 5,
        'ST_running': [2.0] * 5,
        'g_norm': [0.8] * 5,
        'clip_applied': [0] * 5
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_004_test.csv', delete=False) as f:
        df = pd.DataFrame(event_data)
        df.to_csv(f.name, index=False)
        csv_file = f.name
    
    try:
        mandatory_fields = {
            'rho_total': 1.0,
            'gamma_bar': 0.5,
            'gamma_split': 0.5
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = process_seed_output([csv_file], 'test_grid', temp_dir, mandatory_fields)
            result_df = pd.read_csv(processed_files[0])
            row = result_df.iloc[0]
            
            # Should use stream source as fallback
            assert pd.isna(row['privacy_spend_odometer_final']), "Odometer final should be NaN when not available"
            assert row['privacy_spend_stream_final'] == 0.5, f"Expected stream final=0.5, got {row['privacy_spend_stream_final']}"
            assert row['rho_spent_final'] == 0.5, f"Expected rho_spent_final=0.5 (fallback), got {row['rho_spent_final']}"
            
            print("✓ Privacy spend fallback behavior works correctly")
            
    finally:
        os.unlink(csv_file)


if __name__ == "__main__":
    test_privacy_spend_instrumentation()
    test_gamma_regret_acceptance()
    test_st_source_disambiguation()
    test_fallback_with_missing_odometer()
    print("\n✅ All PR improvement tests passed!")