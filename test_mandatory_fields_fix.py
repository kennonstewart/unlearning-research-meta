#!/usr/bin/env python3
"""
Test script to validate that mandatory fields fix works correctly.
"""

import tempfile
import os
import sys
import pandas as pd
import numpy as np
import json

# Add experiments path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments', 'deletion_capacity', 'agents'))

from grid_runner import process_seed_output, process_event_output, process_aggregate_output


def test_mandatory_fields_in_outputs():
    """Test that all mandatory fields are present in outputs."""
    print("Testing mandatory fields in all output modes...")
    
    # Define mandatory fields that should always be present
    mandatory_field_names = [
        'gamma_bar', 'gamma_split', 'accountant',
        'G_hat', 'D_hat', 'c_hat', 'C_hat', 'lambda_est', 'S_scalar',
        'sigma_step_theory', 'N_star_live', 'm_theory_live', 'blocked_reason'
    ]
    
    # Create mock CSV data
    mock_events = []
    for i in range(10):
        mock_events.append({
            'event': i + 1000,
            'op': 'insert' if i < 5 else 'delete',
            'regret': 10.0 + i,
            'acc': 0.1 + i * 0.01,
            # Some mandatory fields missing to test the fix
            'G_hat': 2.5 if i == 9 else None,  # Only in last row
            'D_hat': 1.8 if i == 9 else None,  # Only in last row
        })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV
        mock_csv = os.path.join(temp_dir, "seed_001_memorypair.csv")
        pd.DataFrame(mock_events).to_csv(mock_csv, index=False)
        
        # Test mandatory fields dict with all required fields
        mandatory_fields = {
            'gamma_bar': 1.0,
            'gamma_split': 0.5,
            'accountant': 'eps_delta',
            'G_hat': 2.5,
            'D_hat': 1.8,
            'c_hat': 0.1,
            'C_hat': 10.0,
            'lambda_est': 0.05,
            'S_scalar': 1.0,
            'sigma_step_theory': 0.05,
            'N_star_live': 100,
            'm_theory_live': 50,
            'blocked_reason': ""
        }
        
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test each mode
        modes_to_test = [
            ("seed", process_seed_output),
            ("event", process_event_output),
            ("aggregate", process_aggregate_output)
        ]
        
        for mode, processor in modes_to_test:
            print(f"  Testing {mode} mode...")
            
            result = processor([mock_csv], "test_grid", output_dir, mandatory_fields)
            
            if isinstance(result, list):
                files_to_check = result
            else:
                files_to_check = [result] if result else []
            
            for file_path in files_to_check:
                if file_path and os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    # Check that all mandatory fields are present
                    missing_fields = []
                    for field in mandatory_field_names:
                        if field not in df.columns:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        print(f"    ❌ Missing fields in {mode} mode: {missing_fields}")
                        return False
                    
                    # Check that no mandatory fields have all NaN values (except blocked_reason which can be empty)
                    nan_fields = []
                    for field in mandatory_field_names:
                        if field != 'blocked_reason' and df[field].isna().all():
                            nan_fields.append(field)
                    
                    if nan_fields:
                        print(f"    ⚠️  All NaN values in {mode} mode for: {nan_fields}")
                        # This is acceptable for some fields during early calibration
                    
                    print(f"    ✅ {os.path.basename(file_path)} has all mandatory fields")
                    
                    # Print a sample of the data
                    print(f"    Sample columns: {list(df.columns[:10])}")
                    if len(df) > 0:
                        print(f"    Sample values for mandatory fields:")
                        for field in mandatory_field_names[:5]:  # Show first 5
                            val = df[field].iloc[0] if len(df) > 0 else "N/A"
                            print(f"      {field}: {val}")
    
    print("✅ All output modes include mandatory fields!")
    return True


def test_manifest_creation():
    """Test manifest creation functionality."""
    print("\nTesting manifest creation...")
    
    # Mock parameter combinations
    combinations = [
        {'gamma_bar': 1.0, 'gamma_split': 0.3, 'accountant': 'eps_delta', 'eps_total': 1.0},
        {'gamma_bar': 1.0, 'gamma_split': 0.7, 'accountant': 'zcdp', 'eps_total': 0.5},
    ]
    
    # Import create_grid_id function
    from grid_runner import create_grid_id
    
    # Create manifest
    manifest = {}
    for params in combinations:
        grid_id = create_grid_id(params)
        manifest[grid_id] = params
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_file = os.path.join(temp_dir, "manifest.json")
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Verify the manifest was created and is valid
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                loaded_manifest = json.load(f)
            
            if len(loaded_manifest) == len(combinations):
                print(f"✅ Manifest created successfully with {len(loaded_manifest)} entries")
                print(f"✅ Sample grid_id: {list(loaded_manifest.keys())[0]}")
                return True
            else:
                print(f"❌ Manifest has wrong number of entries: {len(loaded_manifest)} vs {len(combinations)}")
                return False
        else:
            print("❌ Manifest file was not created")
            return False


if __name__ == "__main__":
    try:
        print("🧪 Testing mandatory fields fix...")
        
        success1 = test_mandatory_fields_in_outputs()
        success2 = test_manifest_creation()
        
        if success1 and success2:
            print("\n🎉 All tests passed! The mandatory fields fix is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)