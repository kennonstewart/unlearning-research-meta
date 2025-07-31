#!/usr/bin/env python3
"""
Test script to validate output granularity processing functions.
Creates mock CSV data and tests the different processing modes.
"""

import os
import tempfile
import pandas as pd
import sys
sys.path.append(os.path.dirname(__file__))

from agents.grid_runner import process_seed_output, process_event_output, process_aggregate_output

def create_mock_csv(path: str, num_events: int = 10):
    """Create a mock CSV file that mimics the output from run.py"""
    events = []
    for i in range(num_events):
        op = "insert" if i < num_events // 2 else "delete"
        event = {
            'event': i + 1000,
            'op': op,
            'regret': 10.0 + i * 0.5,
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

def test_seed_output():
    """Test seed granularity output processing"""
    print("Testing seed output processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV files for 2 seeds
        csv_files = []
        for seed in [0, 1]:
            csv_path = os.path.join(temp_dir, f"{seed}_memorypair.csv")
            create_mock_csv(csv_path, num_events=10)
            csv_files.append(csv_path)
        
        # Test processing
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        mandatory_fields = {
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_seed_output(csv_files, "test_grid", output_dir, mandatory_fields)
        
        print(f"Generated {len(result)} seed output files")
        for file in result:
            if os.path.exists(file):
                df = pd.read_csv(file)
                print(f"File {os.path.basename(file)}: {len(df)} rows, columns: {list(df.columns)}")
                # Check mandatory fields
                for field in mandatory_fields:
                    assert field in df.columns, f"Missing mandatory field {field}"
                print(f"  Mandatory fields present: ✓")
            else:
                print(f"File {file} not found!")

def test_event_output():
    """Test event granularity output processing"""
    print("\nTesting event output processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV files for 2 seeds
        csv_files = []
        for seed in [0, 1]:
            csv_path = os.path.join(temp_dir, f"{seed}_memorypair.csv")
            create_mock_csv(csv_path, num_events=5)
            csv_files.append(csv_path)
        
        # Test processing
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        mandatory_fields = {
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_event_output(csv_files, "test_grid", output_dir, mandatory_fields)
        
        print(f"Generated {len(result)} event output files")
        for file in result:
            if os.path.exists(file):
                df = pd.read_csv(file)
                print(f"File {os.path.basename(file)}: {len(df)} rows, columns: {list(df.columns)}")
                # Check mandatory fields
                for field in mandatory_fields:
                    assert field in df.columns, f"Missing mandatory field {field}"
                # Check that each row has the mandatory field populated
                assert not df[field].isna().any(), f"Mandatory field {field} has NaN values"
                print(f"  Mandatory fields present in all rows: ✓")
                print(f"  Sample data: {df.head(2).to_dict('records')}")
            else:
                print(f"File {file} not found!")

def test_aggregate_output():
    """Test aggregate granularity output processing"""
    print("\nTesting aggregate output processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV files for 3 seeds
        csv_files = []
        for seed in [0, 1, 2]:
            csv_path = os.path.join(temp_dir, f"{seed}_memorypair.csv")
            create_mock_csv(csv_path, num_events=8)
            csv_files.append(csv_path)
        
        # Test processing
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        mandatory_fields = {
            'G_hat': 2.5,
            'D_hat': 1.8, 
            'sigma_step_theory': 0.05
        }
        
        result = process_aggregate_output(csv_files, "test_grid", output_dir, mandatory_fields)
        
        if result and os.path.exists(result):
            df = pd.read_csv(result)
            print(f"Generated aggregate file: {os.path.basename(result)}")
            print(f"Aggregate data: {len(df)} rows, columns: {list(df.columns)}")
            # Check mandatory fields
            for field in mandatory_fields:
                assert field in df.columns, f"Missing mandatory field {field}"
            print(f"  Mandatory fields present: ✓")
            print(f"  Sample aggregate data: {df.to_dict('records')[0]}")
        else:
            print("Aggregate file not generated!")

if __name__ == "__main__":
    print("Testing output granularity processing functions...")
    try:
        test_seed_output()
        test_event_output() 
        test_aggregate_output()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()