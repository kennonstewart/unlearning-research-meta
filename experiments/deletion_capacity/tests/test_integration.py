#!/usr/bin/env python3
"""
Integration test for the grid runner with different output granularities.
Creates mock experiment outputs and tests the full processing pipeline.
"""

import os
import tempfile
import shutil
import subprocess
import sys

def create_mock_experiment_output(output_dir: str, seed: int) -> str:
    """Create a mock CSV output that mimics run.py output."""
    import pandas as pd
    
    # Create mock event data
    events = []
    for i in range(10):
        op = "insert" if i < 5 else "delete"
        event = {
            'event': i + 1000,
            'op': op,
            'regret': 10.0 + i * 0.5,
            'acc': 0.1 + i * 0.02,
            'eps_spent': i * 0.01,
            'capacity_remaining': 1.0 - (i * 0.01),
            'G_hat': 2.5,
            'D_hat': 1.8,
            'sigma_step_theory': 0.05,
            'gamma_bar': 1.0,
            'gamma_split': 0.9,
            'quantile': 0.95,
            'accountant_type': 'legacy'
        }
        events.append(event)
    
    df = pd.DataFrame(events)
    csv_path = os.path.join(output_dir, f"{seed}_memorypair.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

def test_integration():
    """Test the full grid runner pipeline with mock data."""
    print("Testing full integration with mock experiment outputs...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        base_out = os.path.join(temp_dir, "results")
        os.makedirs(base_out, exist_ok=True)
        
        # Create a minimal grid file
        grid_file = os.path.join(temp_dir, "test_grid.yaml")
        with open(grid_file, 'w') as f:
            f.write("""gamma_bar: [1.0]
gamma_split: [0.9]
quantile: [0.95]
delete_ratio: [10]
accountant: [\"legacy\"]
eps_total: [1.0]
""")
        
        # Create mock outputs for testing
        grid_id = "gamma1.0_split0.90_q0.95_k10_legacy_eps1.0"
        grid_dir = os.path.join(base_out, "sweep", grid_id)
        os.makedirs(grid_dir, exist_ok=True)
        
        # Create mock CSV files for 2 seeds
        mock_files = []
        for seed in [0, 1]:
            mock_file = create_mock_experiment_output(grid_dir, seed)
            mock_files.append(mock_file)
        
        print(f"Created {len(mock_files)} mock CSV files")
        
        # Test each granularity by processing the mock files directly
        for granularity in ["seed", "event", "aggregate"]:
            print(f"\nTesting {granularity} granularity...")
            
            # Import processing functions
            sys.path.append(os.path.join(os.path.dirname(__file__), "."))
            from agents.grid_runner import process_seed_output, process_event_output, process_aggregate_output
            
            mandatory_fields = {
                'G_hat': 2.5,
                'D_hat': 1.8,
                'sigma_step_theory': 0.05,
                'gamma_bar': 1.0,
                'gamma_split': 0.9,
                'quantile': 0.95,
                'delete_ratio': 10,
                'accountant': 'legacy',
                'eps_total': 1.0
            }
            
            if granularity == "seed":
                result = process_seed_output(mock_files, grid_id, grid_dir, mandatory_fields)
            elif granularity == "event":
                result = process_event_output(mock_files, grid_id, grid_dir, mandatory_fields)
            elif granularity == "aggregate":
                result = process_aggregate_output(mock_files, grid_id, grid_dir, mandatory_fields)
            
            if result:
                print(f"  Generated {len(result) if isinstance(result, list) else 1} files")
                
                # Validate each output file
                result_files = result if isinstance(result, list) else [result]
                for file_path in result_files:
                    if os.path.exists(file_path):
                        # Run validation
                        validation_script = os.path.join(os.path.dirname(__file__), "validate_output.py")
                        cmd = [sys.executable, validation_script, file_path, "--granularity", granularity]
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if proc.returncode == 0:
                            print(f"    ✅ {os.path.basename(file_path)} validated successfully")
                        else:
                            print(f"    ❌ {os.path.basename(file_path)} validation failed: {proc.stdout}{proc.stderr}")
                    else:
                        print(f"    ❌ File not found: {file_path}")
            else:
                print(f"  ❌ No files generated for {granularity}")

if __name__ == "__main__":
    print("Running integration test...")
    try:
        test_integration()
        print("\n✅ Integration test completed!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()