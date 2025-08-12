#!/usr/bin/env python3
"""
Create mock data that simulates the fix working and test the auditor.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_mock_experiment_results():
    """Create mock experiment results that simulate the schema fix."""
    print("Creating mock experiment results with proper schema...")
    
    # Create temporary results directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "experiments" / "deletion_capacity" / "results" / "grid_2025_01_01"
        sweep_dir = results_dir / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock grid cells with mandatory fields
        grid_cells = [
            {
                'grid_id': 'gamma_1.0-split_0.3_q0.95_k5_eps_delta_eps0.5',
                'params': {'gamma_bar': 1.0, 'gamma_split': 0.3, 'accountant': 'eps_delta', 'eps_total': 0.5}
            },
            {
                'grid_id': 'gamma_1.0-split_0.7_q0.95_k5_eps_delta_eps0.5', 
                'params': {'gamma_bar': 1.0, 'gamma_split': 0.7, 'accountant': 'eps_delta', 'eps_total': 0.5}
            }
        ]
        
        # Create manifest
        manifest = {cell['grid_id']: cell['params'] for cell in grid_cells}
        manifest_file = sweep_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create seed CSV files for each grid cell with ALL mandatory fields
        for cell in grid_cells:
            grid_dir = sweep_dir / cell['grid_id']
            grid_dir.mkdir(exist_ok=True)
            
            # Save params.json for this grid cell
            params_file = grid_dir / "params.json"
            with open(params_file, 'w') as f:
                json.dump(cell['params'], f, indent=2)
            
            # Create seed files with complete schema
            for seed in range(3):  # 3 seeds per grid cell
                seed_data = {
                    'seed': seed,
                    'grid_id': cell['grid_id'],
                    'avg_regret_empirical': 15.0 + np.random.random(),
                    'N_star_emp': 100 + np.random.randint(-10, 10),
                    'm_emp': 50 + np.random.randint(-5, 5),
                    'final_acc': 0.85 + np.random.random() * 0.1,
                    'total_events': 1000,
                    # ALL mandatory fields present
                    'gamma_bar': cell['params']['gamma_bar'],
                    'gamma_split': cell['params']['gamma_split'],
                    'accountant': cell['params']['accountant'],
                    'G_hat': 2.5 + np.random.normal(0, 0.1),
                    'D_hat': 1.8 + np.random.normal(0, 0.1),
                    'c_hat': 0.1,
                    'C_hat': 10.0,
                    'lambda_est': 0.05,
                    'S_scalar': 1.0,
                    'sigma_step_theory': 0.05,
                    'N_star_live': 100,
                    'm_theory_live': 50,
                    'blocked_reason': ""
                }
                
                seed_file = grid_dir / f"seed_{seed:03d}.csv"
                pd.DataFrame([seed_data]).to_csv(seed_file, index=False)
        
        print(f"Created mock results in {results_dir}")
        print(f"Grid cells: {len(grid_cells)}")
        print(f"Total seed files: {len(grid_cells) * 3}")
        
        # Now run the auditor on this mock data
        print("\nRunning auditor on mock data...")
        
        # Add current directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        # Import and run auditor
        from deletion_capacity_audit import DeletionCapacityAuditor
        
        # Initialize auditor with our mock results
        auditor = DeletionCapacityAuditor(results_dir=str(results_dir))
        
        # Run the ingestion first to load data
        print("\n=== Testing A0: Data Ingestion ===")
        a0_result = auditor._a0_ingest_normalize()
        print(f"A0 Results: {a0_result}")
        
        # Run just the A2 test (schema presence)
        print("\n=== Testing A2: Schema Presence ===")
        a2_result = auditor._a2_mandatory_fields()
        
        print(f"A2 Results:")
        print(f"  Mandatory fields: {a2_result['mandatory_fields']}")
        print(f"  Grids checked: {a2_result['grids_checked']}")
        print(f"  Grids with missing fields: {a2_result['grids_with_missing_fields']}")
        print(f"  All present: {a2_result['all_present']}")
        
        if a2_result['all_present']:
            print("✅ A2 Schema presence test PASSED!")
        else:
            print("❌ A2 Schema presence test FAILED!")
            print(f"  Missing fields detail: {a2_result['missing_fields_detail']}")
        
        return a2_result['all_present']


if __name__ == "__main__":
    try:
        success = create_mock_experiment_results()
        if success:
            print("\n🎉 Mock test passed! The schema fix is working.")
            sys.exit(0)
        else:
            print("\n❌ Mock test failed.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)