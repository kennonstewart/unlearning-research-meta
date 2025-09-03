#!/usr/bin/env python3
"""
Validation utility to check that output files contain the mandatory fields.
"""

import pandas as pd
import argparse
import os

def validate_mandatory_fields(csv_path: str, output_granularity: str) -> bool:
    """Validate that CSV contains mandatory fields for the given granularity."""
    mandatory_fields = ['G_hat', 'D_hat', 'sigma_step_theory']
    
    try:
        df = pd.read_csv(csv_path)
        
        missing_fields = [field for field in mandatory_fields if field not in df.columns]
        if missing_fields:
            print(f"❌ Missing mandatory fields in {csv_path}: {missing_fields}")
            return False
        
        # Check for NaN values in mandatory fields
        for field in mandatory_fields:
            if df[field].isna().any():
                print(f"❌ Mandatory field {field} has NaN values in {csv_path}")
                return False
        
        # Mode-specific validations
        if output_granularity == "seed":
            expected_cols = ['seed', 'grid_id', 'avg_regret_empirical', 'N_star_emp', 'm_emp']
            missing_mode_cols = [col for col in expected_cols if col not in df.columns]
            if missing_mode_cols:
                print(f"❌ Missing seed mode columns in {csv_path}: {missing_mode_cols}")
                return False
        
        elif output_granularity == "event":
            expected_cols = ['event', 'event_type', 'op', 'regret', 'acc', 'seed', 'grid_id']
            missing_mode_cols = [col for col in expected_cols if col not in df.columns]
            if missing_mode_cols:
                print(f"❌ Missing event mode columns in {csv_path}: {missing_mode_cols}")
                return False
        
        elif output_granularity == "aggregate":
            expected_cols = ['grid_id', 'num_seeds', 'avg_regret_mean', 'N_star_mean', 'm_mean']
            missing_mode_cols = [col for col in expected_cols if col not in df.columns]
            if missing_mode_cols:
                print(f"❌ Missing aggregate mode columns in {csv_path}: {missing_mode_cols}")
                return False
        
        print(f"✅ {csv_path} validation passed for {output_granularity} mode")
        return True
        
    except Exception as e:
        print(f"❌ Error validating {csv_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate output CSV files for mandatory fields")
    parser.add_argument("csv_path", help="Path to CSV file to validate")
    parser.add_argument("--granularity", choices=["seed", "event", "aggregate"], required=True,
                        help="Expected output granularity")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"❌ File not found: {args.csv_path}")
        return 1
    
    if validate_mandatory_fields(args.csv_path, args.granularity):
        print(f"✅ Validation successful!")
        return 0
    else:
        print(f"❌ Validation failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())