#!/usr/bin/env python3
"""
Test script to verify rho_spent privacy metric fix.
Validates that disabling regret gate allows privacy accounting to work properly.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add imports for the deletion capacity experiment
sys.path.insert(0, "/home/runner/work/unlearning-research-meta/unlearning-research-meta/experiments/deletion_capacity")
sys.path.insert(0, "/home/runner/work/unlearning-research-meta/unlearning-research-meta/code")

from config import Config
from runner import ExperimentRunner

def test_rho_fix():
    """Test that the rho_spent fix works correctly."""
    print("=== Testing rho_spent Privacy Metric Fix ===\n")
    
    config = Config()
    config.max_events = 800
    config.bootstrap_iters = 200
    config.dataset = "linear"
    config.algo = "memorypair" 
    config.accountant = "zcdp"
    config.gamma_bar = 1.5
    config.gamma_split = 0.7
    config.seed = 123
    config.delete_ratio = 5
    config.log_privacy_spend = True
    config.privacy_mode = True
    config.odometer_mode = "live"
    config.target_G = 1.0
    config.target_D = 1.0
    config.target_c = 1.0 
    config.target_C = 1.0
    config.target_lambda = 0.1
    config.target_PT = 1.0
    config.target_ST = 1.0
    
    # Test 1: With regret gate enabled (original broken behavior)
    print("Test 1: Regret gate enabled (original behavior)")
    with tempfile.TemporaryDirectory() as temp_dir:
        config.output_dir = temp_dir
        config.disable_regret_gate = False
        
        runner = ExperimentRunner(config)
        result = runner.run_one_seed(config.seed)
        
        events_df = pd.read_csv(result.csv_path)
        delete_events = events_df[events_df['op'] == 'delete']
        
        if not delete_events.empty:
            final_rho = events_df['rho_spent'].iloc[-1]
            print(f"  Delete events: {len(delete_events)}")
            print(f"  Final rho_spent: {final_rho}")
            
            if final_rho == 0.0:
                print("  ✓ Confirmed: rho_spent stays 0 with regret gate (as expected)")
            else:
                print("  ✗ Unexpected: rho_spent is non-zero with regret gate")
        else:
            print("  No delete events found")
    
    print()
    
    # Test 2: With regret gate disabled (fixed behavior)  
    print("Test 2: Regret gate disabled (fixed behavior)")
    with tempfile.TemporaryDirectory() as temp_dir:
        config.output_dir = temp_dir
        config.disable_regret_gate = True
        
        runner = ExperimentRunner(config)
        result = runner.run_one_seed(config.seed)
        
        events_df = pd.read_csv(result.csv_path)
        delete_events = events_df[events_df['op'] == 'delete']
        
        if not delete_events.empty:
            final_rho = events_df['rho_spent'].iloc[-1]
            successful_deletes = delete_events[delete_events['rho_spent'] > 0]
            print(f"  Delete events: {len(delete_events)}")
            print(f"  Successful deletes (rho > 0): {len(successful_deletes)}")
            print(f"  Final rho_spent: {final_rho}")
            
            if final_rho > 0:
                print("  ✓ SUCCESS: rho_spent increments with regret gate disabled")
                
                # Test rho_util computation
                rho_total = 1.0  # from config
                expected_rho_util = final_rho / rho_total
                print(f"  Expected rho_util: {expected_rho_util}")
                
                return True
            else:
                print("  ✗ FAILED: rho_spent still 0 even with regret gate disabled")
                return False
        else:
            print("  No delete events found")
            return False

if __name__ == "__main__":
    success = test_rho_fix()
    if success:
        print("\n🎉 rho_spent privacy metric fix VERIFIED!")
    else:
        print("\n❌ rho_spent privacy metric fix FAILED!")
        sys.exit(1)