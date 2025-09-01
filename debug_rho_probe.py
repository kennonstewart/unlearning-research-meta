#!/usr/bin/env python3
"""
Minimal probe script to reproduce the rho_spent issue.
Run a tiny deletion capacity experiment and check if rho_spent increments.
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

def create_minimal_config():
    """Create minimal config for reproduction."""
    config = Config()
    
    # Make experiment small and fast
    config.max_events = 1000  # Larger to have events left after bootstrap
    config.bootstrap_iters = 200  # Smaller bootstrap to leave room for deletes
    config.dataset = "linear"
    config.algo = "memorypair" 
    config.accountant = "zcdp"
    config.gamma_bar = 1.5
    config.gamma_split = 0.7
    config.seed = 42
    config.delete_ratio = 5  # Every 5th event is delete after learning phase
    
    # Enable privacy logging
    config.log_privacy_spend = True
    config.privacy_mode = True
    config.odometer_mode = "live"
    
    # Enable debugging and bypass regret gate
    config.disable_regret_gate = True
    
    # Small theoretical bounds for fast calibration
    config.target_G = 1.0
    config.target_D = 1.0
    config.target_c = 1.0 
    config.target_C = 1.0
    config.target_lambda = 0.1
    config.target_PT = 1.0  # Set path terms too
    config.target_ST = 1.0
    
    return config

def debug_single_run():
    """Run one tiny experiment and check rho progression."""
    print("=== Debugging rho_spent propagation ===")
    
    # Create temporary output directory  
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_minimal_config()
        config.output_dir = temp_dir
        
        print(f"Running experiment with config:")
        print(f"  max_events: {config.max_events}")
        print(f"  accountant: {config.accountant}")
        print(f"  gamma_bar: {config.gamma_bar}")
        print(f"  gamma_split: {config.gamma_split}")
        print(f"  delete_ratio: {config.delete_ratio}")
        print(f"  output_dir: {config.output_dir}")
        
        try:
            # Run the experiment
            runner = ExperimentRunner(config)
            result = runner.run_one_seed(config.seed)
            
            print(f"\nExperiment completed. Result: {result}")
            
            # Look for event logs  
            print(f"CSV result path: {result.csv_path}")
            if os.path.exists(result.csv_path):
                print(f"CSV file exists, reading...")
                events_df = pd.read_csv(result.csv_path)
                print(f"Event log shape: {events_df.shape}")
                print(f"Event log columns: {list(events_df.columns)}")
                
                # Check for privacy columns
                privacy_cols = [col for col in events_df.columns if 'rho' in col.lower() or 'privacy' in col.lower() or 'sigma' in col.lower()]
                print(f"Privacy-related columns: {privacy_cols}")
                
                if privacy_cols:
                    print(f"\nPrivacy metrics sample:")
                    display_cols = ['event', 'op'] + privacy_cols
                    available_cols = [col for col in display_cols if col in events_df.columns]
                    sample_df = events_df[available_cols].head(20)
                    print(sample_df.to_string(index=False))
                    
                    # Check delete events specifically
                    delete_events = events_df[events_df['op'] == 'delete'] if 'op' in events_df.columns else pd.DataFrame()
                    if not delete_events.empty:
                        print(f"\nDelete events privacy metrics:")
                        delete_sample = delete_events[available_cols].head(10)
                        print(delete_sample.to_string(index=False))
                        
                        # Check final values
                        if 'rho_spent' in events_df.columns:
                            final_rho = events_df['rho_spent'].iloc[-1]
                            print(f"\nFinal rho_spent value: {final_rho}")
                            
                            # Check progression in delete events
                            delete_rho_values = delete_events['rho_spent'].tolist()
                            print(f"rho_spent progression in delete events: {delete_rho_values}")
                    else:
                        print("\nNo delete events found!")
                else:
                    print("\nNo privacy-related columns found!")
            else:
                print(f"CSV file does not exist at {result.csv_path}")
            
            # Also look for event logs in common locations
            event_files = list(Path(temp_dir).glob("**/events.csv"))
            if not event_files:
                event_files = list(Path(temp_dir).glob("**/*events*.csv"))
            if not event_files:
                event_files = list(Path(".").glob("**/events.csv"))
            if not event_files:
                event_files = list(Path("results").glob("**/*events*.csv"))
            
            print(f"Found event files: {event_files}")
            
            if event_files:
                # Load events and check rho progression
                events_df = pd.read_csv(event_files[0])
                print(f"\nEvent log shape: {events_df.shape}")
                print(f"Event log columns: {list(events_df.columns)}")
                
                # Check for privacy columns
                privacy_cols = [col for col in events_df.columns if 'rho' in col.lower() or 'privacy' in col.lower() or 'sigma' in col.lower()]
                print(f"Privacy-related columns: {privacy_cols}")
                
                if privacy_cols:
                    print(f"\nPrivacy metrics sample:")
                    display_cols = ['event', 'op'] + privacy_cols
                    available_cols = [col for col in display_cols if col in events_df.columns]
                    sample_df = events_df[available_cols].head(20)
                    print(sample_df.to_string(index=False))
                    
                    # Check delete events specifically
                    delete_events = events_df[events_df['op'] == 'delete'] if 'op' in events_df.columns else pd.DataFrame()
                    if not delete_events.empty:
                        print(f"\nDelete events privacy metrics:")
                        delete_sample = delete_events[available_cols].head(10)
                        print(delete_sample.to_string(index=False))
                        
                        # Check final values
                        if 'rho_spent' in events_df.columns:
                            final_rho = events_df['rho_spent'].iloc[-1]
                            print(f"\nFinal rho_spent value: {final_rho}")
                            
                            # Check progression in delete events
                            delete_rho_values = delete_events['rho_spent'].tolist()
                            print(f"rho_spent progression in delete events: {delete_rho_values}")
                    else:
                        print("\nNo delete events found!")
                else:
                    print("\nNo privacy-related columns found!")
            else:
                print("\nNo event log files found!")
                
        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_single_run()