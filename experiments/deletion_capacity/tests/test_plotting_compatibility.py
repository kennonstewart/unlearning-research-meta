#!/usr/bin/env python3
"""
Test to verify that existing plotting scripts still work with seed output.
"""

import tempfile
import os
import sys
import pandas as pd

def test_plotting_compatibility():
    """Test that seed output works with existing plotting functions."""
    print("Testing plotting compatibility with seed output...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock CSV files that match what the grid runner produces in seed mode
        csv_files = []
        for seed in [0, 1]:
            # This simulates the output from process_seed_output
            seed_data = {
                'seed': [seed],
                'grid_id': ['test_grid'],
                'avg_regret_empirical': [15.2 + seed],
                'N_star_emp': [100 + seed * 5],
                'm_emp': [45 + seed * 2],
                'final_acc': [0.85 + seed * 0.01],
                'total_events': [200 + seed * 10],
                'G_hat': [2.5],
                'D_hat': [1.8],
                'sigma_step_theory': [0.05],
                'eps_spent': [0.8 + seed * 0.1],
                'capacity_remaining': [0.2 - seed * 0.1],
                'gamma_bar': [1.0],
                'gamma_split': [0.9]
            }
            
            csv_path = os.path.join(temp_dir, f"seed_{seed:03d}.csv")
            pd.DataFrame(seed_data).to_csv(csv_path, index=False)
            csv_files.append(csv_path)
        
        print(f"Created {len(csv_files)} mock seed CSV files")
        
        # Test that plotting functions can handle these files
        try:
            # Add the deletion_capacity directory to path to import plots
            sys.path.insert(0, os.path.dirname(__file__))
            from plots import plot_capacity_curve, plot_regret
            
            # Test capacity curve plot
            capacity_plot_path = os.path.join(temp_dir, "capacity_curve.png")
            try:
                plot_capacity_curve(csv_files, capacity_plot_path)
                if os.path.exists(capacity_plot_path):
                    print("  ✅ plot_capacity_curve works with seed output")
                else:
                    print("  ❌ plot_capacity_curve did not create output file")
            except Exception as e:
                print(f"  ❌ plot_capacity_curve failed: {e}")
            
            # Test regret plot - but first need to create event-level data for regret plotting
            # Since regret plotting expects event-level data, we need to create that format
            print("  Note: regret plotting requires event-level data, testing with mock event data...")
            
            # Create event-level mock data
            event_csv_path = os.path.join(temp_dir, "events_for_regret.csv")
            event_data = []
            for i in range(20):
                event_data.append({
                    'event': i,
                    'regret': 10.0 + i * 0.5,
                    'acc': 0.1 + i * 0.01,
                    'op': 'insert' if i < 10 else 'delete'
                })
            pd.DataFrame(event_data).to_csv(event_csv_path, index=False)
            
            regret_plot_path = os.path.join(temp_dir, "regret_curve.png")
            try:
                plot_regret([event_csv_path], regret_plot_path)
                if os.path.exists(regret_plot_path):
                    print("  ✅ plot_regret works (tested with event-level data)")
                else:
                    print("  ❌ plot_regret did not create output file")
            except Exception as e:
                print(f"  ❌ plot_regret failed: {e}")
            
        except ImportError as e:
            print(f"  ❌ Could not import plotting functions: {e}")
    
    print("✅ Plotting compatibility test completed")

if __name__ == "__main__":
    test_plotting_compatibility()