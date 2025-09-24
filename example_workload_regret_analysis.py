#!/usr/bin/env python3
"""
Example: Workload-Only Regret Analysis

This script demonstrates how to use the new workload regret views to compute
theoretically-aligned average regret that excludes calibration and learning phases.

Usage:
    python example_workload_regret_analysis.py [parquet_events_path]
    
If no path is provided, it will create synthetic example data.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add experiment utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiment"))

try:
    import duckdb
    from experiment.utils.duck_db_loader import load_star_schema, query_workload_regret_analysis
    from exp_engine.engine.duck import create_connection_and_views, query_workload_regret_analysis as query_workload_regret_exp
    DUCKDB_AVAILABLE = True
except ImportError as e:
    print(f"DuckDB not available: {e}")
    DUCKDB_AVAILABLE = False


def create_example_data():
    """Create realistic example data showing the difference between total and workload regret."""
    
    np.random.seed(42)  # Reproducible results
    events = []
    
    for seed in [1, 2, 3]:
        for grid_id in ["low_regret", "high_regret"]:
            event_id = 0
            cum_regret = 0.0
            noise_regret_cum = 0.0
            
            # Set regret characteristics by grid
            if grid_id == "low_regret":
                calibration_regret = 0.08  # High initial regret
                workload_regret = 0.01     # Low workload regret  
            else:
                calibration_regret = 0.15  # Very high initial regret
                workload_regret = 0.03     # Moderate workload regret
            
            # Phase 1: Calibration (high regret as algorithm learns basics)
            print(f"Generating calibration phase for {grid_id} seed {seed}...")
            for i in range(20):
                event_id += 1
                regret_increment = np.random.normal(calibration_regret, 0.02)  
                cum_regret += regret_increment
                noise_increment = np.random.normal(0.002, 0.0005)
                noise_regret_cum += noise_increment
                
                events.append({
                    "grid_id": grid_id,
                    "seed": seed,
                    "event_id": event_id,
                    "event": event_id,
                    "op": "calibrate",
                    "event_type": "calibrate",
                    "regret": regret_increment,
                    "cum_regret": cum_regret,
                    "avg_regret": cum_regret / event_id,
                    "noise_regret_cum": noise_regret_cum,
                    "cum_regret_with_noise": cum_regret + noise_regret_cum,
                    "avg_regret_with_noise": (cum_regret + noise_regret_cum) / event_id,
                    "N_star": 30  # Sample complexity threshold
                })
            
            # Phase 2: Learning/Warmup (decreasing regret as algorithm improves)
            print(f"Generating learning phase for {grid_id} seed {seed}...")
            for i in range(15):
                event_id += 1
                # Regret decreases as algorithm learns
                learning_regret = max(0.005, calibration_regret * (0.7 ** (i/5)))
                regret_increment = np.random.normal(learning_regret, 0.01)
                cum_regret += regret_increment
                noise_increment = np.random.normal(0.002, 0.0005)
                noise_regret_cum += noise_increment
                
                events.append({
                    "grid_id": grid_id,
                    "seed": seed,
                    "event_id": event_id,
                    "event": event_id,
                    "op": "insert",
                    "event_type": "insert", 
                    "regret": regret_increment,
                    "cum_regret": cum_regret,
                    "avg_regret": cum_regret / event_id,
                    "noise_regret_cum": noise_regret_cum,
                    "cum_regret_with_noise": cum_regret + noise_regret_cum,
                    "avg_regret_with_noise": (cum_regret + noise_regret_cum) / event_id,
                    "N_star": 30
                })
            
            # Phase 3: Workload (steady-state performance)
            print(f"Generating workload phase for {grid_id} seed {seed}...")
            for i in range(30):
                event_id += 1
                
                # Mark workload phase start with first delete
                if i == 0:
                    op = "delete"
                elif i % 4 == 0:  # Every 4th event is a delete
                    op = "delete"
                else:
                    op = "insert"
                
                regret_increment = np.random.normal(workload_regret, 0.005)
                cum_regret += regret_increment
                noise_increment = np.random.normal(0.002, 0.0005)
                noise_regret_cum += noise_increment
                
                events.append({
                    "grid_id": grid_id,
                    "seed": seed,
                    "event_id": event_id,
                    "event": event_id,
                    "op": op,
                    "event_type": op,
                    "regret": regret_increment,
                    "cum_regret": cum_regret,
                    "avg_regret": cum_regret / event_id,
                    "noise_regret_cum": noise_regret_cum,
                    "cum_regret_with_noise": cum_regret + noise_regret_cum,
                    "avg_regret_with_noise": (cum_regret + noise_regret_cum) / event_id,
                    "N_star": 30
                })
    
    return pd.DataFrame(events)


def analyze_workload_vs_total_regret(events_path=None):
    """Compare workload-only vs total regret analysis."""
    
    if not DUCKDB_AVAILABLE:
        print("❌ DuckDB not available, cannot run analysis")
        return
    
    print("📊 Workload vs Total Regret Analysis")
    print("=" * 50)
    
    if events_path is None:
        print("📝 Creating synthetic example data...")
        events_df = create_example_data()
        
        # Write to temporary parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = os.path.join(tmpdir, "events.parquet")
            events_df.to_parquet(parquet_path, index=False)
            events_path = parquet_path
            
            return _run_analysis(events_path)
    else:
        return _run_analysis(events_path)


def _run_analysis(events_path):
    """Run the actual analysis on the events data."""
    
    # Load data into DuckDB
    print(f"📂 Loading data from {events_path}...")
    conn = load_star_schema(
        input_path=events_path,
        include_parameters=False,
        run_ddl=True,
        create_events_view=True
    )
    
    # Get workload regret analysis
    workload_analysis = query_workload_regret_analysis(conn)
    
    print("\n📈 Workload vs Total Regret Comparison:")
    print("-" * 80)
    
    # Format and display results
    display_df = workload_analysis[[
        'grid_id', 'seed', 
        'final_total_avg_regret', 
        'final_workload_avg_regret',
        'workload_vs_total_avg_regret_diff',
        'workload_events',
        'pre_workload_events'
    ]].round(6)
    
    print(display_df.to_string(index=False))
    
    print("\n📊 Summary Statistics:")
    print("-" * 40)
    
    avg_total_regret = workload_analysis['final_total_avg_regret'].mean()
    avg_workload_regret = workload_analysis['final_workload_avg_regret'].mean()
    avg_improvement = workload_analysis['workload_vs_total_avg_regret_diff'].mean()
    improvement_pct = (abs(avg_improvement) / avg_total_regret) * 100
    
    print(f"Average Total Regret:          {avg_total_regret:.6f}")
    print(f"Average Workload Regret:       {avg_workload_regret:.6f}")
    print(f"Average Improvement:           {avg_improvement:.6f}")
    print(f"Relative Improvement:          {improvement_pct:.1f}%")
    
    # Show per-grid statistics
    print("\n📋 Per-Grid Analysis:")
    print("-" * 40)
    
    for grid_id in workload_analysis['grid_id'].unique():
        grid_data = workload_analysis[workload_analysis['grid_id'] == grid_id]
        grid_total = grid_data['final_total_avg_regret'].mean()
        grid_workload = grid_data['final_workload_avg_regret'].mean()
        grid_improvement = grid_data['workload_vs_total_avg_regret_diff'].mean()
        grid_improvement_pct = (abs(grid_improvement) / grid_total) * 100
        
        print(f"{grid_id}:")
        print(f"  Total Regret:     {grid_total:.6f}")
        print(f"  Workload Regret:  {grid_workload:.6f}")
        print(f"  Improvement:      {grid_improvement:.6f} ({grid_improvement_pct:.1f}%)")
    
    print("\n✅ Analysis complete!")
    print("\n💡 Key Insights:")
    print("   • Workload-only regret excludes high initial calibration/learning regret")
    print("   • This provides a more accurate measure of steady-state performance")
    print("   • Aligns with theoretical regret guarantees that apply post-warmup")
    print("   • Enables better comparison of algorithm performance across different settings")
    
    return workload_analysis


if __name__ == "__main__":
    if len(sys.argv) > 1:
        events_path = sys.argv[1]
        if not os.path.exists(events_path):
            print(f"❌ Events file not found: {events_path}")
            sys.exit(1)
    else:
        events_path = None
    
    try:
        analyze_workload_vs_total_regret(events_path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)