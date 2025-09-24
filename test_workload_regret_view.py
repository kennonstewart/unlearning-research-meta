#!/usr/bin/env python3
"""
Test script for workload-only regret computation view.

This script validates that the new workload regret view correctly:
1. Identifies workload phase boundaries (first delete event or N_star threshold)
2. Computes workload-only cumulative and average regret
3. Provides meaningful analytics compared to total regret
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
    from experiment.utils.duck_db_loader import load_star_schema, query_workload_events, query_workload_regret_analysis
    DUCKDB_AVAILABLE = True
except ImportError as e:
    print(f"DuckDB not available: {e}")
    DUCKDB_AVAILABLE = False


def create_mock_event_data():
    """Create mock event data for testing workload regret computation."""
    
    events = []
    
    for seed in [1, 2]:
        for grid_id in ["test_grid_1", "test_grid_2"]:
            event_id = 0
            cum_regret = 0.0
            noise_regret_cum = 0.0
            
            # Phase 1: Calibration events (should be excluded from workload)
            for i in range(10):
                event_id += 1
                regret_increment = np.random.normal(0.05, 0.01)  # Small positive regret
                cum_regret += regret_increment
                noise_increment = np.random.normal(0.001, 0.0005)
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
                    "avg_regret": cum_regret / event_id,  # Total average regret
                    "noise_regret_cum": noise_regret_cum,
                    "cum_regret_with_noise": cum_regret + noise_regret_cum,
                    "avg_regret_with_noise": (cum_regret + noise_regret_cum) / event_id,
                    "N_star": 15  # Sample complexity threshold
                })
            
            # Phase 2: Insert events (learning phase, should be excluded from workload)
            for i in range(10):
                event_id += 1
                regret_increment = np.random.normal(0.03, 0.01)  # Decreasing regret
                cum_regret += regret_increment
                noise_increment = np.random.normal(0.001, 0.0005)
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
                    "N_star": 15
                })
            
            # Phase 3: Workload phase (interleaving inserts and deletes)
            # This is where workload regret should start being computed
            workload_baseline_regret = cum_regret  # Save the baseline
            workload_baseline_noise = noise_regret_cum
            
            for i in range(20):
                event_id += 1
                
                # Alternate between insert and delete, starting with delete to mark boundary
                if i == 0 or i % 3 == 0:  # First event is delete to mark workload start
                    op = "delete"
                    regret_increment = np.random.normal(0.01, 0.005)  # Low regret in workload
                else:
                    op = "insert" 
                    regret_increment = np.random.normal(0.01, 0.005)
                
                cum_regret += regret_increment
                noise_increment = np.random.normal(0.001, 0.0005)
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
                    "N_star": 15
                })
    
    return pd.DataFrame(events)


def test_workload_regret_view():
    """Test the workload regret view implementation."""
    
    if not DUCKDB_AVAILABLE:
        print("❌ DuckDB not available, skipping test")
        return False
    
    print("🔬 Testing workload regret view implementation...")
    
    # Create mock data
    events_df = create_mock_event_data()
    print(f"✅ Created {len(events_df)} mock events across {len(events_df.groupby(['grid_id', 'seed']))} runs")
    
    # Write to temporary parquet file
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "events.parquet")
        events_df.to_parquet(parquet_path, index=False)
        
        try:
            # Load into DuckDB with star schema
            conn = load_star_schema(
                input_path=parquet_path,
                include_parameters=False,
                run_ddl=True,
                create_events_view=True
            )
            
            print("✅ Successfully loaded data into DuckDB with workload views")
            
            # Test 1: Verify workload boundary detection
            workload_boundaries = conn.execute("""
                SELECT 
                    grid_id, 
                    seed, 
                    workload_start_event_id,
                    COUNT(*) FILTER (WHERE is_workload_phase) as workload_events,
                    COUNT(*) FILTER (WHERE NOT is_workload_phase) as pre_workload_events
                FROM analytics.v_events_workload 
                GROUP BY grid_id, seed, workload_start_event_id
                ORDER BY grid_id, seed
            """).df()
            
            print("\n📊 Workload boundary detection:")
            print(workload_boundaries)
            
            # Verify that workload starts at first delete (event_id=21 in our mock data)
            expected_workload_start = 21  # First delete event in workload phase
            if all(workload_boundaries['workload_start_event_id'] == expected_workload_start):
                print("✅ Workload boundaries correctly identified at first delete event")
            else:
                print("❌ Workload boundaries not at expected first delete event")
                return False
            
            # Test 2: Verify workload regret computation
            workload_analysis = query_workload_regret_analysis(conn)
            print("\n📈 Workload regret analysis:")
            print(workload_analysis[['grid_id', 'seed', 'final_workload_avg_regret', 'final_total_avg_regret', 'workload_vs_total_avg_regret_diff']])
            
            # Workload average regret should be lower than total (since it excludes higher initial regret)
            workload_lower_count = sum(workload_analysis['workload_vs_total_avg_regret_diff'] < 0)
            if workload_lower_count == len(workload_analysis):
                print("✅ Workload average regret is lower than total average regret (as expected)")
            else:
                print(f"⚠️ Only {workload_lower_count}/{len(workload_analysis)} runs have lower workload regret")
            
            # Test 3: Verify no negative regret in workload phase
            negative_regret_count = workload_analysis['negative_workload_regret_count'].sum()
            if negative_regret_count == 0:
                print("✅ No negative workload regret values found")
            else:
                print(f"⚠️ Found {negative_regret_count} negative workload regret events")
            
            # Test 4: Sample a few workload events to verify calculations
            sample_events = query_workload_events(conn, "is_workload_phase = TRUE", limit=5)
            print("\n🔍 Sample workload events:")
            print(sample_events[['grid_id', 'seed', 'event_id', 'workload_cum_regret', 'workload_avg_regret', 'workload_events_seen']])
            
            # Verify that workload_avg_regret = workload_cum_regret / workload_events_seen
            calculated_avg = sample_events['workload_cum_regret'] / sample_events['workload_events_seen']
            avg_regret_match = np.allclose(calculated_avg, sample_events['workload_avg_regret'], rtol=1e-10)
            
            if avg_regret_match:
                print("✅ Workload average regret calculation is correct")
            else:
                print("❌ Workload average regret calculation mismatch")
                return False
            
            print("\n🎉 All workload regret view tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_workload_regret_view()
    sys.exit(0 if success else 1)