#!/usr/bin/env python3
"""
Demo script showing the exp_engine functionality end-to-end.

This demonstrates:
1. Writing seed and event data to Parquet
2. Content-addressed hashing for deduplication
3. DuckDB views for instant querying
4. Integration with existing experiment workflows
"""

import os
import sys
import tempfile
from pathlib import Path

# Add exp_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import write_seed_rows, write_event_rows, attach_grid_id
from engine.duck import create_connection_and_views, query_seeds, query_events


def demo_basic_functionality():
    """Demonstrate basic exp_engine functionality."""
    print("=== Exp_engine Demo: Basic Functionality ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}")
        
        # 1. Create sample experiment parameters
        params = {
            "algo": "memorypair",
            "accountant": "zcdp", 
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
            "target_PT": 0.8,
            "target_ST": 0.9,
            "delete_ratio": 10,
            "rho_total": 1.0
        }
        
        # Add grid_id via content-addressed hashing
        params_with_grid = attach_grid_id(params)
        grid_id = params_with_grid["grid_id"]
        print(f"Generated grid_id: {grid_id}")
        print(f"Parameters: {params_with_grid}\n")
        
        # 2. Create sample seed-level data
        seed_data = [
            {
                **params_with_grid,
                "seed": 1,
                "avg_regret_empirical": 0.15,
                "N_star_emp": 100,
                "m_emp": 10,
                "final_acc": 0.85
            },
            {
                **params_with_grid,
                "seed": 2,
                "avg_regret_empirical": 0.12,
                "N_star_emp": 105,
                "m_emp": 12,
                "final_acc": 0.87
            },
            {
                **params_with_grid,
                "seed": 3,
                "avg_regret_empirical": 0.18,
                "N_star_emp": 98,
                "m_emp": 8,
                "final_acc": 0.83
            }
        ]
        
        # 3. Write seed data to Parquet
        seeds_path = write_seed_rows(seed_data, tmpdir, params)
        print(f"✓ Wrote seed data to: {seeds_path}")
        
        # 4. Create sample event-level data  
        event_data = []
        for seed in [1, 2, 3]:
            for i in range(5):
                event_data.append({
                    **params_with_grid,
                    "seed": seed,
                    "event_id": i,
                    "op": "insert" if i < 3 else "delete",
                    "regret": 0.05 + 0.01 * i,
                    "acc": 0.8 + 0.01 * i,
                    "eps_spent": 0.1 * i
                })
        
        # 5. Write event data to Parquet
        events_path = write_event_rows(event_data, tmpdir, params)
        print(f"✓ Wrote event data to: {events_path}")
        
        # 6. Create DuckDB views for instant querying
        print("\n--- Creating DuckDB Views ---")
        conn = create_connection_and_views(tmpdir)
        print("✓ Created DuckDB connection and views")
        
        # 7. Query the data
        print("\n--- Querying Data ---")
        
        # Query all seeds
        seeds_df = query_seeds(conn)
        print(f"Seeds query returned {len(seeds_df)} rows")
        print("Sample seed data:")
        print(seeds_df[["seed", "avg_regret_empirical", "N_star_emp", "m_emp"]].to_string(index=False))
        
        # Query events for a specific seed
        events_df = query_events(conn, "seed = 1", limit=5)
        print(f"\nEvents for seed 1 (first 5):")
        print(events_df[["seed", "op", "regret", "acc"]].to_string(index=False))
        
        # Custom aggregation query
        print(f"\n--- Custom Aggregation ---")
        agg_results = conn.execute("""
            SELECT 
                grid_id,
                COUNT(*) as num_seeds,
                AVG(avg_regret_empirical) as mean_regret,
                MIN(avg_regret_empirical) as min_regret,
                MAX(avg_regret_empirical) as max_regret
            FROM seeds
            GROUP BY grid_id
        """).fetchall()
        
        for row in agg_results:
            print(f"Grid {row[0]}: {row[1]} seeds, mean regret = {row[2]:.4f} (range: {row[3]:.4f} - {row[4]:.4f})")
        
        print(f"\n✓ Demo completed successfully!")


def demo_content_addressed_hashing():
    """Demonstrate content-addressed hashing for deduplication."""
    print("\n=== Exp_engine Demo: Content-Addressed Hashing ===\n")
    
    # Show that identical parameter sets get the same grid_id
    params1 = {
        "algo": "memorypair",
        "gamma_bar": 1.0,
        "accountant": "zcdp",
        "seed": 42,  # volatile - should be ignored
        "base_out": "/tmp/results"  # volatile - should be ignored
    }
    
    params2 = {
        "algo": "memorypair", 
        "gamma_bar": 1.0,
        "accountant": "zcdp",
        "seed": 99,  # different volatile value
        "base_out": "/different/path"  # different volatile value
    }
    
    params3 = {
        "algo": "memorypair",
        "gamma_bar": 0.5,  # different non-volatile value
        "accountant": "zcdp"
    }
    
    grid1 = attach_grid_id(params1)
    grid2 = attach_grid_id(params2)
    grid3 = attach_grid_id(params3)
    
    print(f"Params1 grid_id: {grid1['grid_id']}")
    print(f"Params2 grid_id: {grid2['grid_id']} (should be same as params1)")
    print(f"Params3 grid_id: {grid3['grid_id']} (should be different)")
    
    print(f"\nDeduplication works: {grid1['grid_id'] == grid2['grid_id']}")
    print(f"Different params have different hashes: {grid1['grid_id'] != grid3['grid_id']}")


def demo_file_structure():
    """Show the file structure created by exp_engine."""
    print("\n=== Exp_engine Demo: File Structure ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        params = {"algo": "memorypair", "accountant": "zcdp", "gamma_bar": 1.0}
        seed_data = [{"seed": 1, "avg_regret_empirical": 0.1, **params}]
        event_data = [{"seed": 1, "op": "insert", "regret": 0.05, **params}]
        
        # Write data
        write_seed_rows(seed_data, tmpdir, params)
        write_event_rows(event_data, tmpdir, params)
        
        # Show directory structure
        print(f"Created file structure in {tmpdir}:")
        
        def show_tree(path, prefix=""):
            try:
                items = sorted(os.listdir(path))
                for i, item in enumerate(items):
                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    print(f"{prefix}{current_prefix}{item}")
                    
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        show_tree(item_path, next_prefix)
            except PermissionError:
                pass
        
        show_tree(tmpdir)


if __name__ == "__main__":
    print("Exp_engine End-to-End Demo")
    print("=" * 50)
    
    try:
        demo_basic_functionality()
        demo_content_addressed_hashing()
        demo_file_structure()
        
        print("\n" + "=" * 50)
        print("✓ All demos completed successfully!")
        print("\nThe exp_engine provides:")
        print("  • Content-addressed hashing for grid deduplication")
        print("  • HIVE-partitioned Parquet datasets")
        print("  • Instant DuckDB views for querying")
        print("  • Non-invasive integration with existing runners")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)