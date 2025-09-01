#!/usr/bin/env python3
"""
Demo script showing exp_engine Parquet + DuckDB integration.

Usage:
    python demo_parquet_duckdb.py results_parquet
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.deletion_capacity.duck_helpers import connect, q_seeds, q_events


def demo_basic_queries(base_out="results_parquet"):
    """Demonstrate basic DuckDB queries on Parquet datasets."""
    print(f"=== DuckDB Demo: {base_out} ===\n")
    
    # Connect to the Parquet datasets
    conn = connect(base_out)
    
    print("1. Basic seed summary query:")
    seeds = q_seeds(conn, "avg_regret_empirical IS NOT NULL", limit=5)
    if len(seeds) > 0:
        print(seeds[['grid_id', 'seed', 'avg_regret_empirical', 'N_star_emp', 'accountant']].to_string(index=False))
    else:
        print("   No seed data found")
    
    print("\n2. Accountant comparison:")
    try:
        accountant_stats = conn.execute("""
            SELECT accountant, 
                   COUNT(*) as num_seeds,
                   AVG(avg_regret_empirical) as mean_regret,
                   AVG(rho_spent_final) as mean_rho_spent
            FROM seeds 
            WHERE avg_regret_empirical IS NOT NULL
            GROUP BY accountant
        """).df()
        print(accountant_stats.to_string(index=False))
    except Exception as e:
        print(f"   Could not compute accountant stats: {e}")
    
    print("\n3. Privacy budget utilization:")
    try:
        privacy_stats = conn.execute("""
            SELECT grid_id,
                   AVG(rho_spent_final) as avg_rho_spent,
                   AVG(rho_util) as avg_rho_utilization,
                   COUNT(*) as num_seeds
            FROM seeds 
            WHERE rho_spent_final IS NOT NULL
            GROUP BY grid_id
            ORDER BY avg_rho_utilization DESC
            LIMIT 3
        """).df()
        print(privacy_stats.to_string(index=False))
    except Exception as e:
        print(f"   Could not compute privacy stats: {e}")
    
    # Try event queries if available
    try:
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if event_count > 0:
            print(f"\n4. Event data summary ({event_count} total events):")
            event_stats = conn.execute("""
                SELECT op, 
                       COUNT(*) as count,
                       AVG(regret) as avg_regret,
                       AVG(rho_spent) as avg_rho_spent
                FROM events 
                GROUP BY op
            """).df()
            print(event_stats.to_string(index=False))
        else:
            print("\n4. No event data available")
    except Exception:
        print("\n4. Events table not available")
    
    print(f"\n✓ Demo completed for {base_out}")


def demo_advanced_queries(base_out="results_parquet"):
    """Show more advanced DuckDB queries."""
    print(f"\n=== Advanced Queries: {base_out} ===\n")
    
    conn = connect(base_out)
    
    print("1. Grid performance ranking:")
    try:
        grid_ranking = conn.execute("""
            SELECT grid_id,
                   accountant,
                   gamma_bar,
                   gamma_split,
                   COUNT(*) as num_seeds,
                   AVG(avg_regret_empirical) as mean_regret,
                   MIN(avg_regret_empirical) as best_regret,
                   AVG(rho_util) as mean_privacy_util
            FROM seeds
            WHERE avg_regret_empirical IS NOT NULL
            GROUP BY grid_id, accountant, gamma_bar, gamma_split
            ORDER BY mean_regret ASC
            LIMIT 5
        """).df()
        print(grid_ranking.to_string(index=False))
    except Exception as e:
        print(f"   Could not compute grid ranking: {e}")
    
    print("\n2. Gamma adherence analysis:")
    try:
        gamma_analysis = conn.execute("""
            SELECT accountant,
                   AVG(CASE WHEN gamma_pass_overall THEN 1.0 ELSE 0.0 END) as pass_rate,
                   AVG(avg_regret_empirical) as mean_regret,
                   COUNT(*) as total_seeds
            FROM seeds
            WHERE gamma_pass_overall IS NOT NULL
            GROUP BY accountant
        """).df()
        print(gamma_analysis.to_string(index=False))
    except Exception as e:
        print(f"   Could not compute gamma analysis: {e}")


if __name__ == "__main__":
    base_out = sys.argv[1] if len(sys.argv) > 1 else "results_parquet"
    
    if not os.path.exists(base_out):
        print(f"Error: Directory {base_out} does not exist")
        print("Usage: python demo_parquet_duckdb.py <parquet_directory>")
        sys.exit(1)
    
    demo_basic_queries(base_out)
    demo_advanced_queries(base_out)