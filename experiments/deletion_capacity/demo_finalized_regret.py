#!/usr/bin/env python3
"""
Demo script for finalized regret calculation and reporting spec.

Demonstrates:
1. Non-negative, comparator-based regret computation
2. Privacy metrics on delete events
3. Parquet-only data pipeline 
4. DuckDB regret analysis views
"""

import os
import sys
import tempfile
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../code"))
sys.path.insert(0, os.path.dirname(__file__))

from memory_pair.src.comparators import StaticOracle, RollingOracle
from config import Config
from exp_integration import build_params_from_config, write_seed_summary_parquet, write_event_rows_parquet

try:
    from exp_engine.engine.duck import create_connection_and_views, query_regret_analysis, get_negative_regret_summary
    DUCKDB_AVAILABLE = True
except ImportError:
    print("Warning: DuckDB not available. Install with: pip install duckdb")
    DUCKDB_AVAILABLE = False


def demo_nonnegative_regret_enforcement():
    """Demonstrate non-negative regret enforcement."""
    print("=== Demo: Non-negative Regret Enforcement ===")
    
    # Create config with enforcement enabled
    cfg = Config(enforce_nonnegative_regret=True, lambda_reg=0.1)
    
    # Create static oracle
    oracle = StaticOracle(dim=3, lambda_reg=0.1, cfg=cfg)
    
    # Calibrate with some data
    calibration_data = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.0, 1.0, 0.0]), 0.5),
        (np.array([0.0, 0.0, 1.0]), 0.2),
    ]
    oracle.calibrate_with_initial_data(calibration_data)
    
    print(f"Oracle w*: {oracle.w_star_fixed}")
    
    # Test case where regret would be negative without enforcement
    x = np.array([0.1, 0.1, 0.1])
    y = 0.0
    current_theta = np.array([5.0, 5.0, 5.0])  # Large theta to potentially create negative regret
    
    result = oracle.update_regret_accounting(x, y, current_theta)
    
    print(f"Raw regret calculation result: {result['regret_increment']:.6f}")
    print(f"✓ Regret is non-negative: {result['regret_increment'] >= 0}")
    
    # Compare with enforcement disabled
    cfg_no_enforce = Config(enforce_nonnegative_regret=False, lambda_reg=0.1)
    oracle_no_enforce = StaticOracle(dim=3, lambda_reg=0.1, cfg=cfg_no_enforce)
    oracle_no_enforce.calibrate_with_initial_data(calibration_data)
    result_no_enforce = oracle_no_enforce.update_regret_accounting(x, y, current_theta)
    
    print(f"Without enforcement: {result_no_enforce['regret_increment']:.6f}")
    print()


def demo_regularization_consistency():
    """Demonstrate consistent λ regularization in regret computation."""
    print("=== Demo: Regularization Consistency ===")
    
    lambda_reg = 0.2
    cfg = Config(lambda_reg=lambda_reg)
    
    oracle = StaticOracle(dim=2, lambda_reg=lambda_reg, cfg=cfg)
    
    # Calibrate
    calibration_data = [
        (np.array([1.0, 0.0]), 1.0),
        (np.array([0.0, 1.0]), 0.5),
    ]
    oracle.calibrate_with_initial_data(calibration_data)
    
    # Test regret computation
    x = np.array([0.5, 0.5])
    y = 0.7
    current_theta = np.array([0.8, 0.3])
    
    print(f"λ regularization parameter: {lambda_reg}")
    print(f"Current θ: {current_theta}")
    print(f"Oracle w*: {oracle.w_star_fixed}")
    
    result = oracle.update_regret_accounting(x, y, current_theta)
    print(f"Regularized regret: {result['regret_increment']:.6f}")
    
    # Manual verification
    from memory_pair.src.metrics import loss_half_mse
    pred_current = float(current_theta @ x)
    pred_oracle = float(oracle.w_star_fixed @ x)
    
    loss_current = loss_half_mse(pred_current, y) + 0.5 * lambda_reg * float(np.dot(current_theta, current_theta))
    loss_oracle = loss_half_mse(pred_oracle, y) + 0.5 * lambda_reg * float(np.dot(oracle.w_star_fixed, oracle.w_star_fixed))
    
    manual_regret = loss_current - loss_oracle
    if cfg.enforce_nonnegative_regret:
        manual_regret = max(0.0, manual_regret)
    
    print(f"Manual calculation: {manual_regret:.6f}")
    print(f"✓ Consistent: {abs(result['regret_increment'] - manual_regret) < 1e-10}")
    print()


def demo_config_defaults():
    """Demonstrate finalized spec config defaults."""
    print("=== Demo: Finalized Spec Config Defaults ===")
    
    cfg = Config()
    
    print("Finalized spec defaults:")
    print(f"  enforce_nonnegative_regret: {cfg.enforce_nonnegative_regret}")
    print(f"  parquet_only_mode: {cfg.parquet_only_mode}")
    print(f"  regret_warmup_threshold: {cfg.regret_warmup_threshold}")
    print(f"  regret_comparator_mode: {cfg.regret_comparator_mode}")
    
    print("\nRelated regret settings:")
    print(f"  lambda_reg: {cfg.lambda_reg}")
    print(f"  comparator: {cfg.comparator}")
    print(f"  enable_oracle: {cfg.enable_oracle}")
    print()


def demo_parquet_and_duckdb():
    """Demonstrate Parquet-only pipeline and DuckDB regret analysis."""
    if not DUCKDB_AVAILABLE:
        print("=== Demo: Parquet and DuckDB (SKIPPED - DuckDB not available) ===")
        return
        
    print("=== Demo: Parquet Pipeline and DuckDB Regret Analysis ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}")
        
        # Create sample experiment data
        params_with_grid = {
            "grid_id": "finalized_spec_demo",
            "algo": "memorypair",
            "accountant": "zcdp",
            "gamma_bar": 1.0,
            "gamma_split": 0.5,
        }
        
        # Generate seed summary data
        seed_data = []
        for seed in range(3):
            seed_data.append({
                "seed": seed,
                "avg_regret_empirical": 0.1 + seed * 0.02,
                "final_regret": 5.0 + seed * 1.0,
                "regret_dynamic": 5.2 + seed * 1.1,
                "regret_static_term": 4.8 + seed * 0.9,
                "regret_path_term": 0.4 + seed * 0.2,
                "N_star_emp": 100 + seed * 10,
                "m_emp": 10 + seed * 2,
                "G_hat": 2.0,
                "D_hat": 1.5,
                "c_hat": 0.8,
                "C_hat": 2.2,
            })
        
        # Generate event data
        event_data = []
        for seed in range(2):  # Fewer seeds for events to keep demo manageable
            for event in range(20):
                op = "insert" if event < 15 else "delete"
                regret_val = max(0.0, np.random.normal(0.1, 0.05))  # Non-negative regret
                
                event_entry = {
                    "seed": seed,
                    "event": event,
                    "op": op,
                    "regret": regret_val,
                    "cum_regret": (event + 1) * 0.08 + seed * 0.5,
                    "regret_dynamic": regret_val if op == "insert" else None,
                    "regret_static_term": regret_val * 0.9 if op == "insert" else None,
                    "regret_path_term": regret_val * 0.1 if op == "insert" else None,
                    "acc": 0.85 + np.random.normal(0, 0.05),
                }
                
                # Add privacy metrics for delete events
                if op == "delete":
                    event_entry.update({
                        "rho_spent": 0.3 + event * 0.01,
                        "sigma_step": 0.1,
                        "m_capacity": 10 - event // 2,
                        "eps_spent": 0.5 + event * 0.02,
                    })
                
                event_data.append(event_entry)
        
        # Write to Parquet
        print("Writing seed summaries to Parquet...")
        write_seed_summary_parquet(seed_data, tmpdir, params_with_grid)
        
        print("Writing event data to Parquet...")
        write_event_rows_parquet(event_data, tmpdir, params_with_grid)
        
        # Create DuckDB connection and views
        print("Creating DuckDB connection and views...")
        connection = create_connection_and_views(tmpdir)
        
        # Query basic views
        print("\n--- Seeds Summary ---")
        seeds_df = connection.execute("SELECT * FROM seeds LIMIT 5").df()
        print(seeds_df[["seed", "avg_regret_empirical", "final_regret"]].to_string())
        
        print("\n--- Events Summary ---") 
        events_summary_df = connection.execute("SELECT * FROM events_summary").df()
        print(events_summary_df[["seed", "total_events", "inserts", "deletions", "avg_regret_per_event"]].to_string())
        
        # Query regret analysis view
        print("\n--- Regret Analysis by Operation ---")
        regret_df = query_regret_analysis(connection)
        if not regret_df.empty:
            print(regret_df[["seed", "op", "event_count", "avg_regret", "negative_count", "avg_nonneg_regret"]].to_string())
        else:
            print("No regret analysis data available")
        
        # Check for negative regret issues
        print("\n--- Negative Regret Summary ---")
        neg_summary = get_negative_regret_summary(connection)
        if not neg_summary.empty:
            print(neg_summary.to_string())
        else:
            print("✓ No negative regret issues found (as expected with enforcement enabled)")
        
        # Demo privacy metrics on delete events
        print("\n--- Privacy Metrics on Delete Events ---")
        privacy_df = connection.execute("""
            SELECT op, COUNT(*) as count, 
                   AVG(rho_spent) as avg_rho_spent,
                   AVG(m_capacity) as avg_m_capacity
            FROM events 
            WHERE rho_spent IS NOT NULL 
            GROUP BY op
        """).df()
        print(privacy_df.to_string())
        
        print(f"\n✓ Demo completed successfully in {tmpdir}")
        print("Parquet files and DuckDB views created and tested")


def main():
    """Run all demos."""
    print("Finalized Regret Calculation and Reporting Spec Demo")
    print("=" * 55)
    print()
    
    demo_nonnegative_regret_enforcement()
    demo_regularization_consistency()
    demo_config_defaults()
    demo_parquet_and_duckdb()
    
    print("=" * 55)
    print("All demos completed successfully!")
    print("\nKey features demonstrated:")
    print("✓ Non-negative regret enforcement")
    print("✓ Consistent λ regularization")
    print("✓ Finalized spec config defaults")
    print("✓ Parquet-only data pipeline")
    print("✓ DuckDB regret analysis views")
    print("✓ Privacy metrics on delete events")


if __name__ == "__main__":
    main()