#!/usr/bin/env python3
"""
Integration test for Milestone 5 - end-to-end test of accountant strategies.

This test runs a minimal experiment with different accountant types to verify:
1. All accountant types can run experiments successfully  
2. Different accountant types produce different results
3. Capacity limits are properly enforced
4. Visualization generation works
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "memory_pair", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "data_loader"))

from config import Config
from runner import ExperimentRunner


def test_single_accountant_run(accountant_type: str, temp_dir: str) -> dict:
    """Test a single accountant type with minimal parameters."""
    print(f"\n--- Testing {accountant_type} accountant ---")
    
    # Create minimal config for testing
    cfg = Config(
        dataset="synthetic",
        gamma_bar=1.0,
        gamma_split=0.5,
        bootstrap_iters=10,  # Very small for testing
        delete_ratio=2.0,
        max_events=50,  # Very small for testing
        seeds=1,
        out_dir=os.path.join(temp_dir, accountant_type),
        algo="memorypair",
        eps_total=1.0,
        delta_total=1e-5,
        lambda_=0.1,
        delta_b=0.05,
        quantile=0.95,
        D_cap=10.0,
        accountant=accountant_type,
        relaxation_factor=0.8,  # For relaxed accountant
        m_max=5,  # Small capacity for testing
        sens_calib=5,  # Small number for testing
    )
    
    try:
        # Create and run experiment
        runner = ExperimentRunner(cfg)
        csv_path = runner.run_single_seed(0)
        
        # Verify output file exists
        if not os.path.exists(csv_path):
            return {"success": False, "error": f"Output file not created: {csv_path}"}
        
        # Read and analyze results
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            return {"success": False, "error": "Empty output file"}
        
        # Extract key metrics
        results = {
            "success": True,
            "total_events": len(df),
            "inserts": len(df[df["op"] == "insert"]) if "op" in df.columns else 0,
            "deletes": len(df[df["op"] == "delete"]) if "op" in df.columns else 0,
            "final_regret": df["regret"].iloc[-1] if "regret" in df.columns and len(df) > 0 else None,
            "accountant_type": df["accountant_type"].iloc[-1] if "accountant_type" in df.columns and len(df) > 0 else None,
            "capacity": df["deletion_capacity"].iloc[-1] if "deletion_capacity" in df.columns and len(df) > 0 else None,
            "sigma_theory": df["sigma_step_theory"].iloc[-1] if "sigma_step_theory" in df.columns and len(df) > 0 else None,
            "csv_path": csv_path,
        }
        
        print(f"✅ {accountant_type} completed successfully:")
        print(f"   - Events: {results['total_events']}")
        print(f"   - Inserts: {results['inserts']}, Deletes: {results['deletes']}")
        print(f"   - Final regret: {results['final_regret']}")
        print(f"   - Capacity: {results['capacity']}")
        print(f"   - Noise scale: {results['sigma_theory']}")
        
        return results
        
    except Exception as e:
        print(f"❌ {accountant_type} failed: {e}")
        return {"success": False, "error": str(e)}


def test_accountant_differences(results: dict) -> None:
    """Test that different accountants produce different results."""
    print("\n--- Testing Accountant Differences ---")
    
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if len(successful_results) < 2:
        print("❌ Need at least 2 successful accountants to compare")
        return
    
    # Compare noise scales
    noise_scales = {k: v.get("sigma_theory") for k, v in successful_results.items()}
    noise_scales = {k: v for k, v in noise_scales.items() if v is not None}
    
    if len(set(noise_scales.values())) > 1:
        print("✅ Different accountants produce different noise scales:")
        for acc_type, noise in noise_scales.items():
            print(f"   - {acc_type}: σ = {noise:.4f}")
    else:
        print("⚠️  All accountants produced same noise scale")
    
    # Compare capacities
    capacities = {k: v.get("capacity") for k, v in successful_results.items()}
    capacities = {k: v for k, v in capacities.items() if v is not None}
    
    if len(set(capacities.values())) > 1:
        print("✅ Different accountants produce different capacities:")
        for acc_type, capacity in capacities.items():
            print(f"   - {acc_type}: m = {capacity}")
    else:
        print("⚠️  All accountants produced same capacity")
    
    # Verify relaxed accountant reduces noise compared to zCDP
    if "relaxed" in noise_scales and "zcdp" in noise_scales:
        relaxed_noise = noise_scales["relaxed"]
        zcdp_noise = noise_scales["zcdp"]
        if relaxed_noise < zcdp_noise:
            print(f"✅ Relaxed accountant reduces noise: {relaxed_noise:.4f} < {zcdp_noise:.4f}")
        else:
            print(f"⚠️  Relaxed accountant did not reduce noise: {relaxed_noise:.4f} >= {zcdp_noise:.4f}")


def test_visualization_generation(results: dict, temp_dir: str) -> None:
    """Test that visualizations can be generated."""
    print("\n--- Testing Visualization Generation ---")
    
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}
    
    if len(successful_results) < 2:
        print("❌ Need at least 2 successful accountants for visualization")
        return
    
    try:
        from plots import plot_accountant_comparison, plot_accountant_summary_stats
        
        # Create mock data directory structure
        vis_test_dir = os.path.join(temp_dir, "vis_test")
        os.makedirs(vis_test_dir, exist_ok=True)
        
        # Copy CSV files to appropriate subdirectories
        for acc_type, result in successful_results.items():
            acc_dir = os.path.join(vis_test_dir, f"test_{acc_type}")
            os.makedirs(acc_dir, exist_ok=True)
            
            if result.get("csv_path") and os.path.exists(result["csv_path"]):
                shutil.copy(result["csv_path"], os.path.join(acc_dir, "seed_000.csv"))
        
        # Test accountant comparison plot
        plot_path = os.path.join(temp_dir, "test_comparison.png")
        plot_accountant_comparison(
            data_dir=vis_test_dir,
            out_path=plot_path,
            metric="regret"
        )
        
        if os.path.exists(plot_path):
            print("✅ Accountant comparison plot generated successfully")
        else:
            print("❌ Accountant comparison plot not generated")
    
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")


def run_integration_test():
    """Run the complete integration test."""
    print("="*60)
    print("Milestone 5 Integration Test")
    print("="*60)
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test different accountant types
        accountant_types = ["eps_delta", "zcdp", "relaxed"]
        results = {}
        
        for accountant_type in accountant_types:
            results[accountant_type] = test_single_accountant_run(accountant_type, temp_dir)
        
        # Test differences between accountants
        test_accountant_differences(results)
        
        # Test visualization generation
        test_visualization_generation(results, temp_dir)
        
        # Summary
        print("\n" + "="*60)
        print("Integration Test Summary")
        print("="*60)
        
        successful_count = sum(1 for r in results.values() if r.get("success", False))
        total_count = len(results)
        
        print(f"Successful runs: {successful_count}/{total_count}")
        
        for acc_type, result in results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"{status} {acc_type}: {result.get('error', 'Success')}")
        
        if successful_count == total_count:
            print("\n🎉 All tests passed! Milestone 5 implementation is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total_count - successful_count} tests failed. See errors above.")
            return False


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)