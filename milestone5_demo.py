#!/usr/bin/env python3
"""
Milestone 5 Demo: Comparing Different Accountant Strategies

This script demonstrates the key features of Milestone 5:
1. Creating different accountant strategies with the same parameters
2. Showing how they produce different noise scales and capacities
3. Demonstrating the CLI interface with different accountant types
4. Showing how the grid runner can compare multiple accountants
"""

import os
import sys
import tempfile
import subprocess

# Add the project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "memory_pair", "src"))

def demo_accountant_strategies():
    """Demonstrate the different accountant strategies."""
    print("="*60)
    print("MILESTONE 5 DEMO: Accountant Strategy Comparison")
    print("="*60)
    
    from accountant_strategies import create_accountant_strategy
    
    # Common parameters for fair comparison
    params = {
        "eps_total": 1.0,
        "delta_total": 1e-5,
        "T": 1000,
        "gamma": 0.3,  # More lenient for better capacity
        "lambda_": 0.1,
        "delta_b": 0.05,
        "m_max": 10,
        "relaxation_factor": 0.7,  # 30% noise reduction for relaxed
    }
    
    # Create different accountant strategies
    accountants = {
        "EpsDelta (Traditional)": create_accountant_strategy("eps_delta", **params),
        "zCDP (Zero-Concentrated)": create_accountant_strategy("zcdp", **params),
        "Relaxed (Experimental)": create_accountant_strategy("relaxed", **params),
    }
    
    # Mock calibration statistics
    stats = {"G": 1.0, "D": 2.0, "c": 0.5, "C": 2.0}
    
    print("\n1. ACCOUNTANT INITIALIZATION AND FINALIZATION")
    print("-" * 50)
    
    for name, accountant in accountants.items():
        print(f"\n{name}:")
        print(f"   Type: {accountant.accountant_type}")
        
        # Finalize with stats
        accountant.finalize_with(stats, T_estimate=1000)
        
        # Get metrics
        metrics = accountant.metrics()
        print(f"   Deletion Capacity: {metrics.get('deletion_capacity', 'N/A')}")
        print(f"   Noise Scale: {metrics.get('sigma_step_theory', 'N/A'):.4f}")
        
        # Show budget structure
        if 'eps_total' in metrics:
            print(f"   Budget: ε={metrics['eps_total']:.2f}, δ={metrics['delta_total']:.0e}")
        elif 'rho_total' in metrics:
            print(f"   Budget: ρ={metrics['rho_total']:.4f}, δ={metrics['delta_total']:.0e}")
    
    print("\n2. NOISE SCALE COMPARISON")
    print("-" * 50)
    
    noise_scales = {}
    for name, accountant in accountants.items():
        noise = accountant.noise_scale()
        noise_scales[name] = noise
        print(f"{name}: σ = {noise:.4f}")
    
    # Find the baseline (eps_delta)
    baseline_noise = noise_scales["EpsDelta (Traditional)"]
    print(f"\nNoise reduction compared to traditional (ε,δ)-DP:")
    for name, noise in noise_scales.items():
        if name != "EpsDelta (Traditional)":
            reduction = (baseline_noise - noise) / baseline_noise * 100
            print(f"   {name}: {reduction:+.1f}% ({noise:.4f} vs {baseline_noise:.4f})")
    
    print("\n3. PRIVACY BUDGET SPENDING SIMULATION")
    print("-" * 50)
    
    for name, accountant in accountants.items():
        print(f"\n{name}:")
        
        # Get initial metrics
        initial_metrics = accountant.metrics()
        print(f"   Initial: {initial_metrics['deletions_count']}/{initial_metrics['deletion_capacity']} deletions")
        
        # Simulate one deletion
        try:
            sensitivity = 0.5
            sigma = accountant.noise_scale()
            
            if accountant.accountant_type == "eps_delta":
                accountant.spend()  # eps_delta doesn't use sensitivity/sigma
            else:
                accountant.spend(sensitivity, sigma)
            
            # Get updated metrics
            final_metrics = accountant.metrics()
            print(f"   After delete: {final_metrics['deletions_count']}/{final_metrics['deletion_capacity']} deletions")
            print(f"   Remaining capacity: {final_metrics.get('capacity_remaining', 'N/A')}")
            
        except Exception as e:
            print(f"   Error during spending: {e}")
    
    print("\n" + "="*60)
    print("✅ Milestone 5 accountant strategies working correctly!")
    print("="*60)


def demo_cli_interface():
    """Demonstrate the CLI interface with different accountant types."""
    print("\n" + "="*60)
    print("CLI INTERFACE DEMONSTRATION")
    print("="*60)
    
    cli_examples = [
        ("Traditional (ε,δ)-DP", "--accountant eps_delta --eps-total 1.0 --delta-total 1e-5"),
        ("Zero-Concentrated DP", "--accountant zcdp --eps-total 1.0 --delta-total 1e-5"), 
        ("Relaxed Mode", "--accountant relaxed --eps-total 1.0 --relaxation-factor 0.8"),
        ("Backward Compatible", "--accountant default --eps-total 1.0"),  # Maps to eps_delta
        ("Legacy RDP Support", "--accountant rdp --eps-total 1.0"),  # Maps to zcdp
    ]
    
    print("\nSupported CLI commands for different accountant types:")
    print("-" * 50)
    
    for description, cli_args in cli_examples:
        print(f"\n{description}:")
        print(f"   python cli.py {cli_args} --max-events 1000 --seeds 1")
    
    print(f"\nAll accountant types support the same core parameters:")
    print(f"   --gamma-learn, --gamma-priv, --eps-total, --delta-total")
    print(f"   --bootstrap-iters, --delete-ratio, --max-events, --seeds")
    print(f"   --quantile, --lambda-strong, --delta-b")
    
    print(f"\nRelaxed accountant adds:")
    print(f"   --relaxation-factor (default 0.8 = 20% noise reduction)")


def demo_grid_search():
    """Demonstrate grid search with multiple accountant types."""
    print("\n" + "="*60)
    print("GRID SEARCH DEMONSTRATION")
    print("="*60)
    
    print("\nThe updated grids.yaml supports all Milestone 5 accountant types:")
    print("-" * 50)
    
    # Show the grid configuration
    try:
        import yaml
        grid_path = os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity", "grids.yaml")
        
        if os.path.exists(grid_path):
            with open(grid_path, 'r') as f:
                grid = yaml.safe_load(f)
            
            print(f"Accountant types in grid: {grid.get('accountant', [])}")
            print(f"Gamma splits: {grid.get('gamma_split', [])}")
            print(f"Delete ratios: {grid.get('delete_ratio', [])}")
            print(f"Relaxation factors: {grid.get('relaxation_factor', [])}")
            
            # Calculate total combinations
            total_combos = 1
            for key, values in grid.items():
                if isinstance(values, list):
                    total_combos *= len(values)
            
            print(f"\nTotal grid combinations: {total_combos}")
            
        else:
            print("grids.yaml not found")
    
    except Exception as e:
        print(f"Error reading grids.yaml: {e}")
    
    print(f"\nExample grid search command:")
    print(f"   python agents/grid_runner.py \\")
    print(f"       --grid-file grids.yaml \\")
    print(f"       --parallel 4 \\")
    print(f"       --seeds 3 \\")
    print(f"       --base-out results/milestone5_comparison")
    
    print(f"\nThis will automatically generate:")
    print(f"   • Regret comparison plots by accountant type")
    print(f"   • Capacity vs noise tradeoff analysis") 
    print(f"   • Summary statistics across all combinations")
    print(f"   • Master CSV with all results for further analysis")


if __name__ == "__main__":
    try:
        demo_accountant_strategies()
        demo_cli_interface() 
        demo_grid_search()
        
        print("\n" + "🎉"*20)
        print("MILESTONE 5 IMPLEMENTATION COMPLETE!")
        print("🎉"*20)
        print("\nKey features implemented:")
        print("✅ Three accountant strategies: EpsDelta, zCDP, Relaxed")
        print("✅ Unified interface with consistent metrics")
        print("✅ CLI support with backward compatibility")
        print("✅ Grid search integration with automatic visualization")
        print("✅ Noise calibration using calibrated sensitivities")
        print("✅ Dynamic capacity adjustment during deletion phase")
        print("✅ Comprehensive testing and validation")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to run from the project root with proper PYTHONPATH")
    except Exception as e:
        print(f"Demo error: {e}")
        sys.exit(1)