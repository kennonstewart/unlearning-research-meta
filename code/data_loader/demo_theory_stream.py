#!/usr/bin/env python3
"""
Theory Stream Demonstration

This script demonstrates the theory-first data loader functionality,
showing how to use theoretical constants as primary inputs and
validating that the stream enforces the specified targets.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from code.data_loader.theory_loader import get_theory_stream


def demo_theory_stream():
    """Demonstrate theory stream with real-time monitoring."""
    
    print("=== Theory-First Data Loader Demonstration ===\n")
    
    # Configuration
    config = {
        "dim": 8,
        "T": 2000,
        "target_G": 2.0,
        "target_D": 2.0,
        "target_c": 0.1,
        "target_C": 10.0,
        "target_lambda": 0.05,
        "target_PT": 40.0,
        "target_ST": 4000.0,
        "accountant": "zcdp",
        "rho_total": 1.0,
        "path_style": "rotating",
        "seed": 42
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create stream
    stream = get_theory_stream(**config)
    
    # Track metrics
    metrics = {
        "P_T_true": [],
        "g_norm": [],
        "ST_running": [],
        "clip_applied": [],
        "PT_target_residual": [],
        "ST_target_residual": [],
        "privacy_spend": [],
    }
    
    print("Running stream and collecting metrics...")
    
    # Run for configured number of steps
    for i in range(config["T"]):
        event = next(stream)
        
        # Collect metrics
        m = event["metrics"]
        metrics["P_T_true"].append(m["P_T_true"])
        metrics["g_norm"].append(m["g_norm"])
        metrics["ST_running"].append(m["ST_running"])
        metrics["clip_applied"].append(m["clip_applied"])
        metrics["PT_target_residual"].append(m["PT_target_residual"])
        metrics["ST_target_residual"].append(m["ST_target_residual"])
        metrics["privacy_spend"].append(m["privacy_spend_running"])
        
        # Progress updates with block-level ratios
        if (i + 1) % 500 == 0:
            # Calculate current ratios
            t = i + 1
            PT_ratio = m['P_T_true'] / (config['target_PT'] * t / config['T']) if t > 0 else 0
            ST_ratio = m['ST_running'] / (config['target_ST'] * t / config['T']) if t > 0 else 0
            
            print(f"  Step {i+1:4d}: P_T={m['P_T_true']:.2f}, g_norm={m['g_norm']:.3f}, "
                  f"ST={m['ST_running']:.0f}, clip_rate={np.mean(metrics['clip_applied'][-500:]):.3f}")
            print(f"    Block ratios: PT_ratio={PT_ratio:.3f}, ST_ratio={ST_ratio:.3f}")
            print(f"    Target residuals: PT={m['PT_target_residual']:.3f}, ST={m['ST_target_residual']:.3f}")
    
    print(f"\nCompleted {config['T']} steps.\n")
    
    # Final analysis
    final_PT = metrics["P_T_true"][-1]
    final_ST = metrics["ST_running"][-1]
    max_g_norm = max(metrics["g_norm"])
    final_privacy_spend = metrics["privacy_spend"][-1]
    clip_rate = np.mean(metrics["clip_applied"])
    
    print("=== FINAL RESULTS ===")
    print(f"Path Length (P_T):")
    print(f"  Target: {config['target_PT']:.1f}")
    print(f"  Actual: {final_PT:.1f}")
    print(f"  Error: {abs(final_PT/config['target_PT'] - 1)*100:.2f}%")
    print()
    
    print(f"Gradient Bound (G):")
    print(f"  Target: {config['target_G']:.1f}")
    print(f"  Max observed: {max_g_norm:.3f}")
    print(f"  Clip rate: {clip_rate*100:.1f}%")
    print()
    
    print(f"AdaGrad Energy (S_T):")
    print(f"  Target: {config['target_ST']:.0f}")
    print(f"  Actual: {final_ST:.0f}")
    print(f"  Error: {abs(final_ST/config['target_ST'] - 1)*100:.2f}%")
    print()
    
    print(f"Privacy Spend:")
    print(f"  Budget: {config['rho_total']:.1f}")
    print(f"  Spent: {final_privacy_spend:.3f}")
    print(f"  Remaining: {config['rho_total'] - final_privacy_spend:.3f}")
    print()
    
    # Check acceptance criteria
    print("=== ACCEPTANCE CRITERIA ===")
    PT_error = abs(final_PT/config['target_PT'] - 1)
    ST_error = abs(final_ST/config['target_ST'] - 1)
    
    print(f"AT-1 (Path length): {'✅ PASS' if PT_error <= 0.05 else '❌ FAIL'} ({PT_error*100:.2f}% error)")
    print(f"AT-2 (Gradient bound): {'✅ PASS' if max_g_norm <= 1.05*config['target_G'] and clip_rate <= 0.05 else '❌ FAIL'}")
    print(f"AT-5 (AdaGrad energy): {'✅ PASS' if ST_error <= 0.05 else '❌ FAIL'} ({ST_error*100:.2f}% error)")
    print(f"AT-6 (Privacy): {'✅ PASS' if final_privacy_spend <= config['rho_total'] else '❌ FAIL'}")
    
    return metrics, config


def plot_diagnostics(metrics, config):
    """Plot diagnostic charts for theory stream."""
    
    try:
        T = len(metrics["P_T_true"])
        steps = np.arange(T)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Theory Stream Diagnostics", fontsize=14)
        
        # Path length tracking
        ax = axes[0, 0]
        target_PT_line = config["target_PT"] * steps / T
        ax.plot(steps, metrics["P_T_true"], 'b-', label='Actual P_T', alpha=0.8)
        ax.plot(steps, target_PT_line, 'r--', label='Target P_T', alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Path Length")
        ax.set_title("AT-1: Path Length Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gradient norms
        ax = axes[0, 1]
        ax.plot(steps, metrics["g_norm"], 'g-', alpha=0.7)
        ax.axhline(y=config["target_G"], color='r', linestyle='--', label=f'Target G={config["target_G"]}')
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("AT-2: Gradient Bound Enforcement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AdaGrad energy
        ax = axes[1, 0]
        target_ST_line = config["target_ST"] * steps / T
        ax.plot(steps, metrics["ST_running"], 'purple', label='Actual S_T', alpha=0.8)
        ax.plot(steps, target_ST_line, 'orange', linestyle='--', label='Target S_T', alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Squared Gradients")
        ax.set_title("AT-5: AdaGrad Energy Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Privacy spend
        ax = axes[1, 1]
        ax.plot(steps, metrics["privacy_spend"], 'brown', alpha=0.8)
        ax.axhline(y=config["rho_total"], color='r', linestyle='--', label=f'Budget={config["rho_total"]}')
        ax.set_xlabel("Step")
        ax.set_ylabel("Privacy Spend (ρ)")
        ax.set_title("AT-6: Privacy Budget Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "/tmp/theory_stream_diagnostics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nDiagnostic plot saved to: {plot_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping plots.")


if __name__ == "__main__":
    print("Starting theory stream demonstration...")
    
    try:
        metrics, config = demo_theory_stream()
        plot_diagnostics(metrics, config)
        
        print("\n=== DEMONSTRATION COMPLETE ===")
        print("The theory-first data loader successfully demonstrates:")
        print("1. Theory constants as primary inputs")
        print("2. Real-time enforcement of theoretical constraints")
        print("3. Comprehensive diagnostic tracking")
        print("4. Privacy budget management")
        print("\nThis enables experiments to be parameterized directly by")
        print("theoretical properties rather than implementation details.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()