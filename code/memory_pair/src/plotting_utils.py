#!/usr/bin/env python3
"""
Plotting utilities for comparing regret vs accountant type.

This module provides functions to create plots comparing the performance
of different privacy accountants in terms of regret, budget usage, and
noise scale across matched seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import os
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    accountant_type: str
    seed: int
    total_regret: float
    final_budget_usage: float
    deletion_count: int
    avg_noise_scale: float
    deletion_events: List[Any]  # List of DeletionEvent objects
    

def plot_regret_comparison(
    results: List[ExperimentResult],
    save_path: Optional[str] = None,
    title: str = "Regret vs Privacy Accountant Type"
) -> None:
    """
    Plot regret comparison across different accountant types.
    
    Args:
        results: List of experiment results to plot
        save_path: Optional path to save the plot
        title: Plot title
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Group results by accountant type
    grouped_results = {}
    for result in results:
        if result.accountant_type not in grouped_results:
            grouped_results[result.accountant_type] = []
        grouped_results[result.accountant_type].append(result)
    
    accountant_types = list(grouped_results.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(accountant_types)))
    
    # Plot 1: Total Regret
    ax1.set_title("Total Regret by Accountant Type")
    for i, (acc_type, acc_results) in enumerate(grouped_results.items()):
        regrets = [r.total_regret for r in acc_results]
        x_pos = [i] * len(regrets)
        ax1.scatter(x_pos, regrets, c=[colors[i]], label=acc_type, alpha=0.7, s=50)
        
        # Add mean line
        if regrets:
            ax1.hlines(np.mean(regrets), i-0.2, i+0.2, colors=colors[i], linewidth=3)
    
    ax1.set_xticks(range(len(accountant_types)))
    ax1.set_xticklabels(accountant_types, rotation=45)
    ax1.set_ylabel("Total Regret")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Budget Usage
    ax2.set_title("Budget Usage by Accountant Type")
    for i, (acc_type, acc_results) in enumerate(grouped_results.items()):
        budget_usage = [r.final_budget_usage for r in acc_results]
        x_pos = [i] * len(budget_usage)
        ax2.scatter(x_pos, budget_usage, c=[colors[i]], label=acc_type, alpha=0.7, s=50)
        
        # Add mean line
        if budget_usage:
            ax2.hlines(np.mean(budget_usage), i-0.2, i+0.2, colors=colors[i], linewidth=3)
    
    ax2.set_xticks(range(len(accountant_types)))
    ax2.set_xticklabels(accountant_types, rotation=45)
    ax2.set_ylabel("Budget Usage")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Deletion Count
    ax3.set_title("Deletion Count by Accountant Type")
    for i, (acc_type, acc_results) in enumerate(grouped_results.items()):
        deletion_counts = [r.deletion_count for r in acc_results]
        x_pos = [i] * len(deletion_counts)
        ax3.scatter(x_pos, deletion_counts, c=[colors[i]], label=acc_type, alpha=0.7, s=50)
        
        # Add mean line
        if deletion_counts:
            ax3.hlines(np.mean(deletion_counts), i-0.2, i+0.2, colors=colors[i], linewidth=3)
    
    ax3.set_xticks(range(len(accountant_types)))
    ax3.set_xticklabels(accountant_types, rotation=45)
    ax3.set_ylabel("Deletions Performed")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average Noise Scale
    ax4.set_title("Average Noise Scale by Accountant Type")
    for i, (acc_type, acc_results) in enumerate(grouped_results.items()):
        noise_scales = [r.avg_noise_scale for r in acc_results]
        x_pos = [i] * len(noise_scales)
        ax4.scatter(x_pos, noise_scales, c=[colors[i]], label=acc_type, alpha=0.7, s=50)
        
        # Add mean line
        if noise_scales:
            ax4.hlines(np.mean(noise_scales), i-0.2, i+0.2, colors=colors[i], linewidth=3)
    
    ax4.set_xticks(range(len(accountant_types)))
    ax4.set_xticklabels(accountant_types, rotation=45)
    ax4.set_ylabel("Average Noise Scale")
    ax4.set_yscale('log')  # Use log scale for noise
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


def plot_noise_scale_evolution(
    results: List[ExperimentResult],
    save_path: Optional[str] = None,
    title: str = "Noise Scale Evolution Over Deletions"
) -> None:
    """
    Plot how noise scale evolves over deletions for different accountants.
    
    Args:
        results: List of experiment results to plot
        save_path: Optional path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group results by accountant type
    grouped_results = {}
    for result in results:
        if result.accountant_type not in grouped_results:
            grouped_results[result.accountant_type] = []
        grouped_results[result.accountant_type].append(result)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(grouped_results)))
    
    for i, (acc_type, acc_results) in enumerate(grouped_results.items()):
        # Aggregate noise scales across all runs for this accountant type
        all_noise_scales = []
        max_deletions = 0
        
        for result in acc_results:
            if hasattr(result, 'deletion_events') and result.deletion_events:
                noise_scales = [event.noise_scale for event in result.deletion_events]
                all_noise_scales.append(noise_scales)
                max_deletions = max(max_deletions, len(noise_scales))
        
        if all_noise_scales:
            # Compute mean and std across runs
            deletion_indices = range(1, max_deletions + 1)
            mean_noise = []
            std_noise = []
            
            for deletion_idx in deletion_indices:
                values_at_idx = []
                for noise_scales in all_noise_scales:
                    if deletion_idx <= len(noise_scales):
                        values_at_idx.append(noise_scales[deletion_idx - 1])
                
                if values_at_idx:
                    mean_noise.append(np.mean(values_at_idx))
                    std_noise.append(np.std(values_at_idx))
                else:
                    break
            
            # Plot mean with error bars
            deletion_indices = range(1, len(mean_noise) + 1)
            ax.plot(deletion_indices, mean_noise, 
                   color=colors[i], label=acc_type, linewidth=2)
            ax.fill_between(deletion_indices, 
                           np.array(mean_noise) - np.array(std_noise),
                           np.array(mean_noise) + np.array(std_noise),
                           color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Deletion Number")
    ax.set_ylabel("Noise Scale")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Noise evolution plot saved to {save_path}")
    else:
        plt.show()


def plot_budget_consumption(
    results: List[ExperimentResult],
    save_path: Optional[str] = None,
    title: str = "Budget Consumption Over Time"
) -> None:
    """
    Plot cumulative budget consumption over deletions.
    
    Args:
        results: List of experiment results to plot
        save_path: Optional path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Group results by accountant type
    grouped_results = {}
    for result in results:
        if result.accountant_type not in grouped_results:
            grouped_results[result.accountant_type] = []
        grouped_results[result.accountant_type].append(result)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(grouped_results)))
    
    for i, (acc_type, acc_results) in enumerate(grouped_results.items()):
        # Aggregate budget consumption across all runs
        all_budget_series = []
        max_deletions = 0
        
        for result in acc_results:
            if hasattr(result, 'deletion_events') and result.deletion_events:
                cumulative_budget = []
                total_spent = 0.0
                
                for event in result.deletion_events:
                    # Sum all budget types spent
                    if hasattr(event, 'budget_spent') and event.budget_spent:
                        spent_this_round = sum(event.budget_spent.values())
                        total_spent += spent_this_round
                    cumulative_budget.append(total_spent)
                
                all_budget_series.append(cumulative_budget)
                max_deletions = max(max_deletions, len(cumulative_budget))
        
        if all_budget_series:
            # Compute mean and std across runs
            deletion_indices = range(1, max_deletions + 1)
            mean_budget = []
            std_budget = []
            
            for deletion_idx in deletion_indices:
                values_at_idx = []
                for budget_series in all_budget_series:
                    if deletion_idx <= len(budget_series):
                        values_at_idx.append(budget_series[deletion_idx - 1])
                
                if values_at_idx:
                    mean_budget.append(np.mean(values_at_idx))
                    std_budget.append(np.std(values_at_idx))
                else:
                    break
            
            # Plot mean with error bars
            deletion_indices = range(1, len(mean_budget) + 1)
            ax.plot(deletion_indices, mean_budget, 
                   color=colors[i], label=acc_type, linewidth=2)
            ax.fill_between(deletion_indices, 
                           np.array(mean_budget) - np.array(std_budget),
                           np.array(mean_budget) + np.array(std_budget),
                           color=colors[i], alpha=0.2)
    
    ax.set_xlabel("Deletion Number")
    ax.set_ylabel("Cumulative Budget Spent")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Budget consumption plot saved to {save_path}")
    else:
        plt.show()


def create_comprehensive_report(
    results: List[ExperimentResult],
    output_dir: str = "./plots",
    experiment_name: str = "adaptive_odometer_comparison"
) -> None:
    """
    Create a comprehensive report with all comparison plots.
    
    Args:
        results: List of experiment results to analyze
        output_dir: Directory to save plots
        experiment_name: Name prefix for saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Creating comprehensive report for {len(results)} experiment results")
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    grouped_results = {}
    for result in results:
        if result.accountant_type not in grouped_results:
            grouped_results[result.accountant_type] = []
        grouped_results[result.accountant_type].append(result)
    
    for acc_type, acc_results in grouped_results.items():
        regrets = [r.total_regret for r in acc_results]
        deletions = [r.deletion_count for r in acc_results]
        budgets = [r.final_budget_usage for r in acc_results]
        noise_scales = [r.avg_noise_scale for r in acc_results]
        
        print(f"\n{acc_type} ({len(acc_results)} runs):")
        print(f"  Regret: {np.mean(regrets):.4f} ± {np.std(regrets):.4f}")
        print(f"  Deletions: {np.mean(deletions):.1f} ± {np.std(deletions):.1f}")
        print(f"  Budget Usage: {np.mean(budgets):.4f} ± {np.std(budgets):.4f}")
        print(f"  Avg Noise Scale: {np.mean(noise_scales):.4f} ± {np.std(noise_scales):.4f}")
    
    # Create plots
    print("\n📊 Generating plots...")
    
    # Main comparison plot
    plot_regret_comparison(
        results,
        save_path=os.path.join(output_dir, f"{experiment_name}_regret_comparison.png"),
        title=f"Privacy Accountant Comparison - {experiment_name}"
    )
    
    # Noise scale evolution
    plot_noise_scale_evolution(
        results,
        save_path=os.path.join(output_dir, f"{experiment_name}_noise_evolution.png"),
        title=f"Noise Scale Evolution - {experiment_name}"
    )
    
    # Budget consumption
    plot_budget_consumption(
        results,
        save_path=os.path.join(output_dir, f"{experiment_name}_budget_consumption.png"),
        title=f"Budget Consumption - {experiment_name}"
    )
    
    print(f"\n✅ Comprehensive report saved to {output_dir}")


# Demo function for testing the plotting utilities
def demo_plotting():
    """Demo function showing how to use the plotting utilities."""
    print("🎬 Demo: Privacy Accountant Plotting Utilities")
    
    # Create mock experiment results
    np.random.seed(42)
    results = []
    
    for seed in range(5):  # 5 runs per accountant type
        for acc_type in ['zCDP', 'eps_delta', 'relaxed_0.5']:
            # Mock deletion events
            deletion_events = []
            for i in range(np.random.randint(3, 8)):  # 3-7 deletions
                from code.memory_pair.src.adaptive_odometer import DeletionEvent
                event = DeletionEvent(
                    event_id=i,
                    accountant_type=acc_type,
                    sensitivity=0.3 + 0.2 * np.random.random(),
                    noise_scale=1.0 + 2.0 * np.random.random() * (1.5 if acc_type == 'eps_delta' else 1.0),
                    budget_spent={'eps_spent': 0.1} if 'eps' in acc_type else {'rho_spent': 0.05},
                    remaining_capacity=10 - i,
                    gradient_magnitude=0.5 + 0.5 * np.random.random()
                )
                deletion_events.append(event)
            
            result = ExperimentResult(
                accountant_type=acc_type,
                seed=seed,
                total_regret=1.0 + 0.5 * np.random.random() * (1.2 if acc_type == 'eps_delta' else 1.0),
                final_budget_usage=len(deletion_events) * 0.1,
                deletion_count=len(deletion_events),
                avg_noise_scale=np.mean([e.noise_scale for e in deletion_events]),
                deletion_events=deletion_events
            )
            results.append(result)
    
    # Create comprehensive report
    create_comprehensive_report(results, output_dir="/tmp/demo_plots", experiment_name="demo")
    
    print("✅ Demo plotting complete!")


if __name__ == "__main__":
    demo_plotting()