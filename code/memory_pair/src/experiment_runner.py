#!/usr/bin/env python3
"""
Configuration-based experiment runner for Milestone 5.

This module provides a simple experiment runner that demonstrates
the adaptive capacity odometer with different privacy accountants.
It includes:

1. Configuration specification for different accountant types
2. Mock dataset and unlearning scenario
3. Experiment execution with matched seeds
4. Results collection and plotting
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from .adaptive_odometer import AdaptiveCapacityOdometer, DeletionEvent
from .plotting_utils import ExperimentResult, create_comprehensive_report


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    accountant_type: str
    budget_params: Dict[str, float]
    odometer_params: Dict[str, Any]
    dataset_params: Dict[str, Any]
    unlearning_params: Dict[str, Any]
    
    
class MockDataset:
    """Mock dataset for demonstration purposes."""
    
    def __init__(self, n_samples: int = 1000, n_features: int = 10, seed: int = 42):
        np.random.seed(seed)
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randn(n_samples)
        self.n_samples = n_samples
        self.n_features = n_features
        
    def get_sample(self, idx: int):
        """Get a single sample."""
        return self.X[idx], self.y[idx]
        
    def get_deletion_sensitivity(self, idx: int) -> float:
        """Compute mock deletion sensitivity for sample."""
        # Mock sensitivity based on sample norm
        return 0.1 + 0.5 * np.linalg.norm(self.X[idx]) / np.sqrt(self.n_features)


class MockUnlearningAlgorithm:
    """Mock unlearning algorithm for demonstration."""
    
    def __init__(self, dataset: MockDataset, lambda_reg: float = 0.1):
        self.dataset = dataset
        self.lambda_reg = lambda_reg
        self.theta = np.random.randn(dataset.n_features) * 0.1
        self.total_regret = 0.0
        
    def insert_sample(self, idx: int) -> float:
        """Insert a sample and return gradient magnitude."""
        x, y = self.dataset.get_sample(idx)
        
        # Compute gradient
        pred = self.theta @ x
        residual = pred - y
        grad = residual * x + self.lambda_reg * self.theta
        
        # SGD update
        lr = 0.01
        self.theta -= lr * grad
        
        # Update regret (mock)
        loss = 0.5 * residual**2 + 0.5 * self.lambda_reg * np.dot(self.theta, self.theta)
        self.total_regret += loss
        
        return np.linalg.norm(grad)
        
    def delete_sample(self, idx: int, noise_scale: float) -> Dict[str, float]:
        """Delete a sample with noise injection."""
        x, y = self.dataset.get_sample(idx)
        
        # Compute deletion direction (simplified)
        pred = self.theta @ x
        residual = pred - y
        deletion_direction = -residual * x  # Approximate deletion direction
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, size=deletion_direction.shape)
        noisy_deletion = deletion_direction + noise
        
        # Apply deletion update
        lr = 0.05
        self.theta += lr * noisy_deletion
        
        # Compute sensitivity
        sensitivity = np.linalg.norm(deletion_direction)
        
        # Compute gradient magnitude after deletion
        grad_mag = np.linalg.norm(residual * x + self.lambda_reg * self.theta)
        
        return {
            'sensitivity': sensitivity,
            'gradient_magnitude': grad_mag,
            'deletion_loss': 0.5 * residual**2
        }


def create_default_configs() -> List[ExperimentConfig]:
    """Create default experiment configurations for comparison."""
    configs = []
    
    # zCDP configuration
    configs.append(ExperimentConfig(
        accountant_type='zCDP',
        budget_params={'rho_total': 2.0, 'delta_total': 1e-5},
        odometer_params={
            'recalibration_enabled': True,
            'recalibration_interval': 5,
            'drift_threshold': 0.3,
            'gamma': 1.5,
            'lambda_': 0.1
        },
        dataset_params={'n_samples': 500, 'n_features': 10},
        unlearning_params={'lambda_reg': 0.1, 'n_insertions': 50, 'n_deletions': 20}
    ))
    
    # (ε,δ)-DP configuration
    configs.append(ExperimentConfig(
        accountant_type='eps_delta',
        budget_params={'eps_total': 1.0, 'delta_total': 1e-5},
        odometer_params={
            'recalibration_enabled': False,  # Traditional doesn't support recalibration
            'gamma': 1.5,
            'lambda_': 0.1
        },
        dataset_params={'n_samples': 500, 'n_features': 10},
        unlearning_params={'lambda_reg': 0.1, 'n_insertions': 50, 'n_deletions': 20}
    ))
    
    # Relaxed configuration
    configs.append(ExperimentConfig(
        accountant_type='relaxed',
        budget_params={'eps_total': 1.0, 'delta_total': 1e-5},
        odometer_params={
            'relaxation_factor': 0.6,
            'recalibration_enabled': False,
            'gamma': 1.5,
            'lambda_': 0.1
        },
        dataset_params={'n_samples': 500, 'n_features': 10},
        unlearning_params={'lambda_reg': 0.1, 'n_insertions': 50, 'n_deletions': 20}
    ))
    
    return configs


def run_single_experiment(config: ExperimentConfig, seed: int = 42) -> ExperimentResult:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        seed: Random seed for reproducibility
        
    Returns:
        Experiment result
    """
    print(f"🏃 Running {config.accountant_type} experiment with seed {seed}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Create dataset
    dataset = MockDataset(seed=seed, **config.dataset_params)
    
    # Create unlearning algorithm
    algorithm = MockUnlearningAlgorithm(dataset, 
                                       lambda_reg=config.unlearning_params['lambda_reg'])
    
    # Create adaptive odometer
    odometer = AdaptiveCapacityOdometer(
        accountant_type=config.accountant_type,
        budget_params=config.budget_params,
        **config.odometer_params
    )
    
    # Insertion phase (collect statistics)
    print(f"📥 Insertion phase: {config.unlearning_params['n_insertions']} samples")
    grad_norms = []
    
    for i in range(config.unlearning_params['n_insertions']):
        grad_norm = algorithm.insert_sample(i)
        grad_norms.append(grad_norm)
    
    # Compute calibration statistics
    stats = {
        'G': np.quantile(grad_norms, 0.95),  # 95th percentile gradient bound
        'D': 1.0,  # Mock diameter
        'c': 0.5,  # Mock curvature lower bound
        'C': 2.0   # Mock curvature upper bound
    }
    
    # Finalize odometer
    T_estimate = config.unlearning_params['n_insertions'] + config.unlearning_params['n_deletions']
    odometer.finalize_with(stats, T_estimate)
    
    print(f"🔧 Odometer finalized with capacity: {odometer.remaining_capacity()}")
    
    # Deletion phase
    deletion_events = []
    n_deletions = min(config.unlearning_params['n_deletions'], odometer.remaining_capacity())
    
    print(f"🗑️  Deletion phase: {n_deletions} deletions")
    
    for i in range(n_deletions):
        if odometer.remaining_capacity() <= 0:
            print(f"⚠️  Capacity exhausted after {i} deletions")
            break
        
        # Select sample to delete (random for demo)
        sample_idx = np.random.randint(0, config.unlearning_params['n_insertions'])
        
        # Get noise scale
        sensitivity_estimate = dataset.get_deletion_sensitivity(sample_idx)
        noise_scale = odometer.noise_scale(sensitivity_estimate)
        
        # Perform deletion
        deletion_result = algorithm.delete_sample(sample_idx, noise_scale)
        
        # Record deletion event
        event = odometer.spend(
            sensitivity=deletion_result['sensitivity'],
            gradient_magnitude=deletion_result['gradient_magnitude']
        )
        deletion_events.append(event)
        
        if i % 5 == 0:
            print(f"  Deletion {i+1}: sensitivity={deletion_result['sensitivity']:.3f}, "
                  f"noise_scale={noise_scale:.3f}, remaining={odometer.remaining_capacity()}")
    
    # Collect final results
    final_budget_summary = odometer.get_budget_summary()
    
    # Compute total budget used
    total_budget_used = 0.0
    for event in deletion_events:
        if event.budget_spent:
            total_budget_used += sum(event.budget_spent.values())
    
    result = ExperimentResult(
        accountant_type=config.accountant_type,
        seed=seed,
        total_regret=algorithm.total_regret,
        final_budget_usage=total_budget_used,
        deletion_count=len(deletion_events),
        avg_noise_scale=np.mean([e.noise_scale for e in deletion_events]) if deletion_events else 0.0,
        deletion_events=deletion_events
    )
    
    print(f"✅ {config.accountant_type} experiment complete: "
          f"regret={result.total_regret:.3f}, deletions={result.deletion_count}, "
          f"avg_noise={result.avg_noise_scale:.3f}")
    
    return result


def run_comparison_experiment(
    configs: Optional[List[ExperimentConfig]] = None,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "./milestone5_results"
) -> List[ExperimentResult]:
    """
    Run comparison experiment across multiple accountant types and seeds.
    
    Args:
        configs: List of experiment configurations (uses defaults if None)
        seeds: List of random seeds for matched experiments
        output_dir: Directory to save results and plots
        
    Returns:
        List of all experiment results
    """
    if configs is None:
        configs = create_default_configs()
    
    print(f"🚀 Starting comparison experiment")
    print(f"📋 Configurations: {[c.accountant_type for c in configs]}")
    print(f"🎲 Seeds: {seeds}")
    print(f"📁 Output directory: {output_dir}")
    
    # Run all experiments
    all_results = []
    start_time = time.time()
    
    for seed in seeds:
        for config in configs:
            try:
                result = run_single_experiment(config, seed)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error in {config.accountant_type} experiment with seed {seed}: {e}")
    
    duration = time.time() - start_time
    print(f"⏱️  Total experiment time: {duration:.1f} seconds")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results as JSON
    results_data = []
    for result in all_results:
        result_dict = {
            'accountant_type': result.accountant_type,
            'seed': result.seed,
            'total_regret': result.total_regret,
            'final_budget_usage': result.final_budget_usage,
            'deletion_count': result.deletion_count,
            'avg_noise_scale': result.avg_noise_scale,
            'deletion_events': [
                {
                    'event_id': event.event_id,
                    'sensitivity': event.sensitivity,
                    'noise_scale': event.noise_scale,
                    'budget_spent': event.budget_spent,
                    'remaining_capacity': event.remaining_capacity
                }
                for event in result.deletion_events
            ]
        }
        results_data.append(result_dict)
    
    with open(os.path.join(output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Raw results saved to {output_dir}/experiment_results.json")
    
    # Create comprehensive report with plots
    create_comprehensive_report(
        all_results,
        output_dir=output_dir,
        experiment_name="milestone5_comparison"
    )
    
    print(f"📊 Complete report saved to {output_dir}")
    return all_results


def main():
    """Main function for running the experiment."""
    print("🎯 Milestone 5 - Adaptive Capacity Odometer Experiment")
    print("=" * 60)
    
    # Run comparison experiment
    results = run_comparison_experiment(
        seeds=[42, 123, 456, 789, 999],  # 5 seeds for statistical significance
        output_dir="/tmp/milestone5_experiment"
    )
    
    print(f"\n📈 Final Summary:")
    print(f"   Total experiments: {len(results)}")
    print(f"   Accountant types: {len(set(r.accountant_type for r in results))}")
    print(f"   Seeds: {len(set(r.seed for r in results))}")
    
    # Print summary statistics
    grouped = {}
    for result in results:
        if result.accountant_type not in grouped:
            grouped[result.accountant_type] = []
        grouped[result.accountant_type].append(result)
    
    print(f"\n📊 Performance Summary:")
    for acc_type, acc_results in grouped.items():
        regrets = [r.total_regret for r in acc_results]
        deletions = [r.deletion_count for r in acc_results]
        noise_scales = [r.avg_noise_scale for r in acc_results]
        
        print(f"\n{acc_type}:")
        print(f"  Avg Regret: {np.mean(regrets):.3f} ± {np.std(regrets):.3f}")
        print(f"  Avg Deletions: {np.mean(deletions):.1f} ± {np.std(deletions):.1f}")
        print(f"  Avg Noise Scale: {np.mean(noise_scales):.3f} ± {np.std(noise_scales):.3f}")
    
    print("\n🎉 Milestone 5 experiment complete!")
    

if __name__ == "__main__":
    main()