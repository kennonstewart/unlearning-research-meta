#!/usr/bin/env python3
"""
Integration example: Using Dynamic Comparator with sublinear_regret experiment.

This demonstrates how to enable oracle functionality in existing experiments.
"""

import numpy as np
import os
import sys
import pandas as pd
import tempfile
from typing import Dict, Any

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "memory_pair", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "data_loader"))

from memory_pair import MemoryPair
from plotting import plot_regret_decomposition, analyze_regret_decomposition


class ExperimentConfig:
    """Configuration class for oracle-enabled experiments."""
    def __init__(self, **kwargs):
        # Default values
        self.enable_oracle = True
        self.oracle_window_W = 256  # Default from issue spec
        self.oracle_steps = 15
        self.oracle_stride = None  # Will default to W//2
        self.oracle_tol = 1e-6
        self.oracle_warmstart = True
        self.path_length_norm = 'L2'
        self.lambda_reg = 0.0
        
        # MemoryPair defaults
        self.m_max = 10
        self.strong_convexity = False
        
        # Update with provided values
        for k, v in kwargs.items():
            setattr(self, k, v)


def create_synthetic_drifting_stream(
    n_samples: int,
    dim: int,
    drift_frequency: int = 1000,
    drift_magnitude: float = 0.1,
    noise_level: float = 0.1,
    seed: int = 42
):
    """
    Create a synthetic drifting data stream.
    
    Args:
        n_samples: Number of samples to generate
        dim: Feature dimension
        drift_frequency: Number of samples between drifts
        drift_magnitude: Magnitude of parameter drift
        noise_level: Observation noise level
        seed: Random seed
        
    Yields:
        (x, y) tuples representing the stream
    """
    np.random.seed(seed)
    
    # Initialize true parameters
    theta_true = np.random.randn(dim) * 0.5
    
    for i in range(n_samples):
        # Apply drift periodically
        if i > 0 and i % drift_frequency == 0:
            drift_direction = np.random.randn(dim)
            drift_direction /= np.linalg.norm(drift_direction)
            theta_true += drift_magnitude * drift_direction
            print(f"[Stream] Applied drift at sample {i}, ||theta_true|| = {np.linalg.norm(theta_true):.4f}")
        
        # Generate sample
        x = np.random.randn(dim)
        y = float(theta_true @ x) + noise_level * np.random.randn()
        
        yield x, y, np.linalg.norm(theta_true)


def run_oracle_experiment(
    stream_generator,
    n_samples: int,
    config: ExperimentConfig,
    experiment_name: str = "oracle_experiment"
) -> pd.DataFrame:
    """
    Run an experiment with oracle tracking enabled.
    
    Args:
        stream_generator: Generator yielding (x, y, metadata) tuples
        n_samples: Number of samples to process
        config: Experiment configuration
        experiment_name: Name for logging
        
    Returns:
        DataFrame with event-level metrics including oracle data
    """
    print(f"Starting {experiment_name} with oracle enabled...")
    print(f"Oracle config: window={config.oracle_window_W}, steps={config.oracle_steps}")
    
    # Initialize MemoryPair with oracle
    dim = next(iter(stream_generator))[0].shape[0]  # Get dimension from first sample
    mp = MemoryPair(dim=dim, cfg=config)
    
    # Calibration phase
    print("Calibration phase...")
    calibration_samples = []
    for i, (x, y, metadata) in enumerate(stream_generator):
        calibration_samples.append((x, y, metadata))
        mp.calibrate_step(x, y)
        if i >= 4:  # 5 calibration samples
            break
    
    mp.finalize_calibration(gamma=0.5)
    
    # Main experiment loop
    print("Main experiment loop...")
    event_log = []
    
    for i, (x, y, true_theta_norm) in enumerate(stream_generator):
        if i >= n_samples:
            break
            
        # Process event
        pred = mp.insert(x, y)
        
        # Collect metrics
        metrics = mp.get_metrics_dict()
        
        # Log event data
        event_data = {
            'event': mp.events_seen,
            'sample_idx': i,
            'pred': pred,
            'true_y': y,
            'true_theta_norm': true_theta_norm,
            'theta_norm': np.linalg.norm(mp.theta),
            'phase': mp.phase.name,
            'cumulative_regret': mp.cumulative_regret,
        }
        
        # Add all metrics (including oracle metrics)
        event_data.update(metrics)
        
        event_log.append(event_data)
        
        # Progress updates
        if i % 500 == 0:
            oracle_info = ""
            if mp.oracle is not None:
                oracle_metrics = mp.oracle.get_oracle_metrics()
                oracle_info = f", P_T={oracle_metrics['P_T_est']:.4f}"
            print(f"Processed {i+1} samples, events={mp.events_seen}{oracle_info}")
    
    print(f"Experiment {experiment_name} completed.")
    print(f"Total events: {mp.events_seen}, Inserts: {mp.inserts_seen}")
    
    if mp.oracle is not None:
        final_metrics = mp.oracle.get_oracle_metrics()
        print(f"Final oracle metrics:")
        print(f"  P_T_est: {final_metrics['P_T_est']:.4f}")
        print(f"  Oracle refreshes: {final_metrics['oracle_refreshes']}")
        print(f"  Dynamic regret: {final_metrics['regret_dynamic']:.4f}")
        print(f"  Static term: {final_metrics['regret_static_term']:.4f}")
        print(f"  Path term: {final_metrics['regret_path_term']:.4f}")
    
    return pd.DataFrame(event_log)


def demonstrate_oracle_integration():
    """Demonstrate oracle integration with a realistic experiment."""
    print("=" * 60)
    print("Dynamic Comparator Integration Example")
    print("=" * 60)
    
    # Configuration
    config = ExperimentConfig(
        oracle_window_W=200,
        oracle_steps=12,
        oracle_stride=100,  # Refresh every 100 events
        lambda_reg=0.01,
        path_length_norm='L2'
    )
    
    # Generate drifting stream
    dim = 8
    n_samples = 2000
    drift_freq = 500  # Drift every 500 samples
    
    stream = create_synthetic_drifting_stream(
        n_samples=n_samples,
        dim=dim,
        drift_frequency=drift_freq,
        drift_magnitude=0.15,
        noise_level=0.1,
        seed=123
    )
    
    # Run experiment
    df = run_oracle_experiment(
        stream_generator=stream,
        n_samples=n_samples,
        config=config,
        experiment_name="drifting_synthetic"
    )
    
    # Analyze results
    if 'P_T_est' in df.columns:
        analysis = analyze_regret_decomposition(
            df['regret_static_term'].tolist(),
            df['regret_path_term'].tolist(),
            df['P_T_est'].tolist(),
            df['event'].tolist()
        )
        
        print("\n" + "=" * 40)
        print("EXPERIMENT ANALYSIS")
        print("=" * 40)
        print(f"Final P_T (path-length): {analysis['final_P_T']:.4f}")
        print(f"Final dynamic regret: {analysis['final_total_regret']:.4f}")
        print(f"Static vs Path decomposition:")
        print(f"  Static term: {analysis['final_static_term']:.4f}")
        print(f"  Path term: {analysis['final_path_term']:.4f}")
        print(f"  Path fraction: {analysis['path_fraction']:.4f}")
        print(f"Drift detected: {'Yes' if analysis['path_dominates'] else 'No'}")
        print(f"Estimated drift rate: {analysis['drift_rate_estimate']:.6f}")
        
        # Save results and create plots
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save CSV
            csv_path = os.path.join(temp_dir, "oracle_experiment_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
            
            # Create plots
            events = df['event'].tolist()
            regret_static = df['regret_static_term'].tolist()
            regret_path = df['regret_path_term'].tolist()
            P_T_values = df['P_T_est'].tolist()
            
            # Find drift points (approximately)
            drift_events = [drift_freq, 2*drift_freq, 3*drift_freq]
            
            plot_path = os.path.join(temp_dir, "oracle_experiment_plot.png")
            plot_regret_decomposition(
                events=events,
                regret_static=regret_static,
                regret_path=regret_path,
                P_T_values=P_T_values,
                oracle_refreshes=drift_events,
                title="Oracle Integration Example - Synthetic Drift",
                save_path=plot_path,
                show_theory_bound=True,
                G=1.0,
                lambda_param=0.01
            )
            
            print(f"Plot saved to: {plot_path}")
            print(f"Temporary files will be cleaned up automatically.")
    
    else:
        print("Warning: Oracle metrics not found in results")
    
    print("\n" + "=" * 60)
    print("✓ Oracle integration demonstration completed successfully!")
    print("✓ The Dynamic Comparator is ready for use in experiments.")
    print("=" * 60)


def show_usage_example():
    """Show how to use oracle in existing experiments."""
    print("\n" + "=" * 50)
    print("USAGE EXAMPLE FOR EXISTING EXPERIMENTS")
    print("=" * 50)
    
    usage_code = '''
# To enable oracle in existing experiments:

from memory_pair import MemoryPair

# 1. Create config with oracle enabled
class Config:
    enable_oracle = True
    oracle_window_W = 256        # Window size (default from spec)
    oracle_steps = 15            # Optimization steps per refresh
    oracle_stride = 128          # Events between refreshes (W//2)
    oracle_tol = 1e-6           # Convergence tolerance
    oracle_warmstart = True     # Warm-start from previous oracle
    path_length_norm = 'L2'     # L2 or L1 norm for path-length
    lambda_reg = 0.01           # Regularization for oracle ERM

# 2. Initialize MemoryPair with oracle
mp = MemoryPair(dim=your_dimension, cfg=Config())

# 3. Normal usage - oracle runs automatically
for x, y in your_data_stream:
    pred = mp.insert(x, y)
    
    # Get metrics including oracle data
    metrics = mp.get_metrics_dict()
    # metrics now includes: P_T_est, regret_dynamic, 
    # regret_static_term, regret_path_term, etc.

# 4. Access oracle metrics for logging/plotting
oracle_metrics = mp.oracle.get_oracle_metrics()
print(f"Path-length P_T: {oracle_metrics['P_T_est']}")
print(f"Dynamic regret: {oracle_metrics['regret_dynamic']}")

# 5. Use plotting utilities
from plotting import plot_regret_decomposition

plot_regret_decomposition(
    events=event_list,
    regret_static=static_term_list, 
    regret_path=path_term_list,
    P_T_values=P_T_list,
    title="Your Experiment - Regret Decomposition"
)
'''
    
    print(usage_code)


if __name__ == "__main__":
    demonstrate_oracle_integration()
    show_usage_example()