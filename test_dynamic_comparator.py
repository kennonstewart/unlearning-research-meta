#!/usr/bin/env python3
"""
Test script for Dynamic Comparator and Path-Length P_T implementation.
"""

import numpy as np
import os
import sys
import tempfile
import pandas as pd
from typing import Dict, Any

# Add the memory_pair module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "memory_pair", "src"))

from memory_pair import MemoryPair
from comparators import RollingOracle

class SimpleConfig:
    """Simple configuration object for testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_rolling_oracle_basic():
    """Test basic oracle functionality."""
    print("Testing RollingOracle basic functionality...")
    
    dim = 5
    oracle = RollingOracle(dim=dim, window_W=10, oracle_steps=5)
    
    # Generate some synthetic data
    np.random.seed(42)
    theta_true = np.random.randn(dim) * 0.5
    
    for i in range(15):
        x = np.random.randn(dim)
        y = float(theta_true @ x) + 0.1 * np.random.randn()
        current_theta = np.random.randn(dim) * 0.1
        
        refreshed = oracle.maybe_update(x, y, current_theta)
        oracle.update_regret_accounting(x, y, current_theta)
        
        if refreshed:
            print(f"Oracle refreshed at step {i+1}")
    
    metrics = oracle.get_oracle_metrics()
    print(f"Final metrics: P_T_est={metrics['P_T_est']:.4f}, "
          f"oracle_refreshes={metrics['oracle_refreshes']}")
    
    assert metrics['oracle_refreshes'] > 0, "Oracle should have refreshed at least once"
    assert metrics['P_T_est'] >= 0, "Path length should be non-negative"
    print("✓ RollingOracle basic test passed")


def test_memory_pair_with_oracle():
    """Test MemoryPair integration with oracle."""
    print("Testing MemoryPair with oracle integration...")
    
    dim = 3
    cfg = SimpleConfig(
        enable_oracle=True,
        oracle_window_W=8,
        oracle_steps=5,
        oracle_stride=4,
        lambda_reg=0.01,
        path_length_norm='L2'
    )
    
    mp = MemoryPair(dim=dim, cfg=cfg)
    
    # Bootstrap phase
    print("Running calibration phase...")
    np.random.seed(42)
    for i in range(5):
        x = np.random.randn(dim)
        y = float(np.random.randn(1))
        mp.calibrate_step(x, y)
    
    mp.finalize_calibration(gamma=0.5)
    
    # Learning and interleaving phases
    print("Running learning/interleaving phases...")
    for i in range(20):
        x = np.random.randn(dim)
        y = float(np.random.randn(1))
        mp.insert(x, y)
    
    # Check oracle metrics
    metrics = mp.get_metrics_dict()
    
    # Verify oracle metrics are present
    expected_oracle_keys = ['P_T_est', 'regret_dynamic', 'regret_static_term', 'regret_path_term']
    for key in expected_oracle_keys:
        assert key in metrics, f"Missing oracle metric: {key}"
    
    print(f"Oracle metrics: P_T_est={metrics['P_T_est']:.4f}, "
          f"regret_dynamic={metrics['regret_dynamic']:.4f}")
    print("✓ MemoryPair oracle integration test passed")


def test_no_drift_scenario():
    """Test oracle on stationary data (no drift scenario)."""
    print("Testing no-drift control scenario...")
    
    dim = 4
    cfg = SimpleConfig(
        enable_oracle=True,
        oracle_window_W=10,
        oracle_steps=10,
        lambda_reg=0.05
    )
    
    mp = MemoryPair(dim=dim, cfg=cfg)
    
    # Fixed true parameters for stationary scenario
    np.random.seed(123)
    theta_true = np.random.randn(dim) * 0.3
    
    # Calibration
    for i in range(5):
        x = np.random.randn(dim)
        y = float(theta_true @ x) + 0.1 * np.random.randn()
        mp.calibrate_step(x, y)
    
    mp.finalize_calibration(gamma=0.5)
    
    # Generate stationary stream
    for i in range(30):
        x = np.random.randn(dim)
        y = float(theta_true @ x) + 0.1 * np.random.randn()
        mp.insert(x, y)
    
    metrics = mp.get_metrics_dict()
    
    # In stationary case, path term should be relatively small
    path_ratio = abs(metrics['regret_path_term']) / (abs(metrics['regret_static_term']) + 1e-6)
    print(f"Path term ratio: {path_ratio:.4f} (should be small for stationary data)")
    print(f"P_T_est: {metrics['P_T_est']:.4f}")
    
    # Path length should be small but non-zero due to optimization noise
    assert metrics['P_T_est'] >= 0, "Path length should be non-negative"
    print("✓ No-drift control test passed")


def test_drift_scenario():
    """Test oracle on drifting data."""
    print("Testing controlled drift scenario...")
    
    dim = 4
    cfg = SimpleConfig(
        enable_oracle=True,
        oracle_window_W=12,
        oracle_steps=8,
        oracle_stride=6,
        lambda_reg=0.02
    )
    
    mp = MemoryPair(dim=dim, cfg=cfg)
    
    # Calibration with initial parameters
    np.random.seed(456)
    theta_true = np.random.randn(dim) * 0.3
    
    for i in range(5):
        x = np.random.randn(dim)
        y = float(theta_true @ x) + 0.1 * np.random.randn()
        mp.calibrate_step(x, y)
    
    mp.finalize_calibration(gamma=0.5)
    
    # Generate drifting stream
    drift_rate = 0.05
    for i in range(40):
        # Slowly drift the true parameters
        if i % 10 == 0 and i > 0:
            theta_true += np.random.randn(dim) * drift_rate
            print(f"Applied drift at step {i}, new theta_true norm: {np.linalg.norm(theta_true):.4f}")
        
        x = np.random.randn(dim)
        y = float(theta_true @ x) + 0.1 * np.random.randn()
        mp.insert(x, y)
    
    metrics = mp.get_metrics_dict()
    
    print(f"Final P_T_est: {metrics['P_T_est']:.4f}")
    print(f"Dynamic regret: {metrics['regret_dynamic']:.4f}")
    print(f"Static term: {metrics['regret_static_term']:.4f}")
    print(f"Path term: {metrics['regret_path_term']:.4f}")
    
    # With drift, we expect measurable path length
    assert metrics['P_T_est'] > 0, "Path length should be positive with drift"
    assert metrics['oracle_refreshes'] > 1, "Oracle should refresh multiple times"
    print("✓ Controlled drift test passed")


def test_oracle_config_validation():
    """Test various oracle configuration options."""
    print("Testing oracle configuration validation...")
    
    dim = 3
    
    # Test different norms
    for norm in ['L1', 'L2']:
        cfg = SimpleConfig(
            enable_oracle=True,
            oracle_window_W=6,
            path_length_norm=norm
        )
        mp = MemoryPair(dim=dim, cfg=cfg)
        assert mp.oracle is not None
        assert mp.oracle.path_length_norm == norm
    
    # Test warmstart disabled
    cfg = SimpleConfig(
        enable_oracle=True,
        oracle_warmstart=False,
        oracle_window_W=8
    )
    mp = MemoryPair(dim=dim, cfg=cfg)
    assert not mp.oracle.oracle_warmstart
    
    # Test without oracle
    cfg = SimpleConfig(enable_oracle=False)
    mp = MemoryPair(dim=dim, cfg=cfg)
    assert mp.oracle is None
    
    print("✓ Oracle configuration validation passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Dynamic Comparator and Path-Length P_T Tests")
    print("=" * 60)
    
    try:
        test_rolling_oracle_basic()
        print()
        
        test_memory_pair_with_oracle()
        print()
        
        test_no_drift_scenario()
        print()
        
        test_drift_scenario()
        print()
        
        test_oracle_config_validation()
        print()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED! Dynamic Comparator is working correctly.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())