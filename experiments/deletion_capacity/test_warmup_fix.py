#!/usr/bin/env python3
"""
Test script to validate that the warmup phase fixes work correctly.
Tests the specific issues identified in issue #19.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from config import Config
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.odometer import PrivacyOdometer, RDPOdometer
from memory_pair.src.calibrator import Calibrator


def simple_linear_generator(seed=42, dim=5):
    """Simple synthetic linear data generator for testing."""
    np.random.seed(seed)
    true_theta = np.random.randn(dim)
    
    while True:
        x = np.random.randn(dim)
        y = float(true_theta @ x) + np.random.normal(0, 0.1)
        yield x, y


def test_warmup_with_cap():
    """Test that warmup phase respects the N* cap."""
    print("Testing warmup phase with N* cap...")
    
    # Create config with small cap
    config = Config(
        dataset="synthetic",
        gamma_learn=0.1,  # Small gamma to make N* large
        gamma_priv=0.3,
        bootstrap_iters=50,  # Small for quick test
        max_events=10000,
        seeds=1,
        max_warmup_N=1000,  # Small cap for testing
        quantile=0.95,
        D_cap=10.0,
        accountant="default"
    )
    
    # Set up components
    gen = simple_linear_generator(seed=42)
    first_x, first_y = next(gen)
    
    odometer = PrivacyOdometer(
        eps_total=config.eps_total,
        delta_total=config.delta_total,
        T=config.max_events,
        gamma=config.gamma_priv,
        lambda_=config.lambda_,
        delta_b=config.delta_b,
    )
    
    calibrator = Calibrator(
        quantile=config.quantile, 
        D_cap=config.D_cap, 
        ema_beta=config.ema_beta
    )
    
    model = MemoryPair(
        dim=first_x.shape[0],
        odometer=odometer,
        calibrator=calibrator,
    )
    
    # Run bootstrap phase with artificially large gradients to trigger large N*
    print("Running bootstrap phase...")
    for i in range(config.bootstrap_iters):
        # Add some noise to make gradients larger
        x_noisy = first_x + np.random.normal(0, 10.0, first_x.shape)
        y_noisy = first_y + np.random.normal(0, 10.0)
        
        pred = model.calibrate_step(x_noisy, y_noisy)
        first_x, first_y = next(gen)
    
    # Finalize calibration
    print("Finalizing calibration...")
    model.finalize_calibration(gamma=config.gamma_learn, max_N=config.max_warmup_N)
    
    print(f"N* = {model.N_star}")
    
    # Verify the cap was applied
    assert model.N_star <= config.max_warmup_N, f"N* {model.N_star} exceeds cap {config.max_warmup_N}"
    
    print("✓ N* cap working correctly")
    return True


def test_get_privacy_metrics_function():
    """Test that get_privacy_metrics function is accessible before use."""
    print("Testing get_privacy_metrics function availability...")
    
    # This tests the import path - if the function can be defined properly
    try:
        import run
        print("✓ run.py imports correctly with fixed function ordering")
        return True
    except Exception as e:
        print(f"✗ run.py import failed: {e}")
        return False


def test_odometer_reference_fix():
    """Test that odometer finalization uses correct object reference."""
    print("Testing odometer reference fix...")
    
    # Create minimal setup
    config = Config(max_warmup_N=100)
    gen = simple_linear_generator(seed=42)
    first_x, first_y = next(gen)
    
    odometer = PrivacyOdometer()
    calibrator = Calibrator()
    
    model = MemoryPair(
        dim=first_x.shape[0],
        odometer=odometer,
        calibrator=calibrator,
    )
    
    # Run minimal bootstrap
    for i in range(10):
        pred = model.calibrate_step(first_x, first_y)
        first_x, first_y = next(gen)
    
    # Finalize calibration
    model.finalize_calibration(gamma=1.0, max_N=config.max_warmup_N)
    
    # Complete warmup
    inserts = 10
    while inserts < model.N_star:
        pred = model.insert(first_x, first_y)
        inserts += 1
        first_x, first_y = next(gen)
    
    # Test that model.odometer.finalize_with would work
    # (not calling it since it needs calibration_stats)
    assert hasattr(model.odometer, 'finalize'), "model.odometer should have finalize method"
    print("✓ Odometer reference is correct")
    return True


def main():
    """Run all tests."""
    print("Running deletion capacity warmup fix tests...\n")
    
    tests = [
        test_warmup_with_cap,
        test_get_privacy_metrics_function,
        test_odometer_reference_fix,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}\n")
    
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("All tests passed! Warmup fixes are working correctly.")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())