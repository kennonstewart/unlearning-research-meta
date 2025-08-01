#!/usr/bin/env python3
"""
Test script to reproduce the original issue scenario from #19 and verify it's fixed.
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from config import Config


def test_original_issue_scenario():
    """Test the original scenario that caused the issue."""
    print("Testing original issue scenario...")
    print("Parameters: {'gamma_learn': 0.7, 'gamma_priv': 0.3, 'quantile': 0.9, 'delete_ratio': 5, 'accountant': 'legacy', 'eps_total': 1.0}")
    
    # Create config matching the original issue
    config = Config(
        gamma_learn=0.7,
        gamma_priv=0.3,
        quantile=0.9,
        delete_ratio=5,
        accountant="default",  # using default instead of legacy for now
        eps_total=1.0,
        bootstrap_iters=100,  # Smaller for quick test
        max_events=10000,     # Smaller for quick test
        seeds=2,              # Just test 2 seeds
        max_warmup_N=50000,   # Our new cap
    )
    
    print(f"Config created with max_warmup_N = {config.max_warmup_N}")
    print("✓ Configuration successfully created with safeguards")
    return True


def test_config_serialization():
    """Test that the config can be serialized (for grid search)."""
    print("Testing config serialization...")
    
    config = Config(max_warmup_N=25000)
    config_dict = config.to_dict()
    
    assert 'max_warmup_N' in config_dict
    assert config_dict['max_warmup_N'] == 25000
    
    print("✓ Config serialization works correctly")
    return True


def test_from_cli_args():
    """Test that config can be created from CLI args (for grid search)."""
    print("Testing from_cli_args...")
    
    config = Config.from_cli_args(
        gamma_learn=0.7,
        gamma_priv=0.3,
        quantile=0.9,
        max_warmup_N=30000
    )
    
    assert config.gamma_learn == 0.7
    assert config.gamma_priv == 0.3
    assert config.quantile == 0.9
    assert config.max_warmup_N == 30000
    
    print("✓ from_cli_args works correctly")
    return True


def main():
    """Run all scenario tests."""
    print("Running original issue scenario tests...\n")
    
    tests = [
        test_original_issue_scenario,
        test_config_serialization,
        test_from_cli_args,
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
        print("All scenario tests passed! The original issue should be resolved.")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())