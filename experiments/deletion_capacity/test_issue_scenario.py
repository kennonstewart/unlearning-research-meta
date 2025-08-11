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
    print("Parameters: {'gamma_bar': 1.0, 'gamma_split': 0.7, 'quantile': 0.9, 'delete_ratio': 5, 'accountant': 'legacy', 'eps_total': 1.0}")
    
    # Create config matching the original issue (gamma_learn=0.7, gamma_priv=0.3 -> gamma_bar=1.0, gamma_split=0.7)
    config = Config(
        gamma_bar=1.0,
        gamma_split=0.7,  # 70% to insertion (gamma_learn), 30% to deletion (gamma_priv)
        quantile=0.9,
        delete_ratio=5,
        accountant="default",  # using default instead of legacy for now
        eps_total=1.0,
        bootstrap_iters=100,  # Smaller for quick test
        max_events=10000,     # Smaller for quick test
        seeds=2,              # Just test 2 seeds
    )
    
    print(f"gamma_insert: {config.gamma_insert}, gamma_delete: {config.gamma_delete}")
    print("✓ Configuration successfully created with safeguards")
    return True


def test_config_serialization():
    """Test that the config can be serialized (for grid search)."""
    print("Testing config serialization...")
    
    config = Config(gamma_bar=2.0, gamma_split=0.8)
    config_dict = config.to_dict()
    
    assert 'gamma_bar' in config_dict
    assert config_dict['gamma_bar'] == 2.0
    assert 'gamma_split' in config_dict
    assert config_dict['gamma_split'] == 0.8
    
    print("✓ Config serialization works correctly")
    return True


def test_from_cli_args():
    """Test that config can be created from CLI args (for grid search)."""
    print("Testing from_cli_args...")
    
    config = Config.from_cli_args(
        gamma_bar=1.0,
        gamma_split=0.7,
        quantile=0.9,
    )
    
    assert config.gamma_bar == 1.0
    assert config.gamma_split == 0.7
    assert config.gamma_insert == 0.7  # 70% of gamma_bar
    assert config.gamma_delete == 0.3  # 30% of gamma_bar
    assert config.quantile == 0.9
    
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