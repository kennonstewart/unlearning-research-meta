#!/usr/bin/env python3
"""
Test script for Milestone 5 accountant strategies.

Tests the new accountant strategy interface with different implementations:
- ZCDPAccountant
- EpsDeltaAccountant  
- RelaxedAccountant
"""

import sys
import os
import numpy as np

# Add the project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "memory_pair", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity"))

from accountant_strategies import create_accountant_strategy, ZCDPAccountant, EpsDeltaAccountant, RelaxedAccountant


def test_accountant_creation():
    """Test that all accountant types can be created."""
    print("\n=== Testing Accountant Creation ===")
    
    # Test factory function
    for accountant_type in ["zcdp", "eps_delta", "relaxed", "default", "legacy", "rdp"]:
        try:
            accountant = create_accountant_strategy(
                accountant_type=accountant_type,
                eps_total=1.0,
                delta_total=1e-5,
                T=1000,
                gamma=0.5,
                lambda_=0.1
            )
            print(f"✅ Created {accountant_type} -> {accountant.accountant_type}")
        except Exception as e:
            print(f"❌ Failed to create {accountant_type}: {e}")
    
    # Test direct instantiation
    try:
        zcdp = ZCDPAccountant(eps_total=1.0, delta_total=1e-5)
        eps_delta = EpsDeltaAccountant(eps_total=1.0, delta_total=1e-5)
        relaxed = RelaxedAccountant(eps_total=1.0, delta_total=1e-5, relaxation_factor=0.8)
        print(f"✅ Direct instantiation successful")
        print(f"   - ZCDP: {zcdp.accountant_type}")
        print(f"   - EpsDelta: {eps_delta.accountant_type}")
        print(f"   - Relaxed: {relaxed.accountant_type}")
    except Exception as e:
        print(f"❌ Direct instantiation failed: {e}")


def test_accountant_finalization():
    """Test that accountants can be finalized with stats."""
    print("\n=== Testing Accountant Finalization ===")
    
    stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
    T_estimate = 1000
    
    for accountant_type in ["zcdp", "eps_delta", "relaxed"]:
        try:
            accountant = create_accountant_strategy(
                accountant_type=accountant_type,
                eps_total=1.0,
                delta_total=1e-5,
                T=1000,
                gamma=0.5,
                lambda_=0.1
            )
            
            # Test metrics before finalization
            pre_metrics = accountant.metrics()
            print(f"✅ {accountant_type} pre-finalization metrics: {len(pre_metrics)} fields")
            
            # Finalize
            accountant.finalize_with(stats, T_estimate)
            
            # Test metrics after finalization
            post_metrics = accountant.metrics()
            print(f"✅ {accountant_type} post-finalization metrics: {len(post_metrics)} fields")
            print(f"   - Capacity: {accountant.deletion_capacity}")
            print(f"   - Noise scale: {accountant.noise_scale():.4f}")
            
        except Exception as e:
            print(f"❌ {accountant_type} finalization failed: {e}")


def test_accountant_spending():
    """Test that accountants can spend budget correctly."""
    print("\n=== Testing Accountant Budget Spending ===")
    
    stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
    T_estimate = 1000
    
    for accountant_type in ["zcdp", "eps_delta", "relaxed"]:
        try:
            accountant = create_accountant_strategy(
                accountant_type=accountant_type,
                eps_total=1.0,
                delta_total=1e-5,
                T=1000,
                gamma=0.5,
                lambda_=0.1
            )
            
            accountant.finalize_with(stats, T_estimate)
            
            initial_metrics = accountant.metrics()
            initial_count = accountant.deletions_count
            
            # Perform a spend operation
            sensitivity = 0.5
            sigma = accountant.noise_scale()
            
            if accountant_type == "eps_delta":
                accountant.spend()  # eps_delta doesn't use sensitivity/sigma
            else:
                accountant.spend(sensitivity, sigma)
            
            post_spend_metrics = accountant.metrics()
            post_count = accountant.deletions_count
            
            print(f"✅ {accountant_type} spending successful:")
            print(f"   - Deletions: {initial_count} -> {post_count}")
            print(f"   - Budget used: {post_spend_metrics.get('eps_spent', 'N/A')}")
            print(f"   - Remaining capacity: {post_spend_metrics.get('capacity_remaining', 'N/A')}")
            
        except Exception as e:
            print(f"❌ {accountant_type} spending failed: {e}")


def test_accountant_metrics_consistency():
    """Test that different accountants produce consistent metric structures."""
    print("\n=== Testing Metrics Consistency ===")
    
    stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
    T_estimate = 1000
    
    all_metrics = {}
    
    for accountant_type in ["zcdp", "eps_delta", "relaxed"]:
        try:
            accountant = create_accountant_strategy(
                accountant_type=accountant_type,
                eps_total=1.0,
                delta_total=1e-5,
                T=1000,
                gamma=0.5,
                lambda_=0.1
            )
            
            accountant.finalize_with(stats, T_estimate)
            metrics = accountant.metrics()
            all_metrics[accountant_type] = set(metrics.keys())
            
            print(f"✅ {accountant_type} metrics fields: {sorted(metrics.keys())}")
            
        except Exception as e:
            print(f"❌ {accountant_type} metrics failed: {e}")
    
    # Check for common fields
    if all_metrics:
        common_fields = set.intersection(*all_metrics.values())
        unique_fields = {}
        for acc_type, fields in all_metrics.items():
            unique_fields[acc_type] = fields - common_fields
        
        print(f"\n📊 Common fields across all accountants: {sorted(common_fields)}")
        for acc_type, fields in unique_fields.items():
            if fields:
                print(f"📊 {acc_type} unique fields: {sorted(fields)}")


def test_relaxed_noise_reduction():
    """Test that relaxed accountant actually reduces noise."""
    print("\n=== Testing Relaxed Noise Reduction ===")
    
    stats = {"G": 1.0, "D": 1.0, "c": 1.0, "C": 1.0}
    T_estimate = 1000
    
    try:
        # Create zCDP and relaxed accountants with same parameters
        zcdp = create_accountant_strategy("zcdp", eps_total=1.0, delta_total=1e-5, T=1000, gamma=0.5, lambda_=0.1)
        relaxed = create_accountant_strategy("relaxed", eps_total=1.0, delta_total=1e-5, T=1000, gamma=0.5, lambda_=0.1, relaxation_factor=0.8)
        
        zcdp.finalize_with(stats, T_estimate)
        relaxed.finalize_with(stats, T_estimate)
        
        zcdp_noise = zcdp.noise_scale()
        relaxed_noise = relaxed.noise_scale()
        
        print(f"✅ Noise comparison:")
        print(f"   - ZCDP noise: {zcdp_noise:.4f}")
        print(f"   - Relaxed noise: {relaxed_noise:.4f}")
        print(f"   - Reduction factor: {relaxed_noise / zcdp_noise:.4f}")
        
        if relaxed_noise < zcdp_noise:
            print(f"✅ Relaxed accountant successfully reduces noise")
        else:
            print(f"⚠️  Relaxed accountant did not reduce noise as expected")
            
    except Exception as e:
        print(f"❌ Relaxed noise test failed: {e}")


if __name__ == "__main__":
    print("Testing Milestone 5 Accountant Strategies")
    print("=" * 50)
    
    test_accountant_creation()
    test_accountant_finalization()
    test_accountant_spending()
    test_accountant_metrics_consistency()
    test_relaxed_noise_reduction()
    
    print("\n" + "=" * 50)
    print("Milestone 5 Accountant Strategy Tests Complete")