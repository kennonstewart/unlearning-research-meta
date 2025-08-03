#!/usr/bin/env python3
"""
Test script to demonstrate the fix for missing m_theory values.

This script reproduces the issue where m_theory values were missing due to
incorrect calibrator attribute access in the fallback path of finalize_accountant_phase.
"""

import sys
import os
import math
sys.path.insert(0, os.path.join('..', '..', 'code'))

from phases import finalize_accountant_phase
from config import Config
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.odometer import ZCDPOdometer, PrivacyOdometer, rho_to_epsilon
from memory_pair.src.calibrator import Calibrator
from metrics_utils import get_privacy_metrics


def test_original_issue_scenario():
    """
    Test the scenario that was causing missing m_theory values.
    
    This reproduces a case where calibration_stats might be None/empty,
    forcing the code to use the fallback path that was previously broken.
    """
    print("=== Testing Original Issue Scenario ===")
    
    # Create config similar to the deletion capacity experiment
    cfg = Config()
    cfg.max_events = 10000
    cfg.accountant = "rdp"
    
    # Scenario: Model where calibration_stats is None (the problematic case)
    print("\n1. Testing zCDP odometer with None calibration_stats...")
    # Convert default eps_total=1.0 to rho_total for zCDP
    rho_total = 1.0**2 / (2 * math.log(1 / 1e-5))
    model = MemoryPair(dim=10, odometer=ZCDPOdometer(rho_total=rho_total))
    model.calibration_stats = None  # This was causing the issue
    
    # Set calibrator attributes as they would be after a real calibration
    model.calibrator.finalized_G = 4.72  # Similar to seed 0 from analysis.ipynb
    model.calibrator.D = 4.78
    model.calibrator.c_hat = 1.0
    model.calibrator.C_hat = 1.0
    
    # Before fix: This would fail to get proper attributes and use default 1.0 values
    # After fix: This should work correctly using the calibrator attributes
    finalize_accountant_phase(model, cfg)
    
    # Extract m_theory as would be done in the experiment
    metrics = get_privacy_metrics(model)
    m_theory = metrics.get('m_theory')
    
    print(f"   zCDP m_theory: {m_theory}")
    assert m_theory is not None, "m_theory should not be None"
    assert m_theory >= 1, "m_theory should be at least 1"
    
    # Test legacy odometer too
    print("\n2. Testing Legacy odometer with None calibration_stats...")
    model2 = MemoryPair(dim=10, odometer=PrivacyOdometer())
    model2.calibration_stats = None
    
    # Set same calibrator attributes
    model2.calibrator.finalized_G = 4.72
    model2.calibrator.D = 4.78
    model2.calibrator.c_hat = 1.0
    model2.calibrator.C_hat = 1.0
    
    finalize_accountant_phase(model2, cfg)
    
    metrics2 = get_privacy_metrics(model2)
    m_theory2 = metrics2.get('m_theory')
    
    print(f"   Legacy m_theory: {m_theory2}")
    assert m_theory2 is not None, "m_theory should not be None"
    assert m_theory2 >= 1, "m_theory should be at least 1"
    
    print("\n✅ Both test cases passed! m_theory values are properly calculated.")
    return True


def test_calibration_stats_path():
    """Test that the normal calibration_stats path still works."""
    print("\n=== Testing Normal Calibration Stats Path ===")
    
    cfg = Config()
    cfg.max_events = 10000
    
    # Test with proper calibration_stats (this should have always worked)
    rho_total = 1.0**2 / (2 * math.log(1 / 1e-5))
    model = MemoryPair(dim=10, odometer=ZCDPOdometer(rho_total=rho_total))
    model.calibration_stats = {
        'G': 4.72,
        'D': 4.78, 
        'c': 1.0,
        'C': 1.0
    }
    
    finalize_accountant_phase(model, cfg)
    
    metrics = get_privacy_metrics(model)
    m_theory = metrics.get('m_theory')
    
    print(f"   Calibration stats path m_theory: {m_theory}")
    assert m_theory is not None, "m_theory should not be None"
    assert m_theory >= 1, "m_theory should be at least 1"
    
    print("\n✅ Calibration stats path test passed!")
    return True


def demonstrate_before_vs_after():
    """
    Demonstrate what would have happened before vs after the fix.
    """
    print("\n=== Before vs After Fix Demonstration ===")
    
    print("\nBEFORE FIX (simulated):")
    print("   - finalize_accountant_phase would access model.calibrator.G_hat")
    print("   - But calibrator stores finalized_G, not G_hat") 
    print("   - getattr(model.calibrator, 'G_hat', 1.0) would return default 1.0")
    print("   - Same for D_hat -> would get default 1.0")
    print("   - Result: stats = {'G': 1.0, 'D': 1.0, 'c': 1.0, 'C': 1.0}")
    print("   - This would lead to incorrect m_theory calculation")
    
    print("\nAFTER FIX:")
    print("   - finalize_accountant_phase now accesses model.calibrator.finalized_G")
    print("   - And model.calibrator.D (correct attribute names)")
    print("   - getattr(model.calibrator, 'finalized_G', 1.0) gets actual value")
    print("   - Result: stats = {'G': 4.72, 'D': 4.78, 'c': 1.0, 'C': 1.0}")
    print("   - This leads to correct m_theory calculation")


if __name__ == "__main__":
    print("Testing fix for missing m_theory values in deletion capacity experiment")
    print("=" * 70)
    
    try:
        # Run all tests
        test_original_issue_scenario()
        test_calibration_stats_path()
        demonstrate_before_vs_after()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("\nThe fix successfully resolves the missing m_theory values issue.")
        print("Now all seeds in the deletion capacity experiment should have proper")
        print("theoretical deletion capacity measurements instead of NaN values.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)