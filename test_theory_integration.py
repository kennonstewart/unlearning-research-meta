#!/usr/bin/env python3
"""
Test script to verify theory-first integration works end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments", "deletion_capacity"))

from config import Config
from runner import _get_data_stream
from agents.grid_runner import create_grid_id

def test_theory_integration():
    """Test the complete theory-first integration."""
    print("Testing theory-first integration...")
    
    # Test 1: Config with theory parameters
    print("\n1. Testing config creation with theory parameters...")
    cfg = Config(
        dataset='synthetic',
        target_G=2.0,
        target_D=2.0,
        target_c=0.1,
        target_C=10.0,
        target_lambda=0.05,
        target_PT=120.0,
        target_ST=12000,
        path_style='rotating',
        rho_total=1.0,
        max_events=10
    )
    print(f"✓ Config created with target_G={cfg.target_G}, target_PT={cfg.target_PT}")
    
    # Test 2: Theory stream routing
    print("\n2. Testing theory stream routing...")
    stream = _get_data_stream(cfg, seed=42)
    event = next(stream)
    print(f"✓ Theory stream created, got event with keys: {list(event.keys())}")
    
    # Test 3: Legacy stream routing (no theory params)
    print("\n3. Testing legacy stream routing...")
    cfg_legacy = Config(dataset='synthetic', max_events=10)
    stream_legacy = _get_data_stream(cfg_legacy, seed=42)
    event_legacy = next(stream_legacy)
    print(f"✓ Legacy stream created, got event with keys: {list(event_legacy.keys())}")
    
    # Test 4: Grid ID generation
    print("\n4. Testing grid ID generation...")
    params_theory = {
        'gamma_bar': 1.0,
        'gamma_split': 0.5,
        'target_G': 2.0,
        'target_PT': 120.0,
        'target_ST': 12000,
        'path_style': 'rotating'
    }
    grid_id_theory = create_grid_id(params_theory)
    print(f"✓ Theory grid ID: {grid_id_theory}")
    
    params_legacy = {'gamma_bar': 1.0, 'gamma_split': 0.5}
    grid_id_legacy = create_grid_id(params_legacy)
    print(f"✓ Legacy grid ID: {grid_id_legacy}")
    
    # Verify theory parameters show up in theory grid ID but not legacy
    assert "PT120" in grid_id_theory, "Theory parameters should appear in theory grid ID"
    assert "PT120" not in grid_id_legacy, "Theory parameters should not appear in legacy grid ID"
    print("✓ Grid ID differentiation working correctly")
    
    print("\n🎉 All tests passed! Theory-first integration is working correctly.")

if __name__ == "__main__":
    test_theory_integration()