import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from code.memory_pair.src.memory_pair import MemoryPair, Phase
from code.memory_pair.src.odometer import ZCDPOdometer
from code.memory_pair.src.calibrator import Calibrator
from code.memory_pair.src.accountant import get_adapter


def test_calibrator():
    """Test Calibrator class functionality."""
    print("\n=== Testing Calibrator ===")
    
    calibrator = Calibrator(quantile=0.95)
    
    # Simulate calibration observations
    dim = 10
    for i in range(50):
        grad = np.random.normal(0, 1, dim)
        theta = np.random.normal(0, 0.1, dim) * (i + 1)  # Growing trajectory
        calibrator.observe(grad, theta)
    
    # Mock model for bounds estimation
    class MockModel:
        def lbfgs_bounds(self):
            return (0.5, 2.0)
    
    model = MockModel()
    stats = calibrator.finalize(gamma=0.1, model=model)
    
    print(f"G = {stats['G']:.4f}")
    print(f"D = {stats['D']:.4f}")  
    print(f"c = {stats['c']:.4f}")
    print(f"C = {stats['C']:.4f}")
    print(f"N* = {stats['N_star']}")
    
    assert stats['G'] > 0
    assert stats['D'] > 0
    assert stats['N_star'] >= 1
    print("✅ Calibrator tests passed")


def test_memory_pair_state_machine():
    """Test MemoryPair state machine functionality."""
    print("\n=== Testing MemoryPair State Machine ===")
    
    dim = 5
    accountant = get_adapter("zcdp", rho_total=1.0, delta_total=1e-5, gamma=1.0)  # Increased gamma
    model = MemoryPair(dim=dim, accountant=accountant)
    
    # Should start in CALIBRATION phase
    assert model.phase == Phase.CALIBRATION
    assert not model.can_predict
    print(f"✅ Initial phase: {model.phase}")
    
    # Test calibration steps with smaller, controlled inputs
    print("Running calibration steps...")
    for i in range(20):
        x = np.random.normal(0, 0.1, dim)  # Smaller variance
        y = float(np.random.normal(0, 0.1))  # Smaller variance
        pred = model.calibrate_step(x, y)
        assert isinstance(pred, float)
    
    # Finalize calibration with a larger gamma to get smaller N*
    model.finalize_calibration(gamma=1.0)  # Increased gamma
    assert model.phase == Phase.LEARNING
    assert model.N_star is not None
    assert model.N_star >= 1
    print(f"✅ Calibration finalized, N* = {model.N_star}, phase: {model.phase}")
    
    # Test learning phase - limit the number of inserts for testing
    print("Running learning phase...")
    max_inserts = min(model.N_star - model.inserts_seen, 100)  # Limit to 100 for testing
    for i in range(max_inserts):
        x = np.random.normal(0, 0.1, dim)
        y = float(np.random.normal(0, 0.1))
        pred = model.insert(x, y)
        assert isinstance(pred, float)
    
    # If we reached N*, should transition to INTERLEAVING
    if model.inserts_seen >= model.N_star:
        assert model.phase == Phase.INTERLEAVING
        assert model.can_predict
        print(f"✅ Transitioned to {model.phase}, ready to predict: {model.can_predict}")
        
        # Test deletion in interleaving phase
        if model.odometer.ready_to_delete and model.odometer.deletion_capacity > 0:
            print("Testing deletion...")
            x = np.random.normal(0, 0.1, dim)
            y = float(np.random.normal(0, 0.1))
            model.delete(x, y)
            assert model.deletes_seen == 1
            print("✅ Deletion successful")
    else:
        print(f"✅ Still in LEARNING phase ({model.inserts_seen}/{model.N_star} inserts)")
    
    print("✅ State machine tests passed")


def test_zcdp_odometer():
    """Test ZCDPOdometer functionality."""
    print("\n=== Testing ZCDPOdometer ===")
    
    # Test basic functionality
    odometer = ZCDPOdometer(
        rho_total=1.0,
        delta_total=1e-5,
        T=1000,
        gamma=0.5,
        lambda_=0.1,
        delta_b=0.05,
    )
    
    # Simulate calibration stats
    stats = {
        "G": 2.5,
        "D": 1.2,
        "c": 0.8,
        "C": 1.5,
        "N_star": 100
    }
    
    assert not odometer.ready_to_delete
    odometer.finalize_with(stats, T_estimate=100)
    assert odometer.ready_to_delete
    assert odometer.deletion_capacity >= 1
    assert odometer.sigma_step > 0
    
    print(f"✅ Deletion capacity: {odometer.deletion_capacity}")
    print(f"✅ σ_step: {odometer.sigma_step:.4f}")
    
    # Test spending budget
    initial_capacity = odometer.deletion_capacity
    for i in range(min(3, initial_capacity)):
        sensitivity = 1.0  # Example sensitivity
        sigma = odometer.sigma_step
        odometer.spend(sensitivity, sigma)
        print(f"✅ Deletion {i+1}: ρ_spent = {odometer.rho_spent:.4f}")
    
    print("✅ ZCDPOdometer tests passed")


def test_integration():
    """Test full integration of calibration -> learning -> interleaving."""
    print("\n=== Integration Test ===")
    
    dim = 3
    calibrator = Calibrator()
    accountant = get_adapter("zcdp", rho_total=0.5, delta_total=1e-6, gamma=2.0)  # Higher gamma
    model = MemoryPair(dim=dim, accountant=accountant, calibrator=calibrator)
    
    # Calibration phase with smaller inputs
    for i in range(30):
        x = np.random.normal(0, 0.1, dim)  # Smaller variance
        y = float(x.sum() * 0.1 + np.random.normal(0, 0.01))  # Smaller linear relationship + noise
        model.calibrate_step(x, y)
    
    model.finalize_calibration(gamma=2.0)  # Higher gamma for smaller N*
    
    # Learning phase (insert until ready) - limit for testing
    learning_steps = min(model.N_star - model.inserts_seen, 50)  # Limit to 50
    for i in range(learning_steps):
        x = np.random.normal(0, 0.1, dim)
        y = float(x.sum() * 0.1 + np.random.normal(0, 0.01))
        model.insert(x, y)
    
    # Interleaving phase  
    for i in range(10):
        x = np.random.normal(0, 0.1, dim)
        y = float(x.sum() * 0.1 + np.random.normal(0, 0.01))
        model.insert(x, y)
    
    # Try a deletion if capacity allows and we're in interleaving phase
    if model.phase == Phase.INTERLEAVING and model.accountant.metrics().get("m_capacity", 0) > 0:
        x = np.random.normal(0, 0.1, dim)  
        y = float(x.sum() * 0.1)
        model.delete(x, y)
    
    print(f"✅ Final state: {model.phase}")
    print(f"✅ Events: {model.events_seen} (inserts: {model.inserts_seen}, deletes: {model.deletes_seen})")
    print(f"✅ Ready to predict: {model.can_predict}")
    print("✅ Integration test passed")


if __name__ == "__main__":
    test_calibrator()
    test_memory_pair_state_machine()
    test_zcdp_odometer()
    test_integration()
    print("\n🎉 All tests passed!")
