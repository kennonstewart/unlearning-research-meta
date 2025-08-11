"""
Tests for M7-M10 implementation.
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append('.')
sys.path.append('../../code/memory_pair/src')  
sys.path.append('../../code/data_loader')


def test_gamma_split_config():
    """Test unified gamma split configuration."""
    from config import Config
    
    # Test new gamma approach
    cfg = Config.from_cli_args(gamma_bar=2.0, gamma_split=0.7)
    assert abs(cfg.gamma_insert - 1.4) < 1e-10
    assert abs(cfg.gamma_delete - 0.6) < 1e-10
    assert abs(cfg.gamma_insert + cfg.gamma_delete - cfg.gamma_bar) < 1e-10
    
    # Test default values
    cfg_default = Config()
    assert cfg_default.gamma_bar == 1.0
    assert cfg_default.gamma_split == 0.5
    assert cfg_default.gamma_insert == 0.5
    assert cfg_default.gamma_delete == 0.5


def test_event_schema_diagnostics():
    """Test extended event schema with diagnostics."""
    from event_schema import create_event_record_with_diagnostics
    
    x = np.array([1.0, 2.0, 3.0])
    y = 1.5
    
    # Test basic record
    record = create_event_record_with_diagnostics(
        x=x, y=y, sample_id="test_001", event_id=1
    )
    
    assert record["x"].shape == (3,)
    assert record["y"] == 1.5
    assert record["sample_id"] == "test_001"
    assert record["event_id"] == 1
    assert "x_norm" in record["metrics"]
    
    # Test with diagnostics
    record_with_diag = create_event_record_with_diagnostics(
        x=x, y=y, sample_id="test_002", event_id=2,
        gamma_bar=2.0, gamma_split=0.7, lambda_est=0.5, P_T_true=1.2
    )
    
    assert record_with_diag["metrics"]["gamma_bar"] == 2.0
    assert record_with_diag["metrics"]["gamma_split"] == 0.7
    assert record_with_diag["metrics"]["lambda_est"] == 0.5
    assert record_with_diag["metrics"]["P_T_true"] == 1.2


def test_online_standardizer():
    """Test Welford's online standardization."""
    from covtype import OnlineStandardizer
    
    standardizer = OnlineStandardizer(d=3, clip_k=3.0)
    
    # Test single update
    x1 = np.array([1.0, 2.0, 3.0])
    x_std1, diag1 = standardizer.update_and_standardize(x1)
    
    assert x_std1.shape == (3,)
    assert "mean_l2" in diag1
    assert "std_l2" in diag1
    assert "clip_rate" in diag1
    
    # Test second update
    x2 = np.array([2.0, 3.0, 4.0])
    x_std2, diag2 = standardizer.update_and_standardize(x2)
    
    # Mean should have changed
    assert not np.allclose(diag1["mean_l2"], diag2["mean_l2"])
    assert standardizer.count == 2


def test_covariance_generator():
    """Test configurable covariance matrix generation."""
    from linear import CovarianceGenerator
    
    # Test with explicit eigenvalues
    eigs = [1.0, 0.5, 0.1]
    cov_gen = CovarianceGenerator(dim=3, eigs=eigs)
    
    assert cov_gen.eigenvalues.shape == (3,)
    assert np.allclose(cov_gen.eigenvalues, eigs)
    
    # Test sampling
    samples = cov_gen.sample(10)
    assert samples.shape == (10, 3)
    
    # Test with condition number
    cov_gen_cond = CovarianceGenerator(dim=3, cond_number=10.0)
    assert cov_gen_cond.eigenvalues[0] / cov_gen_cond.eigenvalues[-1] == pytest.approx(10.0, rel=1e-10)


def test_parameter_path_controller():
    """Test controlled parameter path generation."""
    from linear import ParameterPathController
    
    controller = ParameterPathController(dim=3, seed=42, path_type="rotating")
    
    # Test parameter evolution
    w1, delta1 = controller.get_next_parameter()
    w2, delta2 = controller.get_next_parameter()
    
    assert w1.shape == (3,)
    assert w2.shape == (3,)
    assert delta1 > 0  # Should have some path increment
    assert delta2 > 0
    assert controller.P_T_cumulative == delta1 + delta2


def test_strong_convexity_estimator():
    """Test online strong convexity estimation."""
    from linear import StrongConvexityEstimator
    
    estimator = StrongConvexityEstimator(ema_beta=0.1)
    
    # First update should return None (need two points)
    grad1 = np.array([1.0, 2.0])
    w1 = np.array([0.1, 0.2])
    lambda_est1 = estimator.update(grad1, w1)
    assert lambda_est1 is None
    
    # Second update should return estimate
    grad2 = np.array([1.1, 2.1])
    w2 = np.array([0.11, 0.21])
    lambda_est2 = estimator.update(grad2, w2)
    assert lambda_est2 is not None
    assert lambda_est2 > 0


def test_metrics_regret_decomposition():
    """Test regret decomposition computation."""
    from metrics import compute_regret_decomposition
    
    decomp = compute_regret_decomposition(
        cumulative_regret=100.0,
        G_hat=2.0, D_hat=3.0, c_hat=0.5, C_hat=5.0,
        S_T=50.0, P_T=10.0, T=1000, lambda_est=0.1
    )
    
    assert "R_static" in decomp
    assert "R_adaptive" in decomp  
    assert "R_path" in decomp
    assert "R_theory_total" in decomp
    assert "R_empirical" in decomp
    
    assert decomp["R_empirical"] == 100.0
    assert decomp["R_static"] > 0
    assert decomp["R_adaptive"] > 0
    assert decomp["R_path"] > 0


def test_metrics_S_T_slope():
    """Test S_T slope estimation."""
    from metrics import estimate_S_T_slope
    
    # Test linear growth
    S_T_linear = [i * 2.0 for i in range(100)]
    slope = estimate_S_T_slope(S_T_linear, window_K=50)
    assert abs(slope - 2.0) < 0.1  # Should be approximately 2.0
    
    # Test empty list
    slope_empty = estimate_S_T_slope([], window_K=10)
    assert slope_empty == 0.0


def test_linear_stream_generation():
    """Test enhanced linear stream with diagnostics."""
    from linear import get_synthetic_linear_stream
    
    stream = get_synthetic_linear_stream(
        dim=5, seed=42, strong_convexity_estimation=True, 
        path_control=True, eigs=[1.0, 0.5, 0.3, 0.2, 0.1]
    )
    
    # Test first few events
    events = []
    for i, event in enumerate(stream):
        events.append(event)
        if i >= 5:
            break
    
    assert len(events) == 6
    
    # Check structure
    for event in events:
        assert "x" in event
        assert "y" in event
        assert "metrics" in event
        assert event["x"].shape == (5,)
        
    # Check that lambda_est appears after first event
    lambda_ests = [e["metrics"].get("lambda_est") for e in events]
    assert lambda_ests[0] is None  # First should be None
    assert any(x is not None for x in lambda_ests[1:])  # Later ones should have estimates


def test_covtype_stream_standardization():
    """Test CovType stream with online standardization."""
    from covtype import get_covtype_stream
    
    # Use simulated data to avoid download issues
    stream = get_covtype_stream(online_standardize=True, use_event_schema=True)
    
    # Test first few events
    events = []
    for i, event in enumerate(stream):
        events.append(event)
        if i >= 3:
            break
    
    assert len(events) == 4
    
    # Check standardization diagnostics
    for event in events:
        assert "mean_l2" in event["metrics"]
        assert "std_l2" in event["metrics"]
        assert "clip_rate" in event["metrics"]
        assert "segment_id" in event
        

if __name__ == "__main__":
    # Run tests individually for easier debugging
    test_functions = [
        test_gamma_split_config,
        test_event_schema_diagnostics,
        test_online_standardizer,
        test_covariance_generator,
        test_parameter_path_controller,
        test_strong_convexity_estimator,
        test_metrics_regret_decomposition,
        test_metrics_S_T_slope,
        test_linear_stream_generation,
        test_covtype_stream_standardization,
    ]
    
    for test_func in test_functions:
        print(f"Running {test_func.__name__}...")
        try:
            test_func()
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()