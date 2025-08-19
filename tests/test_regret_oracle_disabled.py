import numpy as np
from types import SimpleNamespace
import sys
import pathlib

base = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(base / "code" / "data_loader"))
sys.path.append(str(base / "code" / "memory_pair" / "src"))
from linear import get_synthetic_linear_stream  # type: ignore
from utils import set_global_seed  # type: ignore
from memory_pair import MemoryPair  # type: ignore


def test_regret_tracking_without_oracle():
    """Test that regret tracking works even when oracle is disabled."""
    set_global_seed(0)
    
    # Create config without oracle enabled
    cfg = SimpleNamespace(
        lambda_reg=0.1,
        dim=5,
        enable_oracle=False,  # Oracle disabled
    )
    
    model = MemoryPair(dim=cfg.dim, cfg=cfg)
    stream = get_synthetic_linear_stream(dim=5, seed=0, path_type="static")
    
    # Calibrate model
    for _ in range(20):
        rec = next(stream)
        model.calibrate_step(rec["x"], rec["y"])
    model.finalize_calibration(gamma=10.0)
    
    # Run some events
    for _ in range(50):
        rec = next(stream)
        model.insert(rec["x"], rec["y"])
    
    metrics = model.get_metrics_dict()
    avg_regret = metrics["avg_regret"]
    cum_regret = metrics["cum_regret"]
    
    # Should have finite, non-zero regret values even without oracle
    assert np.isfinite(avg_regret), "Average regret should be finite"
    assert avg_regret != 0.0, "Average regret should not be zero when oracle disabled"
    assert np.isfinite(cum_regret), "Cumulative regret should be finite"
    assert model.events_seen > 0, "Should have processed events"


def test_regret_tracking_with_oracle():
    """Test that regret tracking still works when oracle is enabled."""
    set_global_seed(0)
    
    # Create config with oracle enabled
    cfg = SimpleNamespace(
        lambda_reg=0.1,
        dim=5,
        enable_oracle=True,  # Oracle enabled
        comparator="dynamic",
        oracle_window_W=50,
        oracle_steps=5,
    )
    
    model = MemoryPair(dim=cfg.dim, cfg=cfg)
    stream = get_synthetic_linear_stream(dim=5, seed=0, path_type="static")
    
    # Calibrate model
    for _ in range(20):
        rec = next(stream)
        model.calibrate_step(rec["x"], rec["y"])
    model.finalize_calibration(gamma=10.0)
    
    # Run some events
    for _ in range(50):
        rec = next(stream)
        model.insert(rec["x"], rec["y"])
    
    metrics = model.get_metrics_dict()
    avg_regret = metrics["avg_regret"]
    cum_regret = metrics["cum_regret"]
    
    # Should have finite regret values with oracle
    assert np.isfinite(avg_regret), "Average regret should be finite"
    assert avg_regret > 0.0, "Average regret should be positive with oracle"
    assert np.isfinite(cum_regret), "Cumulative regret should be finite"
    assert model.events_seen > 0, "Should have processed events"