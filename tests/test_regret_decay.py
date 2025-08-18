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


def _run_model(stream, cfg, T=200, calib_steps=20, gamma=10.0):
    model = MemoryPair(dim=cfg.dim, cfg=cfg)
    for _ in range(calib_steps):
        rec = next(stream)
        model.calibrate_step(rec["x"], rec["y"])
    model.finalize_calibration(gamma=gamma)
    return model


def test_stationary_regret_decay():
    set_global_seed(0)
    stream = get_synthetic_linear_stream(dim=5, seed=0, path_type="static", strong_convexity_estimation=False)
    cfg = SimpleNamespace(lambda_reg=0.1, dim=5)
    model = _run_model(stream, cfg)

    avg_regs = []
    for _ in range(200):
        rec = next(stream)
        model.insert(rec["x"], rec["y"])
        avg_regs.append(model.get_average_regret())

    assert avg_regs[-1] < 10.0


def test_drifting_regret_bound():
    set_global_seed(1)
    stream = get_synthetic_linear_stream(
        dim=5,
        seed=1,
        path_type="rotating",
        rotate_angle=0.001,
        strong_convexity_estimation=False,
    )
    cfg = SimpleNamespace(
        lambda_reg=0.1,
        dim=5,
        enable_oracle=True,
        oracle_window_W=50,
        oracle_steps=5,
    )
    model = _run_model(stream, cfg)

    T = 200
    for _ in range(T):
        rec = next(stream)
        model.insert(rec["x"], rec["y"])

    metrics = model.get_metrics_dict()
    avg_regret = metrics["avg_regret"]
    P_T = metrics.get("P_T", 0.0)
    bound = (np.log(T) / T) + (P_T / T)
    assert avg_regret <= bound * 200
