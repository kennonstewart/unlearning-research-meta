import numpy as np
from .utils import set_global_seed
from .streams import make_stream


def get_synthetic_linear_stream(dim=20, seed=42, noise_std=0.1):
    """Generate an infinite linear regression stream.

    Parameters
    ----------
    dim : int
        Feature dimension.
    seed : int
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    w_star = rng.normal(size=dim)
    X = rng.normal(size=(1000000, dim)).astype(np.float32)
    noise = rng.normal(scale=noise_std, size=1000000).astype(np.float32)
    y = X @ w_star + noise
    return make_stream(X, y, mode="iid", seed=seed)
