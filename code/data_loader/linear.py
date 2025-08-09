import numpy as np
from .utils import set_global_seed
from .streams import make_stream
from .event_schema import create_event_record


def get_synthetic_linear_stream(dim=20, seed=42, noise_std=0.1, use_event_schema=True):
    """Generate an infinite linear regression stream.

    Parameters
    ----------
    dim : int
        Feature dimension.
    seed : int
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    use_event_schema : bool
        If True, emit event records. If False, emit legacy (x, y) tuples.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    w_star = rng.normal(size=dim)
    X = rng.normal(size=(1000000, dim)).astype(np.float32)
    noise = rng.normal(scale=noise_std, size=1000000).astype(np.float32)
    y = X @ w_star + noise
    
    if use_event_schema:
        # Wrap the stream to emit event records
        base_stream = make_stream(X, y, mode="iid", seed=seed)
        return _wrap_stream_with_schema(base_stream)
    else:
        # Legacy behavior
        return make_stream(X, y, mode="iid", seed=seed)


def _wrap_stream_with_schema(base_stream):
    """Wrap a legacy (x, y) stream to emit event records."""
    event_id = 0
    for x, y in base_stream:
        sample_id = f"linear_{hash((x.tobytes(), float(y))) % 1000000:06d}"
        yield create_event_record(
            x=x,
            y=y,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=0,
            metrics=None  # x_norm will be computed automatically
        )
        event_id += 1
