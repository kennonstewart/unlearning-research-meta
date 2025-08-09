import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream
from .event_schema import create_event_record

try:
    from sklearn.datasets import fetch_covtype
except Exception:  # noqa: BLE001
    fetch_covtype = None


def _simulate_covtype(n=581012, d=54, seed=42):
    set_global_seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(1, 8, size=n, dtype=np.int64)
    return X, y


def download_covtype(data_dir: str):
    if fetch_covtype is None:
        return _simulate_covtype()
    try:
        ds = fetch_covtype(data_home=data_dir)
        X = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        return X, y
    except Exception:
        return _simulate_covtype()


def get_covtype_stream(mode="iid", batch_size=1, seed=42, use_event_schema=True):
    X, y = download_covtype(os.path.expanduser("~/.cache/memory_pair_data"))
    
    if use_event_schema:
        # Wrap the stream to emit event records
        base_stream = make_stream(X, y, mode=mode, seed=seed)
        return _wrap_stream_with_schema(base_stream)
    else:
        # Legacy behavior
        return make_stream(X, y, mode=mode, seed=seed)


def _wrap_stream_with_schema(base_stream):
    """Wrap a legacy (x, y) stream to emit event records."""
    event_id = 0
    for x, y in base_stream:
        sample_id = f"covtype_{hash((x.tobytes(), float(y))) % 1000000:06d}"
        yield create_event_record(
            x=x,
            y=y,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=0,
            metrics=None  # x_norm will be computed automatically
        )
        event_id += 1
