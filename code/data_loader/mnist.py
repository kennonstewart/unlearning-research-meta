import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream
from .event_schema import create_event_record

try:
    from torchvision.datasets import MNIST
except Exception:  # noqa: BLE001
    MNIST = None


def _simulate_mnist(n=70000, seed=42):
    set_global_seed(seed)
    X = np.random.randint(0, 255, size=(n, 28, 28), dtype=np.uint8)
    y = np.random.randint(0, 10, size=n, dtype=np.int64)
    return X, y


def download_rotating_mnist(data_dir: str, split="train"):
    if MNIST is None:
        return _simulate_mnist()
    try:
        ds = MNIST(data_dir, train=split == "train", download=True)
        X = ds.data.numpy()
        y = ds.targets.numpy()
        return X, y
    except Exception:
        return _simulate_mnist()


def get_rotating_mnist_stream(mode="iid", batch_size=1, seed=42, use_event_schema=True):
    X, y = download_rotating_mnist(os.path.expanduser("~/.cache/memory_pair_data"))
    X = X.reshape(len(X), -1)
    
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
        sample_id = f"mnist_{hash((x.tobytes(), float(y))) % 1000000:06d}"
        yield create_event_record(
            x=x,
            y=y,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=0,
            metrics=None  # x_norm will be computed automatically
        )
        event_id += 1
