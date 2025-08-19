import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream
from .event_schema import create_event_record

try:
    from torchvision.datasets import CIFAR10
except Exception:  # noqa: BLE001
    CIFAR10 = None


def _simulate_cifar10(n=60000, seed=42):
    set_global_seed(seed)
    X = np.random.randint(0, 255, size=(n, 32, 32, 3), dtype=np.uint8)
    y = np.random.randint(0, 10, size=n, dtype=np.int64)
    return X, y


def download_cifar10(data_dir: str, split="train"):
    if CIFAR10 is None:
        return _simulate_cifar10()
    try:
        ds = CIFAR10(data_dir, train=split == "train", download=True)
        X = ds.data
        y = np.array(ds.targets)
        return X, y
    except Exception:
        return _simulate_cifar10()


def get_cifar10_stream(mode="iid", batch_size=1, seed=42):
    X, y = download_cifar10(os.path.expanduser("~/.cache/memory_pair_data"))
    X = X.reshape(len(X), -1)
    # Always emit event records
    base_stream = make_stream(X, y, mode=mode, seed=seed)
    event_id = 0
    for x, y_i in base_stream:
        sample_id = f"cifar10_{hash((x.tobytes(), int(y_i))) % 1000000:06d}"
        yield create_event_record(
            x=x,
            y=y_i,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=0,
            metrics=None,
        )
        event_id += 1
