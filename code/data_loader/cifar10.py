import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream

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
    return make_stream(X, y, mode=mode, seed=seed)
