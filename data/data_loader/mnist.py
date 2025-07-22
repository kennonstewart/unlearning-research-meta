import os
import numpy as np
from .utils import set_global_seed
from .streams import make_stream

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


def get_rotating_mnist_stream(mode="iid", batch_size=1, seed=42):
    X, y = download_rotating_mnist(os.path.expanduser("~/.cache/memory_pair_data"))
    X = X.reshape(len(X), -1)
    return make_stream(X, y, mode=mode, seed=seed)
