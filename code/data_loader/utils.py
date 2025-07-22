import hashlib
import os
import urllib.request
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def download_with_progress(url: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        return
    urllib.request.urlretrieve(url, target_path)
