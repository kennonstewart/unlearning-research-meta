import numpy as np
try:
    from .utils import set_global_seed
except ImportError:
    from utils import set_global_seed


def make_stream(X, y, mode="iid", drift_fn=None, adv_fn=None, seed=42):
    set_global_seed(seed)
    n = len(X)
    idx = np.arange(n)
    for i in range(n):
        if mode == "adv" and i % 500 == 0:
            np.random.shuffle(idx)
        if mode == "drift" and i % 1000 == 0 and drift_fn is not None:
            X = drift_fn(X)
        j = idx[i % n]
        yield X[j], y[j]
