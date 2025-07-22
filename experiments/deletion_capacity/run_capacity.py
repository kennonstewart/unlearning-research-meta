import json
import os
import sys
import click
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), "../../code"))
from memory_pair.memory_pair import StreamNewtonMemoryPair as MemoryPair

sys.path.append(os.path.join(os.path.dirname(__file__), "../../data"))
from data_loader import get_rotating_mnist_stream


# Helper to robustly convert x, y to numpy arrays/scalars
def to_numpy(val):
    import numpy as np

    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, str):
        arr = np.fromstring(val.replace("[", "").replace("]", ""), sep=",")
        if arr.size == 1:
            return arr[0]
        return arr
    if isinstance(val, (float, int)):
        return np.array([val])
    return np.array(val)


class SekhariBatchUnlearning:
    def __init__(self, dim):
        self.theta = np.zeros(dim)

    def insert(self, x, y):
        pass

    def delete(self, x, y):
        pass


class QiaoHessianFree:
    def __init__(self, dim):
        self.theta = np.zeros(dim)

    def insert(self, x, y):
        pass

    def delete(self, x, y):
        pass


ALGO_MAP = {
    "memorypair": MemoryPair,
    "sekhari": SekhariBatchUnlearning,
    "qiao": QiaoHessianFree,
}


@click.command()
@click.option("--schedule", type=click.Choice(["burst", "trickle"]), default="burst")
@click.option("--seed", type=int, default=42)
@click.option("--algo", type=click.Choice(list(ALGO_MAP.keys())), default="memorypair")
@click.option("--max_events", type=int, default=10000)
def main(schedule, seed, algo, max_events):
    gen = get_rotating_mnist_stream(mode="iid", seed=seed)
    first_x, first_y = next(gen)
    dim = to_numpy(first_x).reshape(-1).size
    model = ALGO_MAP[algo](dim)
    inserts = deletes = 0
    for step, (x, y) in enumerate(gen, start=1):
        x = to_numpy(x).reshape(-1)
        y = to_numpy(y)
        model.insert(x, y)
        inserts += 1
        if schedule == "burst" and step % 500 == 0:
            for _ in range(250):
                model.delete(x, y)
                deletes += 1
        elif schedule == "trickle" and step % 100 == 0:
            if np.random.rand() < 0.01:
                model.delete(x, y)
                deletes += 1
        if step >= max_events:
            break
    os.makedirs("results", exist_ok=True)
    out = {"steps": step, "inserts": inserts, "deletes": deletes}
    path = f"results/capacity_{algo}_{schedule}_{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    os.system(f"git add {path}")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP:del_capacity {algo}-{schedule} {hash_short}'")


if __name__ == "__main__":
    main()
