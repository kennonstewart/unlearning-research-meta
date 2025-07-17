import json
import os
import click
import numpy as np
from data_loader import get_rotating_mnist_stream
from code.memory_pair.src.memory_pair import StreamNewtonMemoryPair as MemoryPair

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
    dim = first_x.size
    model = ALGO_MAP[algo](dim)
    inserts = deletes = 0
    for step, (x, y) in enumerate(gen, start=1):
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
