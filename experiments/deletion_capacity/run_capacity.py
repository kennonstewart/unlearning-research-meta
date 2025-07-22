import json
import os
import sys
import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
from data_loader import get_rotating_mnist_stream
from memory_pair.src.memory_pair import StreamNewtonMemoryPair as MemoryPair
from baselines import SekhariBatchUnlearning, QiaoHessianFree

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
@click.option(
    "--warmup",
    type=int,
    default=5000,
    help="Number of initial points to insert before any deletions",
)
def main(schedule, seed, algo, max_events, warmup):
    gen = get_rotating_mnist_stream(mode="iid", seed=seed)
    first_x, first_y = next(gen)
    dim = first_x.size
    model = ALGO_MAP[algo](dim)
    # --- warm‑up phase -------------------------------------------------
    inserts = deletes = 0
    for _ in range(warmup):
        try:
            x_w, y_w = next(gen)
        except StopIteration:
            raise RuntimeError("Data stream exhausted before completing warm‑up.")
        model.insert(x_w, y_w)
        inserts += 1
    # -------------------------------------------------------------------
    for step, (x, y) in enumerate(gen, start=warmup + 1):
        try:
            model.delete(x, y)
            deletes += 1
        except RuntimeError as e:
            if "max_deletions" in str(e):
                print(f"[!] Privacy budget exceeded at step {step}. Stopping early.")
                break
            else:
                raise e
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
