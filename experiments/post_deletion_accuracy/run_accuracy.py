import json
import os
import sys
import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))
from data_loader import get_cifar10_stream, get_covtype_stream
from code.memory_pair.src.memory_pair import StreamNewtonMemoryPair as MemoryPair
from baselines import SekhariBatchUnlearning, QiaoHessianFree

ALGO_MAP = {
    "memorypair": MemoryPair,
    "sekhari": SekhariBatchUnlearning,
    "qiao": QiaoHessianFree,
}

DATASET_MAP = {
    "cifar10": get_cifar10_stream,
    "covtype": get_covtype_stream,
}

@click.command()
@click.option("--dataset", type=click.Choice(list(DATASET_MAP.keys())), required=True)
@click.option("--algo", type=click.Choice(list(ALGO_MAP.keys())), default="memorypair")
@click.option("--gamma", type=float, default=0.05)
@click.option("--seed", type=int, default=7)
def main(dataset, algo, gamma, seed):
    gen_fn = DATASET_MAP[dataset]
    gen = gen_fn(mode="iid", seed=seed)
    first_x, first_y = next(gen)
    dim = first_x.size
    model = ALGO_MAP[algo](dim)
    inserts = deletes = 0
    for step, (x, y) in enumerate(gen, start=1):
        model.insert(x, y)
        inserts += 1
        if step % 500 == 0:
            for _ in range(50):
                model.delete(x, y)
                deletes += 1
        if deletes >= 1000:
            break
    os.makedirs("results", exist_ok=True)
    out = {"inserts": inserts, "deletes": deletes}
    path = f"results/accuracy_{dataset}_{algo}_{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    os.system(f"git add {path}")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP:post_del_acc {dataset}-{algo} {hash_short}'")

if __name__ == "__main__":
    main()
