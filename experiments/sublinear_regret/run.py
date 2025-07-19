# run.py
import csv
import os
import sys
import numpy as np
import click
from baselines import OnlineSGD, AdaGrad, OnlineNewtonStep
from plotting import plot_regret
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../../data"))
from data_loader import get_rotating_mnist_stream, get_covtype_stream

sys.path.append(os.path.join(os.path.dirname(__file__), "../../code"))
# StreamNewtonMemoryPair provides the online delete-insert functionality
from memory_pair.src.memory_pair import StreamNewtonMemoryPair as MemoryPair

ALGO_MAP = {
    "memorypair": MemoryPair,
    "sgd": OnlineSGD,
    "adagrad": AdaGrad,
    "ons": OnlineNewtonStep,
}

DATASET_MAP = {
    "rotmnist": get_rotating_mnist_stream,
    "covtype": get_covtype_stream,
}


@click.command()
@click.option("--dataset", type=click.Choice(["rotmnist", "covtype"]), required=True)
@click.option("--stream", type=click.Choice(["iid", "drift", "adv"]), default="iid")
@click.option("--algo", type=click.Choice(list(ALGO_MAP.keys())), default="memorypair")
@click.option("--t", "--T", type=int, default=100000)
@click.option("--seed", type=int, default=42)
def main(dataset, stream, algo, t, seed):
    """Run an experiment with the specified dataset, stream type, and algorithm.
    Args:
        dataset: Name of the dataset to use (rotmnist or covtype).
        stream: Type of data stream (iid, drift, adv).
        algo: Algorithm to use (memorypair, sgd, adagrad, ons).
        t: Number of steps to run the experiment.
        seed: Random seed for reproducibility.
    """

    # intialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running {algo} on {dataset} with stream {stream} for {t} steps.")

    gen_fn = DATASET_MAP[dataset]
    stream_gen = gen_fn(
        mode = stream,
        batch_size = 1, 
        seed = seed
    )
    first_x, first_y = next(stream_gen)
    logging.info(f"First input prior to reshape: {first_x}, label: {first_y}")
    first_x = first_x.reshape(-1) # Flatten the first input
    print(f"First input after reshape: {first_x}, label: {first_y}")
    dim = first_x.size
    algo_cls = ALGO_MAP[algo]
    model = algo_cls(dim)
    logger.info(f"Running {algo} on {dataset} with stream {stream} for {t} steps.")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{dataset}_{stream}_{algo}.csv")

    csv_path = os.path.join("results", f"{dataset}_{stream}_{algo}.csv")
    png_path = csv_path.replace(".csv", ".png")
    cum_regret = 0.0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "regret"])
        x, y = first_x, first_y
        logger.info(f"Initial input: {x}, label: {y}")
        # Call insert, then calculate loss manually
        model.insert(x, y)
        loss = 0.5 * (np.dot(model.theta, x) - y) ** 2
        cum_regret += loss
        writer.writerow([1, cum_regret])
        for step, (x, y) in enumerate(stream_gen, start=2):
            x = x.reshape(-1)
            model.insert(x, y)
            loss = 0.5 * (np.dot(model.theta, x) - y) ** 2
            cum_regret += loss
            if step % 100 == 0:
                writer.writerow([step, cum_regret])
            if step >= t:
                break

    plot_regret(csv_path, png_path)
    os.system("git add results/*")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(
        f"git commit -m 'EXP:sublinear_regret {dataset}-{stream}-{algo} {hash_short}'"
    )


if __name__ == "__main__":
    main()
