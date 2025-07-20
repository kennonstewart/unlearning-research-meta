# run.py
import csv
import os
import numpy as np
import sys
import click
from baselines import OnlineSGD, AdaGrad, OnlineNewtonStep
from plotting import plot_regret
import logging


def to_numpy(val):
    """Convert input to numpy array, handling strings (including scientific notation), scalars, and arrays."""
    import numpy as np

    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, str):
        # Handles scientific notation and comma-separated values
        try:
            arr = np.fromstring(val.replace("[", "").replace("]", ""), sep=",")
            if arr.size == 1:
                return arr[0]
            return arr
        except Exception:
            return np.array([float(val)])
    if isinstance(val, (float, int)):
        return np.array([val])
    return np.array(val)


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


def setup_logging():
    """Initialize logging configuration."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def initialize_data_stream(dataset, stream, seed):
    """Initialize the data stream generator and get the first sample."""
    gen_fn = DATASET_MAP[dataset]
    stream_gen = gen_fn(mode=stream, batch_size=1, seed=seed)
    first_x, first_y = next(stream_gen)
    first_x = to_numpy(first_x).reshape(-1)  # Flatten the first input
    first_y = to_numpy(first_y)
    return stream_gen, first_x, first_y


def initialize_model(algo, dim):
    """Initialize the model based on the algorithm choice."""
    algo_cls = ALGO_MAP[algo]
    return algo_cls(dim)


def calculate_loss(model, x, y):
    """Calculate the loss for a given model prediction."""
    prediction = np.dot(model.theta, x)
    return 0.5 * (prediction - y) ** 2


def setup_output_paths(dataset, stream, algo):
    """Setup output directory and file paths."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join("results", f"{dataset}_{stream}_{algo}.csv")
    png_path = csv_path.replace(".csv", ".png")
    return csv_path, png_path


def run_experiment(model, stream_gen, first_x, first_y, t, csv_path, logger):
    """Run the main experiment loop and write results to CSV."""
    cum_regret = 0.0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "regret"])

        # Process first sample
        x, y = first_x, first_y
        logger.info(f"Initial input shape: {x.shape}, label: {y}")
        model.insert(x, y)
        loss = calculate_loss(model, x, y)
        cum_regret += loss
        writer.writerow([1, cum_regret])

        # Process remaining samples
        for step, (x, y) in enumerate(stream_gen, start=2):
            x = to_numpy(x).reshape(-1)
            y = to_numpy(y)
            model.insert(x, y)
            loss = calculate_loss(model, x, y)
            cum_regret += loss

            if step % 100 == 0:
                writer.writerow([step, cum_regret])

            if step >= t:
                break

    return cum_regret


def finalize_experiment(csv_path, png_path, dataset, stream, algo):
    """Generate plots and commit results."""
    plot_regret(csv_path, png_path)
    os.system("git add results/*")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(
        f"git commit -m 'EXP:sublinear_regret {dataset}-{stream}-{algo} {hash_short}'"
    )


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
    # Setup
    logger = setup_logging()
    logger.info(f"Running {algo} on {dataset} with stream {stream} for {t} steps.")

    # Initialize data stream and model
    stream_gen, first_x, first_y = initialize_data_stream(dataset, stream, seed)
    dim = first_x.size
    model = initialize_model(algo, dim)

    # Setup output paths
    csv_path, png_path = setup_output_paths(dataset, stream, algo)

    # Run experiment
    final_regret = run_experiment(
        model, stream_gen, first_x, first_y, t, csv_path, logger
    )
    logger.info(
        f"Experiment completed. Final cumulative regret: {float(final_regret):.4f}"
    )

    # Finalize
    finalize_experiment(csv_path, png_path, dataset, stream, algo)


if __name__ == "__main__":
    main()
