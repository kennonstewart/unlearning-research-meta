import csv
import json
import os
import sys
from typing import List

import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from data_loader import (
    get_rotating_mnist_stream,
    get_synthetic_linear_stream,
)
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.odometer import PrivacyOdometer
from baselines import SekhariBatchUnlearning, QiaoHessianFree

from metrics import regret, abs_error
from plots import plot_capacity_curve, plot_regret

ALGO_MAP = {
    "memorypair": MemoryPair,
    "sekhari": SekhariBatchUnlearning,
    "qiao": QiaoHessianFree,
}

DATASET_MAP = {
    "rot-mnist": get_rotating_mnist_stream,
    "synthetic": get_synthetic_linear_stream,
}


@click.command()
@click.option(
    "--dataset", type=click.Choice(["rot-mnist", "synthetic"]), default="rot-mnist"
)
# --- NEW OPTION ---
@click.option(
    "--gamma", type=float, default=0.5, help="Target average regret to end warmup."
)
@click.option(
    "--warmup-max-iter",
    type=int,
    default=10_000,
    help="Max iterations for warmup to prevent infinite loops.",
)
@click.option("--delete-ratio", type=float, default=10.0)
@click.option("--max-events", type=int, default=100_000)
@click.option("--seeds", type=int, default=10)
@click.option("--out-dir", type=click.Path(), default="results/")
@click.option("--algo", type=click.Choice(list(ALGO_MAP.keys())), default="memorypair")
def main(
    dataset: str,
    gamma: float,
    warmup_max_iter: int,
    delete_ratio: float,
    max_events: int,
    seeds: int,
    out_dir: str,
    algo: str,
) -> None:
    # Setup output directories
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figs")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Initialize data stream generator
    gen = DATASET_MAP[dataset](seed=0)  # Use seed=0 for deterministic warmup start
    first_x, first_y = next(gen)

    # Instantiate model and odometer
    model_class = ALGO_MAP[algo]
    odometer = PrivacyOdometer()

    # Initialize model (dim inferred from first_x)
    model = model_class(dim=first_x.shape[0], odometer=odometer)

    # Regret tracking
    cum_regret = 0.0

    for seed in range(seeds):
        summaries = []
        csv_paths: List[str] = []

        logs = []
        inserts = deletes = 0
        event = 0
        x, y = first_x, first_y

        # --- MODIFIED WARM-UP PHASE ---
        print(f"Starting warmup for seed {seed}. Target gamma: {gamma}")
        while event < warmup_max_iter and model.get_average_regret() > gamma:
            # The model now calculates regret internally.
            pred = model.insert(x, y)

            # We still need to log the metrics for plotting.
            acc_val = abs_error(pred, y)
            eps_spent = getattr(getattr(model, "odometer", None), "eps_spent", 0.0)

            logs.append(
                {
                    "event": event,
                    "op": "insert",
                    "regret": model.cumulative_regret,  # Get from model
                    "acc": acc_val,
                    "eps_spent": eps_spent,
                    "capacity_remaining": float("inf"),
                }
            )
            inserts += 1
            event += 1
            x, y = next(gen)

        print(
            f"Warmup complete after {event} events. Average regret: {model.get_average_regret():.4f}"
        )

        # workload phase
        while event < max_events:
            # k inserts
            for _ in range(int(delete_ratio)):
                pred = float(model.theta @ x)
                model.insert(x, y)
                cum_regret += regret(pred, y)
                acc_val = abs_error(pred, y)
                eps_spent = getattr(getattr(model, "odometer", None), "eps_spent", 0.0)
                remaining = getattr(
                    getattr(model, "odometer", None), "remaining", lambda: float("inf")
                )()
                logs.append(
                    {
                        "event": event,
                        "op": "insert",
                        "regret": cum_regret,
                        "acc": acc_val,
                        "eps_spent": eps_spent,
                        "capacity_remaining": remaining,
                    }
                )
                inserts += 1
                event += 1
                if event >= max_events:
                    break
                x, y = next(gen)
            if event >= max_events:
                break
            # delete
            try:
                pred = float(model.theta @ x)
                model.delete(x, y)
                deletes += 1
                cum_regret += regret(pred, y)
                acc_val = abs_error(pred, y)
                eps_spent = getattr(model.odometer, "eps_spent", 0.0)
                remaining = getattr(model.odometer, "remaining", lambda: 0.0)()
            except RuntimeError:
                break
            logs.append(
                {
                    "event": event,
                    "op": "delete",
                    "regret": cum_regret,
                    "acc": acc_val,
                    "eps_spent": eps_spent,
                    "capacity_remaining": remaining,
                }
            )
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        csv_path = os.path.join(runs_dir, f"{seed}_{algo}.csv")
        csv_paths.append(csv_path)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            writer.writerows(logs)

        summaries.append(
            {
                "inserts": inserts,
                "deletes": deletes,
                "eps_spent": getattr(
                    getattr(model, "odometer", None), "eps_spent", 0.0
                ),
                "final_regret": cum_regret,
            }
        )

    # aggregate summary
    def mean_ci(values: List[float]):
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        ci = float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))
        return mean, ci

    summary = {}
    for key in ["inserts", "deletes", "eps_spent", "final_regret"]:
        mean, ci = mean_ci([s[key] for s in summaries])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_ci95"] = ci

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"summary_{dataset}_{algo}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    plot_capacity_curve(csv_paths, os.path.join(figs_dir, "capacity_curve.pdf"))
    plot_regret(csv_paths, os.path.join(figs_dir, "regret.pdf"))

    os.system(f"git add {summary_path}")
    os.system(f"git add {figs_dir}/*.pdf")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP:del_capacity2 {dataset}-{algo} {hash_short}'")


if __name__ == "__main__":
    main()
