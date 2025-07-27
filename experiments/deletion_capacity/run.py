"""
Experiment 2 runner with:
  1) 500-step bootstrap to estimate G, D, and (c, C) from L-BFGS.
  2) Automatic computation of sample complexity N* (warmup length) using theory.
  3) Adaptive PrivacyOdometer finalization after warmup to derive deletion capacity, eps_step, etc.

Assumptions:
  - MemoryPair.insert(x, y, return_grad=True) returns (pred, grad). If not, see the helper
    `get_pred_and_grad` below and adapt to your actual API.
  - MemoryPair exposes either:
        * model.lbfgs.B_matrix()  -> returns current B approximation (d x d)
      or  * model.lbfgs_bounds()  -> returns (c_hat, C_hat)
    If not, we fallback to conservative c=1.0, C=1.0.
  - PrivacyOdometer is the adaptive version you implemented earlier.

Edit where marked if your APIs differ.
"""

import csv
import json
import os
import sys
from typing import List, Tuple

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


# ---------------------- Utilities ---------------------- #


def estimate_lbfgs_bounds(model) -> Tuple[float, float]:
    """Estimate eigenvalue bounds (c, C) of the L-BFGS Hessian approx.
    Try to access a matrix or bounds method; otherwise fallback to (1.0, 1.0).
    """
    # If the model provides direct bounds
    if hasattr(model, "lbfgs_bounds"):
        cC = model.lbfgs_bounds()
        if isinstance(cC, (tuple, list)) and len(cC) == 2:
            return float(cC[0]), float(cC[1])
    # If the model exposes B approximation
    if hasattr(model, "lbfgs") and hasattr(model.lbfgs, "B_matrix"):
        try:
            B = model.lbfgs.B_matrix()
            eigs = np.linalg.eigvalsh(B)
            c_hat = float(np.max([np.min(eigs), 1e-12]))  # avoid zero
            C_hat = float(np.max(eigs))
            return c_hat, C_hat
        except Exception:
            pass
    # Fallback conservative constants
    return 1.0, 1.0


def compute_sample_complexity(G, D, gamma, c=1.0, C=1.0, max_cap=None, q=None):
    # Optional: q is a quantile for G to reduce outliers (e.g. q=0.95)
    if isinstance(G, (list, np.ndarray)) and q is not None:
        G = float(np.quantile(G, q))
    N_star = int(np.ceil(((G * D * np.sqrt(c * C)) / gamma) ** 2))
    if max_cap is not None:
        N_star = min(N_star, max_cap)
    return max(N_star, 1)


def get_pred_and_grad(model, x, y):
    """Helper to obtain (pred, grad). Adjust for your API.
    Expected: model.insert(x, y, return_grad=True) -> (pred, grad)
    Otherwise, compute pred and grad manually if model exposes methods.
    """
    if hasattr(model, "insert"):
        try:
            return model.insert(x, y, return_grad=True)
        except TypeError:
            # Fallback if insert doesn't return grad
            pred = model.insert(x, y)
            if hasattr(model, "last_grad"):
                grad = model.last_grad
            elif hasattr(model, "gradient"):
                grad = model.gradient(x, y)
            else:
                raise RuntimeError("Model must expose gradient to estimate G.")
            return pred, grad
    raise RuntimeError("Model has no insert method.")


# ---------------------- CLI Config ---------------------- #

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
@click.option(
    "--gamma",
    type=float,
    default=0.5,
    help="Target avg regret (per-step) for theory bounds.",
)
@click.option(
    "--bootstrap-iters",
    type=int,
    default=500,
    help="Initial inserts to estimate G, D, c, C.",
)
@click.option("--delete-ratio", type=float, default=10.0, help="k inserts per delete.")
@click.option("--max-events", type=int, default=100_000)
@click.option("--seeds", type=int, default=10)
@click.option("--out-dir", type=click.Path(), default="results/")
@click.option("--algo", type=click.Choice(list(ALGO_MAP.keys())), default="memorypair")
@click.option("--eps-total", type=float, default=1.0)
@click.option("--delta-total", type=float, default=1e-5)
@click.option(
    "--lambda-strong",
    "lambda_",
    type=float,
    default=0.1,
    help="Strong convexity lower-bound.",
)
@click.option(
    "--delta-b", type=float, default=0.05, help="Failure prob for regret noise term."
)
def main(
    dataset: str,
    gamma: float,
    bootstrap_iters: int,
    delete_ratio: float,
    max_events: int,
    seeds: int,
    out_dir: str,
    algo: str,
    eps_total: float,
    delta_total: float,
    lambda_: float,
    delta_b: float,
) -> None:
    # Setup output directories
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figs")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    for seed in range(seeds):
        print(f"\n=== Seed {seed} ===")
        gen = DATASET_MAP[dataset](seed=seed)
        first_x, first_y = next(gen)

        # Instantiate odometer in adaptive mode; T is unknown yet but set to max_events for upper-bound
        odometer = PrivacyOdometer(
            eps_total=eps_total,
            delta_total=delta_total,
            T=max_events,
            gamma=gamma,
            lambda_=lambda_,
            delta_b=delta_b,
        )

        # Initialize model
        model_class = ALGO_MAP[algo]
        model = model_class(dim=first_x.shape[0], odometer=odometer)

        # Regret tracking
        cum_regret = 0.0

        summaries = []
        csv_paths: List[str] = []
        logs = []
        inserts = deletes = 0
        event = 0
        x, y = first_x, first_y

        # -------- Bootstrap to estimate G, D, c, C -------- #
        grads = []
        thetas = [model.theta.copy()]
        print(
            f"[Bootstrap] Collecting {bootstrap_iters} steps to estimate G, D, c, C..."
        )
        for _ in range(bootstrap_iters):
            pred, grad = get_pred_and_grad(model, x, y)
            gnorm = np.linalg.norm(grad)
            grads.append(gnorm)
            thetas.append(model.theta.copy())

            # Feed odometer for later L, D
            odometer.observe(grad, model.theta)

            acc_val = abs_error(pred, y)
            logs.append(
                {
                    "event": event,
                    "op": "insert",
                    "regret": model.cumulative_regret
                    if hasattr(model, "cumulative_regret")
                    else np.nan,
                    "acc": acc_val,
                    "eps_spent": odometer.eps_spent,
                    "capacity_remaining": float("inf"),
                }
            )
            inserts += 1
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        G_hat = float(max(grads)) if grads else 1.0
        D_hat = (
            float(max(np.linalg.norm(th - thetas[0]) for th in thetas))
            if len(thetas) > 1
            else 1.0
        )
        c_hat, C_hat = estimate_lbfgs_bounds(model)
        print(f"[Bootstrap] G={G_hat:.4f}, D={D_hat:.4f}, c={c_hat:.4e}, C={C_hat:.4e}")

        # -------- Compute sample complexity N* & finish warmup -------- #
        N_star = compute_sample_complexity(
            G_hat, D_hat, gamma, c_hat, C_hat, max_cap=max_events, q=0.95
        )
        print(
            f"[Warmup] Target sample complexity N* = {N_star} (current inserts={inserts})"
        )

        while inserts < N_star and event < max_events:
            pred, grad = get_pred_and_grad(model, x, y)
            odometer.observe(grad, model.theta)

            cum_regret += regret(pred, y)
            acc_val = abs_error(pred, y)
            logs.append(
                {
                    "event": event,
                    "op": "insert",
                    "regret": cum_regret,
                    "acc": acc_val,
                    "eps_spent": odometer.eps_spent,
                    "capacity_remaining": float("inf"),
                }
            )
            inserts += 1
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        print(f"[Warmup] Complete after {inserts} inserts.")

        # -------- Finalize odometer (compute m, eps_step, etc.) -------- #
        # We already observed grads/thetas through odometer.observe
        odometer.finalize()

        # -------- Workload phase: interleave inserts/deletes -------- #
        while event < max_events:
            # k inserts
            for _ in range(int(delete_ratio)):
                pred = float(model.theta @ x)
                model.insert(x, y)  # no need for grad now
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
                print("[Delete] Capacity exceeded. Stopping deletes.")
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

        # -------- Write CSV for this seed -------- #
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
                "N_star": N_star,
                "G_hat": G_hat,
                "D_hat": D_hat,
                "c_hat": c_hat,
                "C_hat": C_hat,
                "m_capacity": getattr(model.odometer, "deletion_capacity", None),
            }
        )

    # -------- Aggregate summary across seeds -------- #
    def mean_ci(values: List[float]):
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        ci = float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, ci

    summary = {}
    keys = [
        "inserts",
        "deletes",
        "eps_spent",
        "final_regret",
        "N_star",
        "G_hat",
        "D_hat",
        "c_hat",
        "C_hat",
        "m_capacity",
    ]
    for key in keys:
        mean, ci = mean_ci([s[key] for s in summaries if s[key] is not None])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_ci95"] = ci

    summary_path = os.path.join(out_dir, f"summary_{dataset}_{algo}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_capacity_curve(csv_paths, os.path.join(figs_dir, "capacity_curve.pdf"))
    plot_regret(csv_paths, os.path.join(figs_dir, "regret.pdf"))

    # Git stage & commit (optional)
    os.system(f"git add {summary_path}")
    os.system(f"git add {figs_dir}/*.pdf")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(f"git commit -m 'EXP2:auto_warmup {dataset}-{algo} {hash_short}'")


if __name__ == "__main__":
    main()
