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
from typing import List, Tuple, Optional

import click
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from data_loader import (
    get_rotating_mnist_stream,
    get_synthetic_linear_stream,
)
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.odometer import PrivacyOdometer, RDPOdometer
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


def get_pred_and_grad(model, x, y, is_calibration=False):
    """Helper to obtain (pred, grad). Adjust for your API.
    For calibration phase, uses calibrate_step(). For other phases, uses insert().
    """
    if is_calibration and hasattr(model, "calibrate_step"):
        # During calibration, use calibrate_step and get gradient from last_grad
        pred = model.calibrate_step(x, y)
        if hasattr(model, "last_grad") and model.last_grad is not None:
            grad = model.last_grad
            return pred, grad
        else:
            raise RuntimeError("Model must expose last_grad after calibrate_step.")
    elif hasattr(model, "insert"):
        try:
            return model.insert(x, y, return_grad=True)
        except (TypeError, RuntimeError):
            # Fallback if insert doesn't return grad or is in wrong phase
            pred = model.insert(x, y)
            if hasattr(model, "last_grad") and model.last_grad is not None:
                grad = model.last_grad
            elif hasattr(model, "gradient"):
                grad = model.gradient(x, y)
            else:
                raise RuntimeError("Model must expose gradient to estimate G.")
            return pred, grad
    raise RuntimeError("Model has no insert or calibrate_step method.")


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
    "--gamma-learn",
    type=float,
    default=1.0,
    help="Target avg regret for learning (sample complexity N*).",
)
@click.option(
    "--gamma-priv",
    type=float,
    default=0.5,
    help="Target avg regret for privacy (odometer capacity m).",
)
@click.option(
    "--bootstrap-iters",
    type=int,
    default=500,
    help="Initial inserts to estimate G, D, c, C.",
)
@click.option("--delete-ratio", type=float, default=10.0, help="k inserts per delete.")
@click.option("--max-events", type=int, default=10_000_000)
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
@click.option(
    "--quantile", type=float, default=0.95, help="Quantile for robust G estimation."
)
@click.option(
    "--D-cap", type=float, default=10.0, help="Upper bound for hypothesis diameter."
)
@click.option(
    "--accountant",
    type=click.Choice(["rdp", "legacy"]),
    default="rdp",
    help="Privacy accountant type.",
)
@click.option(
    "--alphas",
    type=str,
    default="1.5,2,3,4,8,16,32,64",
    help="Comma-separated RDP orders for RDP accountant.",
)
@click.option(
    "--ema-beta",
    type=float,
    default=0.9,
    help="EMA decay parameter for drift detection.",
)
@click.option(
    "--recal-window",
    type=int,
    default=None,
    help="Events between recalibration checks (None = disabled).",
)
@click.option(
    "--recal-threshold",
    type=float,
    default=0.3,
    help="Relative threshold for drift detection.",
)
@click.option(
    "--m-max",
    type=int,
    default=None,
    help="Upper bound for deletion capacity binary search.",
)
def main(
    dataset: str,
    gamma_learn: float,
    gamma_priv: float,
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
    quantile: float,
    d_cap: float,
    accountant: str,
    alphas: str,
    ema_beta: float,
    recal_window: Optional[int],
    recal_threshold: float,
    m_max: Optional[int],
) -> None:
    # Setup output directories
    os.makedirs(out_dir, exist_ok=True)
    figs_dir = os.path.join(out_dir, "figs")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    print(f"[Config] gamma_learn = {gamma_learn}, gamma_priv = {gamma_priv}")
    print(f"[Config] quantile = {quantile}, D_cap = {d_cap}")
    print(f"[Config] accountant = {accountant}")
    print(
        f"[Config] EMA beta = {ema_beta}, recal_window = {recal_window}, recal_threshold = {recal_threshold}"
    )
    if m_max is not None:
        print(f"[Config] m_max = {m_max}")

    for seed in range(seeds):
        print(f"\n=== Seed {seed} ===")
        gen = DATASET_MAP[dataset](seed=seed)
        first_x, first_y = next(gen)

        # Parse alphas for RDP accountant
        if accountant == "rdp":
            alpha_list = [float(a.strip()) for a in alphas.split(",")]
            alpha_list.append(float("inf"))  # Always include infinity
            print(f"[Config] RDP alphas = {alpha_list}")

        # Instantiate odometer based on accountant type
        if accountant == "rdp":
            odometer = RDPOdometer(
                eps_total=eps_total,
                delta_total=delta_total,
                T=max_events,
                gamma=gamma_priv,  # Use privacy gamma for capacity computation
                lambda_=lambda_,
                delta_b=delta_b,
                alphas=alpha_list,
                m_max=m_max,
            )
        else:  # legacy
            odometer = PrivacyOdometer(
                eps_total=eps_total,
                delta_total=delta_total,
                T=max_events,
                gamma=gamma_priv,  # Use privacy gamma for capacity computation
                lambda_=lambda_,
                delta_b=delta_b,
            )

        # Create calibrator with robust parameters and EMA tracking
        from memory_pair.src.calibrator import Calibrator

        calibrator = Calibrator(quantile=quantile, D_cap=d_cap, ema_beta=ema_beta)

        # Initialize model
        model_class = ALGO_MAP[algo]
        if algo == "memorypair":
            model = model_class(
                dim=first_x.shape[0],
                odometer=odometer,
                calibrator=calibrator,
                recal_window=recal_window,
                recal_threshold=recal_threshold,
            )
        else:
            model = model_class(dim=first_x.shape[0], odometer=odometer)

        # Regret tracking
        cum_regret = 0.0

        summaries = []
        csv_paths: List[str] = []
        logs = []
        inserts = deletes = 0
        event = 0
        x, y = first_x, first_y

        # -------- Bootstrap/Calibration Phase -------- #
        print(
            f"[Bootstrap] Collecting {bootstrap_iters} steps to estimate G, D, c, C..."
        )

        # Use the new calibration API
        for _ in range(bootstrap_iters):
            pred, grad = get_pred_and_grad(model, x, y, is_calibration=True)
            gnorm = np.linalg.norm(grad)

            acc_val = abs_error(pred, y)
            # During calibration, odometer isn't finalized yet
            log_entry = {
                "event": event,
                "op": "calibrate",
                "regret": model.cumulative_regret
                if hasattr(model, "cumulative_regret")
                else np.nan,
                "acc": acc_val,
            }

            # Add default privacy metrics for calibration phase
            if accountant == "rdp":
                log_entry.update(
                    {
                        "eps_converted": 0.0,
                        "delta_total": odometer.delta_total,
                        "eps_remaining": odometer.eps_total,
                    }
                )
            else:
                log_entry.update(
                    {
                        "eps_spent": 0.0,
                        "capacity_remaining": float("inf"),
                    }
                )

            logs.append(log_entry)
            inserts += 1
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        # Finalize calibration using learning gamma
        print("[Bootstrap] Finalizing calibration...")
        model.finalize_calibration(gamma=gamma_learn)  # Use learning gamma for N*
        N_star = model.N_star

        print(
            f"[Warmup] Target sample complexity N* = {N_star} (current inserts={inserts})"
        )

        # Continue learning phase until N_star (if needed)
        while inserts < N_star and event < max_events:
            pred, grad = get_pred_and_grad(model, x, y, is_calibration=False)

            cum_regret += regret(pred, y)
            acc_val = abs_error(pred, y)

            # During warmup, odometer may not be finalized yet
            log_entry = {
                "event": event,
                "op": "insert",
                "regret": cum_regret,
                "acc": acc_val,
            }

            if odometer.ready_to_delete:
                log_entry.update(get_privacy_metrics(model.odometer))
            else:
                # Odometer not finalized yet
                if accountant == "rdp":
                    log_entry.update(
                        {
                            "eps_converted": 0.0,
                            "delta_total": odometer.delta_total,
                            "eps_remaining": odometer.eps_total,
                        }
                    )
                else:
                    log_entry.update(
                        {
                            "eps_spent": 0.0,
                            "capacity_remaining": float("inf"),
                        }
                    )

            logs.append(log_entry)
            inserts += 1
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        print(
            f"[Warmup] Complete after {inserts} inserts. Ready to predict: {model.can_predict}"
        )

        # NOW finalize odometer after warmup using total expected events
        print(f"[Warmup] Finalizing odometer with privacy gamma = {gamma_priv}")
        if hasattr(model, "calibration_stats") and model.calibration_stats:
            odometer.finalize_with(model.calibration_stats, T_estimate=max_events)
        else:
            print("[Warning] No calibration stats available, using legacy finalize")
            odometer.finalize()

        # Track empirical regret
        avg_regret_empirical = cum_regret / max(inserts, 1)
        print(f"[Empirical] Average regret after warmup: {avg_regret_empirical:.6f}")

        # Log theoretical vs empirical metrics
        theoretical_metrics = {
            "N_star_theory": N_star,
            "m_theory": odometer.deletion_capacity,
            "sigma_step_theory": odometer.sigma_step,
            "avg_regret_empirical": avg_regret_empirical,
            "accountant_type": accountant,
        }

        # Add recalibration information
        recal_stats = model.get_recalibration_stats()
        theoretical_metrics.update(
            {
                "recalibrations_count": recal_stats["recalibrations_count"],
                "current_G_ema": recal_stats.get("current_G_ema"),
                "finalized_G": recal_stats.get("finalized_G"),
            }
        )

        # Add accountant-specific metrics
        if accountant == "rdp":
            theoretical_metrics.update(
                {
                    "eps_total": odometer.eps_total,
                    "delta_total": odometer.delta_total,
                    "m_max_param": m_max,
                    "sens_bound": getattr(odometer, "sens_bound", None),
                }
            )
        else:
            theoretical_metrics.update(
                {
                    "eps_step_theory": odometer.eps_step,
                    "delta_step_theory": odometer.delta_step,
                }
            )

        print(f"[Theory vs Practice] {theoretical_metrics}")

        # Helper function to get privacy metrics for logging
        def get_privacy_metrics(odometer_obj):
            if isinstance(odometer_obj, RDPOdometer):
                eps_converted, delta_converted = odometer_obj.remaining_eps_delta()
                sens_stats = odometer_obj.get_sensitivity_stats()
                return {
                    "eps_converted": odometer_obj.eps_total - eps_converted,
                    "delta_total": delta_converted,
                    "eps_remaining": eps_converted,
                    "m_current": odometer_obj.deletion_capacity,
                    "sigma_current": odometer_obj.sigma_step,
                    "sens_count": sens_stats.get("count", 0),
                    "sens_mean": sens_stats.get("mean", 0.0),
                    "sens_max": sens_stats.get("max", 0.0),
                    "sens_q95": sens_stats.get("q95", 0.0),
                }
            else:
                return {
                    "eps_spent": getattr(odometer_obj, "eps_spent", 0.0),
                    "capacity_remaining": getattr(
                        odometer_obj, "remaining", lambda: float("inf")
                    )(),
                }

        # -------- Workload phase: interleave inserts/deletes -------- #
        while event < max_events:
            # k inserts
            for _ in range(int(delete_ratio)):
                pred = float(model.theta @ x)
                model.insert(x, y)  # no need for grad now
                cum_regret += regret(pred, y)
                acc_val = abs_error(pred, y)
                privacy_metrics = get_privacy_metrics(model.odometer)

                log_entry = {
                    "event": event,
                    "op": "insert",
                    "regret": cum_regret,
                    "acc": acc_val,
                }
                log_entry.update(privacy_metrics)
                logs.append(log_entry)
                inserts += 1
                event += 1
                if event >= max_events:
                    break
                x, y = next(gen)
            if event >= max_events:
                break

            # delete
            if odometer.deletion_capacity == 1 and odometer.deletions_count >= 1:
                print("[Delete] Capacity is 1; stopping deletes to avoid retrain.")
                break

            try:
                pred = float(model.theta @ x)
                model.delete(x, y)
                deletes += 1
                cum_regret += regret(pred, y)
                acc_val = abs_error(pred, y)
                privacy_metrics = get_privacy_metrics(model.odometer)

                if odometer.deletion_capacity == 1:
                    print("[Delete] Capacity exhausted after single delete. Stopping.")
                    log_entry = {
                        "event": event,
                        "op": "delete",
                        "regret": cum_regret,
                        "acc": acc_val,
                    }
                    log_entry.update(privacy_metrics)
                    logs.append(log_entry)
                    event += 1
                    break

            except RuntimeError as e:
                print(f"[Delete] {e}")
                break

            log_entry = {
                "event": event,
                "op": "delete",
                "regret": cum_regret,
                "acc": acc_val,
            }
            log_entry.update(privacy_metrics)
            logs.append(log_entry)
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        # -------- Write CSV for this seed -------- #
        csv_path = os.path.join(runs_dir, f"{seed}_{algo}.csv")
        csv_paths.append(csv_path)

        # Collect all possible fieldnames from all log entries
        all_fieldnames = set()
        for log_entry in logs:
            all_fieldnames.update(log_entry.keys())
        fieldnames = sorted(list(all_fieldnames))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

        # Build summary with base metrics
        summary_entry = {
            "inserts": inserts,
            "deletes": deletes,
            "final_regret": cum_regret,
            "avg_regret_empirical": cum_regret / max(inserts, 1),
            "N_star_theory": N_star,
            "m_theory": getattr(model.odometer, "deletion_capacity", None),
            "sigma_step_theory": getattr(model.odometer, "sigma_step", None),
            "G_hat": model.calibration_stats.get("G")
            if hasattr(model, "calibration_stats") and model.calibration_stats
            else None,
            "D_hat": model.calibration_stats.get("D")
            if hasattr(model, "calibration_stats") and model.calibration_stats
            else None,
            "c_hat": model.calibration_stats.get("c")
            if hasattr(model, "calibration_stats") and model.calibration_stats
            else None,
            "C_hat": model.calibration_stats.get("C")
            if hasattr(model, "calibration_stats") and model.calibration_stats
            else None,
            "gamma_learn": gamma_learn,
            "gamma_priv": gamma_priv,
            "quantile": quantile,
            "D_cap": d_cap,
            "accountant_type": accountant,
        }

        # Add accountant-specific metrics
        if accountant == "rdp":
            eps_converted, delta_remaining = model.odometer.remaining_eps_delta()
            summary_entry.update(
                {
                    "eps_total": model.odometer.eps_total,
                    "delta_total": model.odometer.delta_total,
                    "eps_converted": model.odometer.eps_total - eps_converted,
                    "eps_remaining": eps_converted,
                }
            )
        else:
            summary_entry.update(
                {
                    "eps_spent": getattr(model.odometer, "eps_spent", 0.0),
                    "eps_step_theory": getattr(model.odometer, "eps_step", None),
                    "delta_step_theory": getattr(model.odometer, "delta_step", None),
                }
            )

        summaries.append(summary_entry)

    # -------- Aggregate summary across seeds -------- #
    def mean_ci(values: List[float]):
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        ci = float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, ci

    summary = {}
    # Base keys that exist for both accountant types
    base_keys = [
        "inserts",
        "deletes",
        "final_regret",
        "avg_regret_empirical",
        "N_star_theory",
        "m_theory",
        "sigma_step_theory",
        "G_hat",
        "D_hat",
        "c_hat",
        "C_hat",
        "gamma_learn",
        "gamma_priv",
        "quantile",
        "D_cap",
    ]

    # Get all unique keys from summaries (handles both accountant types)
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())

    # Add all numeric keys
    for key in sorted(all_keys):
        if key in ["accountant_type"]:  # Skip non-numeric keys
            continue
        values = [
            s.get(key)
            for s in summaries
            if s.get(key) is not None and isinstance(s.get(key), (int, float))
        ]
        if values:
            mean, ci = mean_ci(values)
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
