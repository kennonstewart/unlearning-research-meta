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

import json
import math
import os
import sys
from typing import List, Tuple
from config import Config
import numpy as np
from pathlib import Path
from logger import RunLogger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

from data_loader import (
    get_rotating_mnist_stream,
    get_synthetic_linear_stream,
)
from memory_pair.src.memory_pair import MemoryPair
from memory_pair.src.calibrator import CalibSnapshot
from memory_pair.src.odometer import (
    PrivacyOdometer,
    ZCDPOdometer,
    rho_to_epsilon,
    N_star_live,
    m_theory_live,
)
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


def main():
    # Helper function to get privacy metrics for logging
    def get_privacy_metrics(odometer_obj):
        if isinstance(odometer_obj, ZCDPOdometer):
            rho_remaining, delta_total = odometer_obj.remaining_rho_delta()
            sens_stats = odometer_obj.get_sensitivity_stats()
            eps_converted = rho_to_epsilon(odometer_obj.rho_total - rho_remaining, delta_total)
            return {
                "eps_converted": eps_converted,
                "delta_total": delta_total,
                "rho_spent": odometer_obj.rho_spent,
                "rho_remaining": rho_remaining,
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

    def live_metrics(model, grad, x, update_bounds: bool = True):
        if update_bounds:
            snap = model.calibrator.live_bounds_update(
                {"grad_norm": float(np.linalg.norm(grad)), "x_norm": float(np.linalg.norm(x))}
            )
        else:
            snap = CalibSnapshot(
                G_hat=model.calibrator.G_hat_t,
                D_hat=model.calibrator.D_hat_t,
                c_hat=model.calibrator.c_hat if model.calibrator.c_hat is not None else 1.0,
                C_hat=model.calibrator.C_hat if model.calibrator.C_hat is not None else 1.0,
            )
        N_live = N_star_live(
            model.S_scalar,
            snap.G_hat,
            snap.D_hat,
            snap.c_hat,
            snap.C_hat,
            config.gamma_learn,
        )
        m_live = m_theory_live(
            model.S_scalar,
            model.t,
            snap.G_hat,
            snap.D_hat,
            snap.c_hat,
            snap.C_hat,
            config.gamma_priv,
            getattr(model.odometer, "sigma_step", 1.0),
        )
        return {
            "S_scalar": model.S_scalar,
            "eta_t": model.eta_t,
            "lambda_est": model.lambda_est,
            "sc_active": model.sc_active,
            "lr_mode": "sc" if model.sc_active else "adagrad",
            "N_star_live": N_live,
            "m_theory_live": m_live,
        }

    # Load experiment config
    config = Config()
    # alias config attributes to locals for backward-compatible references
    dataset = config.dataset
    algo = config.algo
    bootstrap_iters = config.bootstrap_iters
    delete_ratio = config.delete_ratio
    max_events = config.max_events
    seeds = config.seeds
    out_dir = config.out_dir
    eps_total = config.eps_total
    delta_total = config.delta_total
    lambda_ = config.lambda_
    delta_b = config.delta_b
    quantile = config.quantile
    d_cap = config.D_cap
    accountant = config.accountant
    alphas = config.alphas
    ema_beta = config.ema_beta
    recal_window = config.recal_window
    recal_threshold = config.recal_threshold
    m_max = config.m_max
    # Setup output directories
    os.makedirs(config.out_dir, exist_ok=True)
    figs_dir = os.path.join(config.out_dir, "figs")
    runs_dir = os.path.join(config.out_dir, "runs")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    print(
        f"[Config] gamma_learn = {config.gamma_learn}, gamma_priv = {config.gamma_priv}"
    )
    print(f"[Config] quantile = {config.quantile}, D_cap = {config.D_cap}")
    print(f"[Config] accountant = {config.accountant}")
    print(
        f"[Config] EMA beta = {config.ema_beta}, recal_window = {config.recal_window}, recal_threshold = {config.recal_threshold}"
    )
    if config.m_max is not None:
        print(f"[Config] m_max = {config.m_max}")

    summaries = []

    for seed in range(config.seeds):
        logger = RunLogger(seed, config.algo, Path(runs_dir))
        print(f"\n=== Seed {seed} ===")
        gen = DATASET_MAP[config.dataset](seed=seed)
        first_x, first_y = next(gen)

        if config.accountant == "rdp":
            alpha_list = config.alphas
            print(f"[Config] zCDP conversion from RDP alphas = {alpha_list}")
            # Convert eps_total to rho_total for zCDP
            rho_total = config.eps_total**2 / (2 * math.log(1 / config.delta_total))
            print(f"[Config] Converted eps_total={config.eps_total} to rho_total={rho_total:.6f}")

        if config.accountant == "rdp":
            odometer = ZCDPOdometer(
                rho_total=rho_total,
                delta_total=config.delta_total,
                T=config.max_events,
                gamma=config.gamma_priv,
                lambda_=config.lambda_,
                delta_b=config.delta_b,
                m_max=config.m_max,
            )
        else:
            odometer = PrivacyOdometer(
                eps_total=config.eps_total,
                delta_total=config.delta_total,
                T=config.max_events,
                gamma=config.gamma_priv,
                lambda_=config.lambda_,
                delta_b=config.delta_b,
            )

        from memory_pair.src.calibrator import Calibrator

        calibrator = Calibrator(
            quantile=config.quantile,
            D_cap=config.D_cap,
            ema_beta=config.ema_beta,
            trim_quantile=config.trim_quantile,
        )

        model_class = ALGO_MAP[config.algo]
        if config.algo == "memorypair":
            model = model_class(
                dim=first_x.shape[0],
                odometer=odometer,
                calibrator=calibrator,
                recal_window=config.recal_window,
                recal_threshold=config.recal_threshold,
                cfg=config,
            )
        else:
            model = model_class(dim=first_x.shape[0], odometer=odometer)

        cum_regret = 0.0
        csv_paths: List[str] = []
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
            acc_val = abs_error(pred, y)
            log_entry = {
                "event": event,
                "op": "calibrate",
                "regret": model.cumulative_regret
                if hasattr(model, "cumulative_regret")
                else np.nan,
                "acc": acc_val,
            }
            if config.accountant == "rdp":
                log_entry.update(
                    {
                        "eps_converted": 0.0,
                        "delta_total": odometer.delta_total,
                        "rho_remaining": odometer.rho_total,
                    }
                )
            else:
                log_entry.update(
                    {
                        "eps_spent": 0.0,
                        "capacity_remaining": float("inf"),
                    }
                )
            log_entry.update(live_metrics(model, grad, x))
            logger.log(log_entry)
            inserts += 1
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        # Finalize calibration using learning gamma
        print("[Bootstrap] Finalizing calibration...")
        model.finalize_calibration(
            gamma=config.gamma_learn
        )  # Use learning gamma for N*
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
                if config.accountant == "rdp":
                    log_entry.update(
                        {
                            "eps_converted": 0.0,
                            "delta_total": odometer.delta_total,
                            "rho_remaining": odometer.rho_total,
                        }
                    )
                else:
                    log_entry.update(
                        {
                            "eps_spent": 0.0,
                            "capacity_remaining": float("inf"),
                        }
                    )

            log_entry.update(live_metrics(model, grad, x))
            logger.log(log_entry)
            inserts += 1
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        print(
            f"[Warmup] Complete after {inserts} inserts. Ready to predict: {model.can_predict}"
        )

        # NOW finalize odometer after warmup using total expected events
        print(f"[Warmup] Finalizing odometer with privacy gamma = {config.gamma_priv}")
        if hasattr(model, "calibration_stats") and model.calibration_stats:
            model.odometer.finalize_with(model.calibration_stats, T_estimate=max_events)
        else:
            print("[Warning] No calibration stats available, using legacy finalize")
            model.odometer.finalize()

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
        if config.accountant == "rdp":
            theoretical_metrics.update(
                {
                    "rho_total": odometer.rho_total,
                    "delta_total": odometer.delta_total,
                    "m_max_param": config.m_max,
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

        # -------- Workload phase: interleave inserts/deletes -------- #
        while event < max_events:
            # k inserts
            for _ in range(int(delete_ratio)):
                pred, grad = model.insert(x, y, return_grad=True)
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
                log_entry.update(live_metrics(model, grad, x))
                logger.log(log_entry)
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
                grad = (pred - y) * x
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
                    log_entry.update(live_metrics(model, grad, x, update_bounds=False))
                    logger.log(log_entry)
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
            log_entry.update(live_metrics(model, grad, x, update_bounds=False))
            logger.log(log_entry)
            event += 1
            if event >= max_events:
                break
            x, y = next(gen)

        # -------- Write CSV for this seed -------- #
        csv_path = logger.flush()
        csv_paths.append(str(csv_path))

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
            "gamma_learn": config.gamma_learn,
            "gamma_priv": config.gamma_priv,
            "quantile": config.quantile,
            "D_cap": config.D_cap,
            "accountant_type": config.accountant,
        }

        # Add accountant-specific metrics
        if config.accountant == "rdp":
            rho_remaining, delta_remaining = model.odometer.remaining_rho_delta()
            eps_converted = rho_to_epsilon(model.odometer.rho_total - rho_remaining, delta_remaining)
            summary_entry.update(
                {
                    "rho_total": model.odometer.rho_total,
                    "delta_total": model.odometer.delta_total,
                    "eps_converted": eps_converted,
                    "rho_spent": model.odometer.rho_spent,
                    "rho_remaining": rho_remaining,
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

    summary_path = os.path.join(out_dir, f"summary_{dataset}_{algo}.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

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
