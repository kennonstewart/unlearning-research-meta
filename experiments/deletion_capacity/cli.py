"""
CLI entry point for deletion capacity experiments.
Parses arguments and delegates to ExperimentRunner.
"""

import click
from config import Config
from runner import ExperimentRunner, ALGO_MAP


@click.command()
@click.option(
    "--dataset", type=click.Choice(["rot-mnist", "synthetic"]), default="rot-mnist"
)
@click.option(
    "--gamma-bar",
    type=float,
    default=1.0,
    help="Total regret budget (unified approach). Use with --gamma-split.",
)
@click.option(
    "--gamma-split",
    type=float,
    default=0.5,
    help="Fraction of gamma_bar allocated to insertions (learning). Default 0.5.",
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
    "--D-cap", "D_cap", type=float, default=10.0, help="Upper bound for hypothesis diameter."
)
@click.option(
    "--accountant",
    type=click.Choice(["zcdp"]),
    default="zcdp",
    help="Privacy accountant type (zCDP-only).",
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
@click.option(
    "--relaxation-factor",
    type=float,
    default=0.8,
    help="Relaxation factor for relaxed accountant (0.8 = 20% less noise).",
)
@click.option(
    "--sens-calib",
    type=int,
    default=50,
    help="Number of sensitivity samples for RDP calibration.",
)
@click.option(
    "--comparator",
    type=click.Choice(["static", "dynamic"]),
    default="dynamic",
    help="Comparator type: static (fixed w_0*) or dynamic (rolling oracle w_t*).",
)
@click.option(
    "--drift-threshold",
    type=float,
    default=0.1,
    help="Threshold for drift detection (relative P_T increase).",
)
@click.option(
    "--drift-kappa",
    type=float,
    default=0.5,
    help="Learning rate boost factor during drift: η_t *= (1 + κ).",
)
@click.option(
    "--drift-window",
    type=int,
    default=10,
    help="Duration of learning rate boost in steps after drift detection.",
)
@click.option(
    "--enable-oracle",
    is_flag=True,
    default=False,
    help="Enable oracle/comparator functionality for regret decomposition.",
)
@click.option(
    "--drift-adaptation",
    is_flag=True,
    default=False,
    help="Enable drift-responsive learning rate adaptation.",
)
@click.option("--target-g", "target_G", type=float, default=None, help="Theory-first: gradient norm bound G.")
@click.option("--target-d", "target_D", type=float, default=None, help="Theory-first: domain diameter D.")
@click.option("--target-c", "target_c", type=float, default=None, help="Theory-first: inverse-Hessian min eigenvalue clamp c.")
@click.option("--target-C", "target_C", type=float, default=None, help="Theory-first: inverse-Hessian max eigenvalue clamp C.")
@click.option("--target-lambda", "target_lambda", type=float, default=None, help="Theory-first: strong convexity λ.")
@click.option("--target-pt", "target_PT", type=float, default=None, help="Theory-first: total path length P_T.")
@click.option("--target-st", "target_ST", type=float, default=None, help="Theory-first: AdaGrad energy S_T (sum of squared gradients).")
@click.option("--rho-total", "rho_total", type=float, default=None, help="zCDP total privacy budget ρ (use with accountant=zcdp/rdp/relaxed).")
@click.option("--path-style", "path_style", type=click.Choice(["rotating", "brownian", "piecewise-constant"]), default="rotating", help="Theory-first path evolution style.")
def main(**kwargs):
    """Run deletion capacity experiment with zCDP accountant."""
    # Create config from CLI arguments
    cfg = Config.from_cli_args(**kwargs)
    
    # Create and run experiment (zCDP-only)
    runner = ExperimentRunner(cfg)
    runner.run_all()


if __name__ == "__main__":
    main()