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
    "--D-cap", "D_cap", type=float, default=10.0, help="Upper bound for hypothesis diameter."
)
@click.option(
    "--accountant",
    type=click.Choice(["default", "legacy", "eps_delta", "zcdp", "rdp", "relaxed"]),
    default="default",
    help="Privacy accountant type: default/legacy/eps_delta=(ε,δ)-DP, zcdp/rdp=zCDP, relaxed=experimental.",
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
def main(**kwargs):
    """Run deletion capacity experiment with configurable privacy accountant."""
    # Create config from CLI arguments
    cfg = Config.from_cli_args(**kwargs)
    
    # Create and run experiment
    runner = ExperimentRunner(cfg)
    runner.run_all()


if __name__ == "__main__":
    main()