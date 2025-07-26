import csv
import os
import sys
import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code"))

# Assuming baselines are updated to have a similar .insert() interface
from baselines import OnlineSGD, AdaGrad, OnlineNewtonStep

# Import the new, more detailed plotting function
from plotting import plot_regret_with_bound

from data_loader import get_rotating_mnist_stream, get_covtype_stream

# Use the specific class name for clarity
from memory_pair.src.memory_pair import MemoryPair

ALGO_MAP = {
    # Updated class name for consistency
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
# --- General Experiment Options ---
@click.option("--dataset", type=click.Choice(["rotmnist", "covtype"]), required=True)
@click.option("--stream", type=click.Choice(["iid", "drift", "adv"]), default="iid")
@click.option("--algo", type=click.Choice(list(ALGO_MAP.keys())), default="memorypair")
@click.option("--t", "--T", type=int, default=100000)
@click.option("--seed", type=int, default=42)
# --- NEW: Theoretical Bound Parameters ---
@click.option(
    "--G",
    "g",
    type=float,
    default=1.0,
    help="Lipschitz constant for the loss gradient (G).",
)
@click.option(
    "--D", "d", type=float, default=1.0, help="Diameter of the hypothesis space (D)."
)
@click.option(
    "--c",
    type=float,
    default=0.1,
    help="Lower bound on Hessian approx eigenvalues (c).",
)
@click.option(
    "--C",
    "upper_C",
    type=float,
    default=10.0,
    help="Upper bound on Hessian approx eigenvalues (C).",
)
@click.option("--m", type=int, default=100, help="Deletion capacity (m).")
@click.option(
    "--L",
    "L_loss",
    type=float,
    default=1.0,
    help="Lipschitz constant for the loss function (L).",
)
@click.option(
    "--lam", type=float, default=0.01, help="Regularization parameter (lambda)."
)
@click.option(
    "--eps_star", type=float, default=1.0, help="Total privacy budget epsilon."
)
@click.option(
    "--delta_star", type=float, default=1e-5, help="Total privacy budget delta."
)
@click.option(
    "--delta_B",
    "delta_B",
    type=float,
    default=0.05,
    help="Confidence parameter for the regret bound.",
)
def main(
    dataset,
    stream,
    algo,
    t,
    seed,
    # Add new parameters to the function signature
    g,
    d,
    c,
    upper_C,
    m,
    L_loss,
    lam,
    eps_star,
    delta_star,
    delta_B,
):
    gen_fn = DATASET_MAP[dataset]
    stream_gen = gen_fn(mode=stream, batch_size=1, seed=seed)
    first_x, first_y = next(stream_gen)
    dim = first_x.size
    first_x = first_x.reshape(-1)
    algo_cls = ALGO_MAP[algo]
    model = algo_cls(dim)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Define file paths
    base_filename = f"{dataset}_{stream}_{algo}"
    csv_path = os.path.join(results_dir, f"{base_filename}.csv")
    png_path = os.path.join(results_dir, f"{base_filename}.png")

    cum_regret = 0.0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "regret"])
        x, y = first_x, first_y

        # The model's insert method should return the loss/regret for that step
        loss = model.insert(x, y)
        cum_regret += loss
        writer.writerow([1, cum_regret])

        for step, (x, y) in enumerate(stream_gen, start=2):
            if step > t:
                break
            x = x.reshape(-1)
            loss = model.insert(x, y)
            cum_regret += loss
            # Log periodically to keep CSV files manageable
            if step % 100 == 0:
                writer.writerow([step, cum_regret])
        # Ensure the final point is written
        writer.writerow([t, cum_regret])

    # --- MODIFIED PLOTTING CALL ---
    # Call the new function with all the required parameters
    print(f"Plotting results and theoretical bound to {png_path}...")
    plot_regret_with_bound(
        csv_paths=[csv_path],  # The new function can handle multiple paths
        out_path=png_path,
        G=g,
        D=d,
        c=c,
        C=upper_C,
        m=m,
        L=L_loss,
        lam=lam,
        eps_star=eps_star,
        delta_star=delta_star,
        delta_B=delta_B,
    )

    # --- Git commit logic remains the same ---
    os.system(f"git add {results_dir}/*")
    hash_short = os.popen("git rev-parse --short HEAD").read().strip()
    os.system(
        f"git commit -m 'EXP:sublinear_regret {dataset}-{stream}-{algo} {hash_short}'"
    )


if __name__ == "__main__":
    main()
