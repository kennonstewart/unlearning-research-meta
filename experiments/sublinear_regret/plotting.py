import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_regret_with_bound(
    csv_paths: list[str],
    out_path: str,
    G: float,
    D: float,
    c: float,
    C: float,
    m: int,
    L: float,
    lam: float,
    eps_star: float,
    delta_star: float,
    delta_B: float,
):
    """
    Plots empirical regret against the theoretical bound from Theorem 5.5.

    Args:
        csv_paths (list[str]): Paths to CSV files from experimental runs.
        out_path (str): Path to save the generated plot.
        G (float): Lipschitz constant for the loss gradient.
        D (float): Diameter of the hypothesis space.
        c (float): Lower bound on the Hessian approximation's eigenvalues.
        C (float): Upper bound on the Hessian approximation's eigenvalues.
        m (int): Deletion capacity (max number of deletions).
        L (float): Lipschitz constant for the loss function.
        lam (float): Regularization parameter (lambda).
        eps_star (float): Total privacy budget epsilon.
        delta_star (float): Total privacy budget delta.
        delta_B (float): Confidence parameter for the regret bound.
    """
    # --- 1. Load and Process Empirical Data ---
    all_regrets = []
    max_steps = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        # Ensure 'step' and 'regret' columns exist
        if "step" in df.columns and "regret" in df.columns:
            # Store the regret values, which are already cumulative
            all_regrets.append(df["regret"].values)
            if len(df["step"]) > max_steps:
                max_steps = len(df["step"])

    # Pad shorter runs to the length of the longest run
    padded_regrets = [np.pad(r, (0, max_steps - len(r)), "edge") for r in all_regrets]

    # Calculate mean and standard deviation across runs
    mean_regret = np.mean(padded_regrets, axis=0)
    std_regret = np.std(padded_regrets, axis=0)
    steps = np.arange(1, max_steps + 1)

    # --- 2. Calculate Theoretical Regret Bound ---
    # The bound is a function of T (time steps)
    T = steps

    # Theorem 5.5, part 2: R_T = O(sqrt(T)) + O(m)
    # Term 1: Noise-free regret from online L-BFGS (O(sqrt(T)))
    noise_free_regret = G * D * np.sqrt(c * C * T)

    # Term 2: Cumulative regret from injected noise (O(m))
    # This is a constant added to the sqrt(T) term
    noise_term = (
        (m * L / lam)
        * np.sqrt(2 * np.log(1.25 * m / delta_star) / eps_star)
        * np.sqrt(2 * np.log(1 / delta_B))
    )

    theoretical_bound = noise_free_regret + noise_term

    # --- 3. Generate the Plot ---
    plt.figure(figsize=(10, 6))

    # Plot the empirical regret (mean and std deviation)
    plt.loglog(steps, mean_regret, label="Empirical Cumulative Regret (Mean)")
    plt.fill_between(
        steps,
        mean_regret - std_regret,
        mean_regret + std_regret,
        alpha=0.2,
        label="Standard Deviation",
    )

    # Plot the theoretical regret bound
    plt.loglog(
        steps,
        theoretical_bound,
        color="r",
        linestyle="--",
        label="Theoretical Regret Bound",
    )

    # For reference, plot a simple sqrt(T) curve to check the growth rate
    # We scale it to start near the empirical curve for better visual comparison
    sqrt_t_scale = mean_regret[10] / np.sqrt(10)  # Adjust scale based on an early point
    plt.loglog(
        steps,
        sqrt_t_scale * np.sqrt(steps),
        linestyle=":",
        color="gray",
        label="O(sqrt(T)) Reference",
    )

    # --- Formatting ---
    plt.title("Empirical Regret vs. Theoretical Bound")
    plt.xlabel("Time Steps (T)")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    # Save the figure
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")
    plt.close()
