# /code/memory_pair/src/plotting.py
"""
Plotting utilities for dynamic regret decomposition visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional
import os


def plot_regret_decomposition(
    events: List[int],
    regret_static: List[float],
    regret_path: List[float],
    P_T_values: List[float],
    oracle_refreshes: Optional[List[int]] = None,
    title: str = "Dynamic Regret Decomposition",
    save_path: Optional[str] = None,
    show_theory_bound: bool = True,
    G: float = 1.0,
    lambda_param: float = 0.01,
    stepsize_policy: str = "strongly-convex",
    stepsize_params: Optional[Dict] = None,
    S_T_values: Optional[List[float]] = None,
    c: float = 1.0,
    C: float = 1.0,
) -> None:
    """
    Plot dynamic regret decomposition into static vs path terms.

    Args:
        events: List of event numbers
        regret_static: Static regret term values
        regret_path: Path regret term values
        P_T_values: Path-length P_T estimates
        oracle_refreshes: Event numbers where oracle was refreshed
        title: Plot title
        save_path: Optional path to save plot
        show_theory_bound: Whether to show theoretical bound
        G: Gradient norm bound for theory bound
        lambda_param: Strong convexity parameter for theory bound (strongly-convex policy)
        stepsize_policy: Step-size policy ("strongly-convex" or "adagrad")
        stepsize_params: Additional parameters for the policy
        S_T_values: Cumulative squared gradients for AdaGrad bound
        c: Lower curvature bound for AdaGrad
        C: Upper curvature bound for AdaGrad
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Regret decomposition
    ax1.plot(
        events,
        regret_static,
        label="Static Term (vs first oracle)",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        events,
        regret_path,
        label="Path Term (drift component)",
        color="red",
        linewidth=2,
    )

    total_regret = [s + p for s, p in zip(regret_static, regret_path)]
    ax1.plot(
        events,
        total_regret,
        label="Total Dynamic Regret",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    # Add oracle refresh markers
    if oracle_refreshes:
        for refresh_event in oracle_refreshes:
            if refresh_event <= max(events):
                ax1.axvline(
                    x=refresh_event,
                    color="green",
                    alpha=0.7,
                    linestyle=":",
                    label="Oracle Refresh"
                    if refresh_event == oracle_refreshes[0]
                    else "",
                )

    # Show theoretical bound if requested
    if show_theory_bound and len(events) > 0:
        T_values = np.array(events)
        
        if stepsize_policy == "strongly-convex":
            # Strongly-convex bound: O(G²/(λ c) log T + G P_T)
            log_term = (G**2 / (lambda_param * c)) * (1 + np.log(np.maximum(T_values, 1)))
            if len(P_T_values) == len(T_values):
                path_term = G * np.array(P_T_values)
                theory_bound = log_term + path_term
                bound_label = "Strongly-convex bound: O(G²/(λ c)(1+ln T) + G·P_T)"
            else:
                theory_bound = log_term
                bound_label = "Strongly-convex bound: O(G²/(λ c)(1+ln T))"
        
        elif stepsize_policy == "adagrad":
            # AdaGrad bound: O(G D √(c C S_T))
            if S_T_values is not None and len(S_T_values) == len(T_values):
                D = stepsize_params.get("D", 1.0) if stepsize_params else 1.0
                theory_bound = G * D * np.sqrt(c * C * np.array(S_T_values))
                bound_label = "Adaptive (AdaGrad) bound: O(G D √(c C S_T))"
            else:
                # Fallback if S_T values not available - use rough approximation
                D = stepsize_params.get("D", 1.0) if stepsize_params else 1.0
                # Assume S_T ≈ G² * T for rough visualization
                theory_bound = G * D * np.sqrt(c * C * G**2 * T_values)
                bound_label = "Adaptive (AdaGrad) bound: O(G D √(c C S_T)) [approx]"
        
        else:
            # Fallback to strongly-convex
            log_term = (G**2 / lambda_param) * np.log(np.maximum(T_values, 1))
            theory_bound = log_term
            bound_label = "Theory bound (default)"

        ax1.plot(
            events,
            theory_bound,
            label=bound_label,
            color="orange",
            linewidth=1,
            linestyle="-.",
        )

    ax1.set_xlabel("Event Number")
    ax1.set_ylabel("Regret")
    ax1.set_title(f"{title} - Regret Terms")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Path-length P_T
    ax2.plot(events, P_T_values, label="Path-Length P_T", color="purple", linewidth=2)

    # Add oracle refresh markers
    if oracle_refreshes:
        for refresh_event in oracle_refreshes:
            if refresh_event <= max(events):
                ax2.axvline(
                    x=refresh_event,
                    color="green",
                    alpha=0.7,
                    linestyle=":",
                    label="Oracle Refresh"
                    if refresh_event == oracle_refreshes[0]
                    else "",
                )

    ax2.set_xlabel("Event Number")
    ax2.set_ylabel("Path-Length P_T")
    ax2.set_title(f"{title} - Path-Length Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_oracle_diagnostics(
    events: List[int],
    oracle_objectives: List[float],
    oracle_w_norms: List[float],
    window_sizes: List[int],
    title: str = "Oracle Diagnostics",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot oracle diagnostic information.

    Args:
        events: Event numbers
        oracle_objectives: Oracle objective values at refresh
        oracle_w_norms: Oracle parameter norms at refresh
        window_sizes: Window sizes at refresh
        title: Plot title
        save_path: Optional save path
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Oracle objective values
    ax1.plot(events, oracle_objectives, "o-", color="blue", markersize=4)
    ax1.set_xlabel("Event Number")
    ax1.set_ylabel("Oracle Objective")
    ax1.set_title(f"{title} - Oracle Objective Values")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Oracle parameter norms
    ax2.plot(events, oracle_w_norms, "s-", color="red", markersize=4)
    ax2.set_xlabel("Event Number")
    ax2.set_ylabel("||w_star|| (Oracle Norm)")
    ax2.set_title(f"{title} - Oracle Parameter Norms")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Window sizes
    ax3.plot(events, window_sizes, "^-", color="green", markersize=4)
    ax3.set_xlabel("Event Number")
    ax3.set_ylabel("Window Size")
    ax3.set_title(f"{title} - Oracle Window Sizes")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Oracle diagnostics saved to {save_path}")

    plt.show()


def create_regret_decomposition_from_csv(
    csv_path: str,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Create regret decomposition plots from a CSV file with oracle metrics.

    Args:
        csv_path: Path to CSV file with oracle metrics
        title: Optional custom title
        save_dir: Optional directory to save plots
    """
    df = pd.read_csv(csv_path)

    # Determine path-length column name
    P_col = "P_T" if "P_T" in df.columns else "P_T_est"

    # Check for required columns
    required_cols = ["event", "regret_static_term", "regret_path_term", P_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return

    # Extract data
    events = df["event"].tolist()
    regret_static = df["regret_static_term"].tolist()
    regret_path = df["regret_path_term"].tolist()
    P_T_values = df[P_col].tolist()

    # Find oracle refresh events (where P_T increases)
    oracle_refreshes = []
    prev_P_T = 0
    for i, P_T in enumerate(P_T_values):
        if P_T > prev_P_T:
            oracle_refreshes.append(events[i])
            prev_P_T = P_T

    # Generate title
    if title is None:
        title = f"Dynamic Regret Decomposition - {os.path.basename(csv_path)}"

    # Save path
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_regret_decomposition.png")

    # Create plot
    plot_regret_decomposition(
        events=events,
        regret_static=regret_static,
        regret_path=regret_path,
        P_T_values=P_T_values,
        oracle_refreshes=oracle_refreshes,
        title=title,
        save_path=save_path,
    )

    # Oracle diagnostics if available
    if all(
        col in df.columns
        for col in ["oracle_objective", "oracle_w_norm", "window_size"]
    ):
        oracle_events = df[df["oracle_objective"].notna()]["event"].tolist()
        oracle_objectives = df[df["oracle_objective"].notna()][
            "oracle_objective"
        ].tolist()
        oracle_w_norms = df[df["oracle_w_norm"].notna()]["oracle_w_norm"].tolist()
        window_sizes = df[df["window_size"].notna()]["window_size"].tolist()

        diag_save_path = None
        if save_dir:
            diag_save_path = os.path.join(
                save_dir, f"{base_name}_oracle_diagnostics.png"
            )

        plot_oracle_diagnostics(
            events=oracle_events,
            oracle_objectives=oracle_objectives,
            oracle_w_norms=oracle_w_norms,
            window_sizes=window_sizes,
            title=title,
            save_path=diag_save_path,
        )


def analyze_regret_decomposition(
    regret_static: List[float],
    regret_path: List[float],
    P_T_values: List[float],
    events: List[int],
) -> Dict[str, float]:
    """
    Analyze regret decomposition and return summary statistics.

    Args:
        regret_static: Static regret term values
        regret_path: Path regret term values
        P_T_values: Path-length values
        events: Event numbers

    Returns:
        Dictionary with analysis results
    """
    if not regret_static or not regret_path or not P_T_values:
        return {}

    total_regret = [s + p for s, p in zip(regret_static, regret_path)]

    analysis = {
        "final_static_term": regret_static[-1],
        "final_path_term": regret_path[-1],
        "final_total_regret": total_regret[-1],
        "final_P_T": P_T_values[-1],
        "path_fraction": abs(regret_path[-1]) / (abs(total_regret[-1]) + 1e-10),
        "max_P_T": max(P_T_values),
        "avg_path_increment": P_T_values[-1] / len(P_T_values)
        if P_T_values[-1] > 0
        else 0,
        "total_events": len(events),
    }

    # Check if path term dominates (indicating significant drift)
    analysis["path_dominates"] = analysis["path_fraction"] > 0.5

    # Estimate drift rate
    if len(events) > 1 and P_T_values[-1] > 0:
        analysis["drift_rate_estimate"] = P_T_values[-1] / (events[-1] - events[0])
    else:
        analysis["drift_rate_estimate"] = 0.0

    return analysis
