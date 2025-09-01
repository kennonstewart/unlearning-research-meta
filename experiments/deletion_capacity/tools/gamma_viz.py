#!/usr/bin/env python3
"""
Gamma (γ) adherence analysis and visualization utilities.

This module provides functions to:
- Load seed-level summaries across sweep directories
- Compute normalized regret ratios and pass flags
- Generate comprehensive visualizations for gamma adherence analysis

The gamma metrics are precomputed by process_seed_output in agents/grid_runner.py.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from scipy import stats


def load_seed_summaries(sweep_dir: str) -> pd.DataFrame:
    """
    Load all seed-level summaries across a sweep directory and normalize types.

    Args:
        sweep_dir: Path to sweep directory containing grid subdirectories

    Returns:
        DataFrame with all seed summaries, normalized types, and grid_id attached
    """
    all_rows = []

    # Find all grid subdirectories
    grid_dirs = [
        d for d in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, d))
    ]

    for grid_id in grid_dirs:
        grid_path = os.path.join(sweep_dir, grid_id)

        # Find seed CSV files (exclude events files)
        seed_files = glob.glob(os.path.join(grid_path, "seed_*.csv"))
        seed_files = [f for f in seed_files if "_events" not in f]

        for seed_file in seed_files:
            try:
                df = pd.read_csv(seed_file)
                if len(df) > 0:
                    # Take last row if multiple (though should be single row for seed summaries)
                    row = df.iloc[-1].copy()

                    # Extract seed number from filename
                    basename = os.path.basename(seed_file)
                    seed_match = [
                        part for part in basename.split("_") if part.isdigit()
                    ]
                    if seed_match:
                        row["seed"] = int(seed_match[0])

                    # Attach grid_id
                    row["grid_id"] = grid_id

                    all_rows.append(row)

            except Exception as e:
                print(f"Warning: Failed to load {seed_file}: {e}")
                continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Normalize types
    df = normalize_types(df)

    return df


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types in the dataframe.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with normalized types
    """
    df = df.copy()

    # Convert boolean columns from strings
    bool_cols = ["gamma_pass_overall", "gamma_pass_insert", "gamma_pass_delete"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = coerce_boolean(df[col])

    # Convert numeric columns
    numeric_cols = [
        "avg_regret_final",
        "gamma_bar_threshold",
        "gamma_split_threshold",
        "gamma_insert_threshold",
        "gamma_delete_threshold",
        "gamma_error",
        "PT_error",
        "ST_error",
        "rho_spent_final",
        "rho_total",
        "delete_ratio",
        "max_g_norm",
        "target_G",
        "avg_clip_rate",
        "target_PT",
        "target_ST",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def coerce_boolean(series: pd.Series) -> pd.Series:
    """
    Coerce a series to boolean, handling string representations.

    Args:
        series: Input series

    Returns:
        Boolean series
    """
    if series.dtype == bool:
        return series

    # Handle string representations
    series_str = series.astype(str).str.lower()
    return series_str.isin(["true", "1", "1.0", "yes"])


def compute_regret_ratio_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized regret ratio r = avg_regret_final / gamma_bar_threshold
    and boolean pass flags.

    Args:
        df: DataFrame with gamma metrics

    Returns:
        DataFrame with added columns: regret_ratio, regret_ratio_capped
    """
    df = df.copy()

    # Handle empty DataFrame
    if df.empty:
        df["regret_ratio"] = pd.Series(dtype=float)
        df["regret_ratio_capped"] = pd.Series(dtype=float)
        return df

    # Compute regret ratio (handle missing columns gracefully)
    if "avg_regret_final" in df.columns and "gamma_bar_threshold" in df.columns:
        df["regret_ratio"] = df["avg_regret_final"] / df["gamma_bar_threshold"]
        # Capped version for visualization (cap extreme outliers)
        df["regret_ratio_capped"] = df["regret_ratio"].clip(upper=5.0)
    else:
        df["regret_ratio"] = np.nan
        df["regret_ratio_capped"] = np.nan

    return df


def compute_at_clean_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AT-clean filter flags when columns are present.

    Args:
        df: DataFrame with experiment metrics

    Returns:
        DataFrame with added AT filter columns
    """
    df = df.copy()

    # Handle empty DataFrame
    if df.empty:
        for col in [
            "AT1_clean",
            "AT2_clean",
            "AT5_clean",
            "AT6_clean",
            "AT_clean_overall",
        ]:
            df[col] = pd.Series(dtype=bool)
        return df

    # AT1: PT_error ≤ 0.05
    if "PT_error" in df.columns:
        df["AT1_clean"] = df["PT_error"] <= 0.05
    else:
        df["AT1_clean"] = True  # Default pass if no data

    # AT2: max_g_norm ≤ 1.05*target_G and avg_clip_rate ≤ 0.05
    if (
        "max_g_norm" in df.columns
        and "target_G" in df.columns
        and "avg_clip_rate" in df.columns
    ):
        df["AT2_clean"] = (df["max_g_norm"] <= 1.05 * df["target_G"]) & (
            df["avg_clip_rate"] <= 0.05
        )
    else:
        df["AT2_clean"] = True

    # AT5: ST_error ≤ 0.05
    if "ST_error" in df.columns:
        df["AT5_clean"] = df["ST_error"] <= 0.05
    else:
        df["AT5_clean"] = True

    # AT6: rho_spent_final ≤ rho_total
    if "rho_spent_final" in df.columns and "rho_total" in df.columns:
        df["AT6_clean"] = df["rho_spent_final"] <= df["rho_total"]
    else:
        df["AT6_clean"] = True

    # Overall AT-clean flag
    df["AT_clean_overall"] = (
        df["AT1_clean"] & df["AT2_clean"] & df["AT5_clean"] & df["AT6_clean"]
    )

    return df


def load_event_logs(
    sweep_dir: str, grid_id: str, max_seeds: int = 4
) -> Dict[int, pd.DataFrame]:
    """
    Load event-level logs for time series plotting.

    Args:
        sweep_dir: Path to sweep directory
        grid_id: Grid ID to load
        max_seeds: Maximum number of seeds to load

    Returns:
        Dictionary mapping seed to event DataFrame
    """
    grid_path = os.path.join(sweep_dir, grid_id)

    # Prefer event files, fallback to seed files
    event_files = glob.glob(os.path.join(grid_path, "seed_*_events.csv"))
    if not event_files:
        event_files = glob.glob(os.path.join(grid_path, "seed_*.csv"))
        event_files = [f for f in event_files if "_events" not in f]

    event_data = {}

    for i, event_file in enumerate(sorted(event_files)[:max_seeds]):
        try:
            # Read only the columns we need for plotting to reduce I/O
            needed_cols = {"event", "avg_regret", "cum_regret", "regret", "op"}
            df = pd.read_csv(
                event_file,
                usecols=lambda c: c in needed_cols,
                engine="c",
            )

            # Downcast dtypes to reduce memory
            if "event" in df.columns:
                df["event"] = pd.to_numeric(
                    df["event"], errors="coerce", downcast="integer"
                )
            for col in ["avg_regret", "cum_regret", "regret"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
            if "op" in df.columns:
                try:
                    df["op"] = df["op"].astype("category")
                except Exception:
                    pass

            # Extract seed from filename
            basename = os.path.basename(event_file)
            seed_match = [part for part in basename.split("_") if part.isdigit()]
            if seed_match:
                seed = int(seed_match[0])

                # Only keep event-level data (multi-row)
                if len(df) > 1:
                    event_data[seed] = df

        except Exception as e:
            print(f"Warning: Failed to load {event_file}: {e}")
            continue

    return event_data


def compute_avg_regret_time_series(df: pd.DataFrame) -> pd.Series:
    """
    Compute average regret time series from event data.
    Preference: 'avg_regret' > 'cum_regret'/(event+1) > cumulative mean of 'regret'

    Args:
        df: Event-level DataFrame

    Returns:
        Series of average regret over time
    """
    if "avg_regret" in df.columns:
        return df["avg_regret"]
    elif "cum_regret" in df.columns and "event" in df.columns:
        return df["cum_regret"] / (df["event"] + 1)
    elif "regret" in df.columns:
        return df["regret"].expanding().mean()
    else:
        return pd.Series([np.nan] * len(df))


def plot_regret_time_series_with_phases(
    sweep_dir: str,
    grid_ids: List[str],
    gamma_bar: float,
    max_seeds: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    max_points: int = 2000,
) -> None:
    """
    Plot average regret time series with phase overlays for sampled grid_ids.

    Args:
        sweep_dir: Path to sweep directory
        grid_ids: List of grid IDs to plot
        gamma_bar: Gamma bar threshold for horizontal line
        max_seeds: Maximum seeds per grid
        figsize: Figure size
    """
    n_grids = len(grid_ids)
    fig, axes = plt.subplots(
        n_grids,
        1,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
        constrained_layout=True,
    )

    if n_grids == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, max_seeds))

    for i, grid_id in enumerate(grid_ids):
        ax = axes[i, 0]

        # Load event data
        event_data = load_event_logs(sweep_dir, grid_id, max_seeds)

        if not event_data:
            ax.text(
                0.5,
                0.5,
                f"No event data for {grid_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Grid: {grid_id}")
            continue

        # Plot each seed
        for j, (seed, df) in enumerate(event_data.items()):
            # Compute x-axis (prefer event column, fallback to index)
            if "event" in df.columns:
                x_vals = df["event"]
            else:
                x_vals = range(len(df))

            # Compute average regret time series
            avg_regret_series = compute_avg_regret_time_series(df)

            # Downsample to at most max_points to speed up plotting
            n = len(avg_regret_series)
            if n > max_points:
                step = max(1, n // max_points)
                x_plot = (
                    x_vals.iloc[::step] if hasattr(x_vals, "iloc") else x_vals[::step]
                )
                y_plot = avg_regret_series.iloc[::step]
            else:
                x_plot = x_vals
                y_plot = avg_regret_series

            ax.plot(
                x_plot,
                y_plot,
                color=colors[j],
                label=f"Seed {seed}",
                alpha=0.8,
                linewidth=1.2,
                rasterized=True,
                antialiased=False,
            )

            # Add phase overlays (if op column exists)
            if "op" in df.columns:
                add_phase_overlays(ax, df, x_vals)

        # Add gamma bar line
        ax.axhline(
            y=gamma_bar,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"γ̄ = {gamma_bar:.3f}",
        )

        ax.set_xlabel("Event ID" if "event" in df.columns else "Event Index")
        ax.set_ylabel("Average Regret")
        ax.set_title(f"Grid: {grid_id}")
        # Keep legend lightweight (seed lines only)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, alpha=0.3)

    plt.show()


def add_phase_overlays(ax, df: pd.DataFrame, x_vals: pd.Series) -> None:
    """
    Add phase overlays to time series plot based on 'op' column.

    Args:
        ax: Matplotlib axis
        df: Event DataFrame with 'op' column
        x_vals: X-axis values
    """
    if "op" not in df.columns:
        return

    # Define phase colors (merge insert/delete into a single 'edit' phase)
    phase_colors = {
        "calibrate": "yellow",
        "warmup": "orange",
        "edit": "lightcyan",
    }

    # Group consecutive operations
    # Normalize ops: treat 'insert' and 'delete' as a combined 'edit' phase
    def normalize_op(op_val: str) -> str:
        if pd.isna(op_val):
            return "unknown"
        ov = str(op_val)
        return "edit" if ov in ("insert", "delete") else ov

    current_op = None
    start_idx = 0
    labels_added = set()

    ops_series = df["op"]
    for i, op in enumerate(ops_series):
        norm_op = normalize_op(op)
        if norm_op != current_op:
            # End previous phase span
            if current_op is not None and i > start_idx:
                color = phase_colors.get(current_op, "lightgray")
                label = current_op if current_op not in labels_added else "_nolegend_"
                ax.axvspan(
                    x_vals.iloc[start_idx],
                    x_vals.iloc[i - 1],
                    alpha=0.15,
                    color=color,
                    label=label,
                )
                labels_added.add(current_op)

            # Start new phase
            current_op = norm_op
            start_idx = i

    # Final span
    if current_op is not None and len(df) > start_idx:
        color = phase_colors.get(current_op, "lightgray")
        label = current_op if current_op not in labels_added else "_nolegend_"
        ax.axvspan(
            x_vals.iloc[start_idx],
            x_vals.iloc[len(df) - 1],
            alpha=0.15,
            color=color,
            label=label,
        )


def plot_regret_ratio_ecdf(
    df: pd.DataFrame,
    at_clean_subset: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot ECDF of regret ratio with r=1 threshold line.

    Args:
        df: DataFrame with regret_ratio column
        at_clean_subset: Optional AT-clean subset to overlay
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Main data
    if "regret_ratio" in df.columns:
        ratios = df["regret_ratio"].dropna()
        if len(ratios) > 0:
            sorted_ratios = np.sort(ratios)
            y_vals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
            ax.plot(
                sorted_ratios,
                y_vals,
                label=f"All seeds (n={len(ratios)})",
                linewidth=2,
                alpha=0.8,
            )

    # AT-clean subset
    if at_clean_subset is not None and "regret_ratio" in at_clean_subset.columns:
        ratios_clean = at_clean_subset["regret_ratio"].dropna()
        if len(ratios_clean) > 0:
            sorted_ratios_clean = np.sort(ratios_clean)
            y_vals_clean = np.arange(1, len(sorted_ratios_clean) + 1) / len(
                sorted_ratios_clean
            )
            ax.plot(
                sorted_ratios_clean,
                y_vals_clean,
                label=f"AT-clean subset (n={len(ratios_clean)})",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
            )

    # Add r=1 threshold line
    ax.axvline(
        x=1.0, color="red", linestyle="--", linewidth=2, label="r = 1 (pass threshold)"
    )

    ax.set_xlabel("Regret Ratio r = avg_regret_final / gamma_bar")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("ECDF of Regret Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regret_ratio_by_grid(
    df: pd.DataFrame, top_n: int = 20, figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot violin/box of regret ratio by grid_id, sorted by median r.

    Args:
        df: DataFrame with regret_ratio and grid_id columns
        top_n: Number of worst grids to show
        figsize: Figure size (auto-computed if None)
    """
    if "regret_ratio" not in df.columns or "grid_id" not in df.columns:
        print("Missing required columns: regret_ratio, grid_id")
        return

    # Compute median regret ratio per grid
    grid_stats = (
        df.groupby("grid_id")["regret_ratio"].agg(["median", "count"]).reset_index()
    )

    # Sort by worst median and take top N
    worst_grids = grid_stats.sort_values("median", ascending=False).head(top_n)

    # Filter data
    plot_data = df[df["grid_id"].isin(worst_grids["grid_id"])].copy()

    # Set figure size
    if figsize is None:
        figsize = (max(12, top_n * 0.6), 8)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Create violin plot
    sns.violinplot(
        data=plot_data,
        y="grid_id",
        x="regret_ratio",
        order=worst_grids["grid_id"],
        ax=ax,
        inner="box",
    )

    # Add r=1 reference line
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Regret Ratio r = avg_regret_final / γ̄")
    ax.set_ylabel("Grid ID")
    ax.set_title(f"Regret Ratio Distribution by Grid (Top {top_n} worst median r)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()


def plot_pass_rate_by_grid(
    df: pd.DataFrame, top_n: int = 20, figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot pass-rate bar chart by grid_id.

    Args:
        df: DataFrame with gamma_pass_overall and grid_id columns
        top_n: Number of grids to show (worst pass rates first)
        figsize: Figure size (auto-computed if None)
    """
    if "gamma_pass_overall" not in df.columns or "grid_id" not in df.columns:
        print("Missing required columns: gamma_pass_overall, grid_id")
        return

    # Compute pass rate per grid
    pass_rates = (
        df.groupby("grid_id")["gamma_pass_overall"].agg(["mean", "count"]).reset_index()
    )
    pass_rates.columns = ["grid_id", "pass_rate", "total_seeds"]

    # Sort by worst pass rate
    pass_rates = pass_rates.sort_values("pass_rate").head(top_n)

    # Set figure size
    if figsize is None:
        figsize = (max(12, top_n * 0.6), 8)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Create bar plot
    bars = ax.barh(
        range(len(pass_rates)),
        pass_rates["pass_rate"],
        color=["red" if x < 1.0 else "green" for x in pass_rates["pass_rate"]],
    )

    # Add labels
    ax.set_yticks(range(len(pass_rates)))
    ax.set_yticklabels(pass_rates["grid_id"], fontsize=8)
    ax.set_xlabel("Pass Rate (gamma_pass_overall)")
    ax.set_title(f"Gamma Pass Rate by Grid (Bottom {top_n})")
    ax.set_xlim(0, 1.05)

    # Add value labels on bars
    for i, (idx, row) in enumerate(pass_rates.iterrows()):
        ax.text(
            row["pass_rate"] + 0.01,
            i,
            f"{row['pass_rate']:.2f} ({row['total_seeds']})",
            va="center",
            fontsize=8,
        )

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_pass_rate_heatmaps(df: pd.DataFrame) -> None:
    """
    Plot pass-rate heatmaps by parameter combinations.

    Args:
        df: DataFrame with gamma metrics and parameters
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    # Heatmap 1: gamma_bar vs gamma_split
    if all(
        col in df.columns
        for col in [
            "gamma_bar_threshold",
            "gamma_split_threshold",
            "gamma_pass_overall",
        ]
    ):
        heatmap_data1 = (
            df.groupby(["gamma_bar_threshold", "gamma_split_threshold"])[
                "gamma_pass_overall"
            ]
            .mean()
            .unstack()
        )
        if not heatmap_data1.empty:
            sns.heatmap(
                heatmap_data1,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                ax=axes[0],
                cbar_kws={"label": "Pass Rate"},
            )
            axes[0].set_title("Pass Rate by (gamma_bar, gamma_split)")
            axes[0].set_xlabel("gamma_split_threshold")
            axes[0].set_ylabel("gamma_bar_threshold")
        else:
            axes[0].text(
                0.5,
                0.5,
                "No data for gamma thresholds",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
    else:
        axes[0].text(
            0.5,
            0.5,
            "Missing gamma threshold columns",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    # Heatmap 2: rho_total vs delete_ratio
    if all(
        col in df.columns for col in ["rho_total", "delete_ratio", "gamma_pass_overall"]
    ):
        heatmap_data2 = (
            df.groupby(["rho_total", "delete_ratio"])["gamma_pass_overall"]
            .mean()
            .unstack()
        )
        if not heatmap_data2.empty:
            sns.heatmap(
                heatmap_data2,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                ax=axes[1],
                cbar_kws={"label": "Pass Rate"},
            )
            axes[1].set_title("Pass Rate by (ρ_total, delete_ratio)")
            axes[1].set_xlabel("delete_ratio")
            axes[1].set_ylabel("rho_total")
        else:
            axes[1].text(
                0.5,
                0.5,
                "No data for rho/delete ratio",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Missing rho_total or delete_ratio columns",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    plt.tight_layout()
    plt.show()


def plot_regret_vs_gamma_scatter(
    df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot scatter of avg_regret_final vs gamma_bar with y=x reference and pass/fail coloring.

    Args:
        df: DataFrame with gamma metrics
        figsize: Figure size
    """
    if not all(
        col in df.columns
        for col in ["avg_regret_final", "gamma_bar_threshold", "gamma_pass_overall"]
    ):
        print(
            "Missing required columns: avg_regret_final, gamma_bar_threshold, gamma_pass_overall"
        )
        return

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Prepare data
    plot_df = df.dropna(
        subset=["avg_regret_final", "gamma_bar_threshold", "gamma_pass_overall"]
    )

    if len(plot_df) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid data for scatter plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Size by delete_ratio if available
    if "delete_ratio" in plot_df.columns:
        sizes = 20 + plot_df["delete_ratio"] * 5  # Base size + scaling
    else:
        sizes = 30

    # Color by pass/fail
    colors = [
        "red" if not passed else "green" for passed in plot_df["gamma_pass_overall"]
    ]

    # Create scatter plot
    scatter = ax.scatter(
        plot_df["gamma_bar_threshold"],
        plot_df["avg_regret_final"],
        c=colors,
        s=sizes,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add y=x reference line
    min_val = min(
        plot_df["gamma_bar_threshold"].min(), plot_df["avg_regret_final"].min()
    )
    max_val = max(
        plot_df["gamma_bar_threshold"].max(), plot_df["avg_regret_final"].max()
    )
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="y = x (perfect adherence)",
    )

    ax.set_xlabel("γ̄ (gamma_bar_threshold)")
    ax.set_ylabel("avg_regret_final")
    ax.set_title("Average Regret vs γ̄ Threshold")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Pass"),
        Patch(facecolor="red", label="Fail"),
        plt.Line2D([0], [0], color="k", linestyle="--", label="y = x"),
    ]
    ax.legend(handles=legend_elements)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_gamma_error_histogram(
    df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot histogram of gamma_error (only violations > 0).

    Args:
        df: DataFrame with gamma_error column
        figsize: Figure size
    """
    if "gamma_error" not in df.columns:
        print("Missing gamma_error column")
        return

    # Filter to violations only
    violations = df[df["gamma_error"] > 0]["gamma_error"].dropna()

    if len(violations) == 0:
        print("No gamma violations found (all errors ≤ 0)")
        return

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    ax.hist(violations, bins=30, alpha=0.7, color="red", edgecolor="black")
    ax.set_xlabel("Gamma Error (relative violation)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Gamma Violations (n={len(violations)})")
    ax.grid(True, alpha=0.3)

    # Add statistics
    ax.text(
        0.7,
        0.9,
        f"Mean: {violations.mean():.3f}\nStd: {violations.std():.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_correlation_panels(
    df: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Plot correlation panels of regret_ratio vs control variables.

    Args:
        df: DataFrame with regret_ratio and control columns
        target_cols: Optional list of columns to correlate with. If None, uses default set.
        figsize: Figure size
    """
    if "regret_ratio" not in df.columns:
        print("Missing regret_ratio column")
        return

    # Default correlation columns
    if target_cols is None:
        target_cols = [
            "PT_error",
            "ST_error",
            "rho_spent_final",
            "target_PT",
            "target_ST",
        ]

    # Filter to available columns
    available_cols = [col for col in target_cols if col in df.columns]

    if not available_cols:
        print("No target correlation columns available")
        return

    n_cols = len(available_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, facecolor="white")
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    for i, col in enumerate(available_cols):
        ax = axes[i]

        # Get valid data
        plot_data = df[[col, "regret_ratio"]].dropna()

        if len(plot_data) < 3:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data for {col}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(col)
            continue

        # Create scatter plot
        ax.scatter(plot_data[col], plot_data["regret_ratio"], alpha=0.6, s=20)

        # Compute correlations
        spearman_corr, spearman_p = stats.spearmanr(
            plot_data[col], plot_data["regret_ratio"]
        )
        pearson_corr, pearson_p = stats.pearsonr(
            plot_data[col], plot_data["regret_ratio"]
        )

        # Add correlation info to title
        title = f"{col}\nSpearman: {spearman_corr:.3f}, Pearson: {pearson_corr:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("regret_ratio")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(available_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def filter_at_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to AT-clean subset.

    Args:
        df: DataFrame with AT-clean flags

    Returns:
        Filtered DataFrame
    """
    df = compute_at_clean_flags(df)
    return df[df["AT_clean_overall"]].copy()
