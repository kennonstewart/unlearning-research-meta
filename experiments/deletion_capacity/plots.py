import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from code.memory_pair.src import theory


def plot_capacity_curve(csv_paths: List[str], out_path: str) -> None:
    curves = []
    max_len = 0
    for p in csv_paths:
        df = pd.read_csv(p)
        # Handle both RDP and legacy field names
        if 'capacity_remaining' in df.columns:
            vals = df['capacity_remaining'].to_numpy()
        elif 'eps_remaining' in df.columns:
            vals = df['eps_remaining'].to_numpy()
        else:
            # Fallback: create a dummy curve
            vals = np.ones(len(df))
        curves.append(vals)
        max_len = max(max_len, len(vals))
    
    data = np.full((len(curves), max_len), np.nan)
    for i, arr in enumerate(curves):
        data[i, : len(arr)] = arr
    avg = np.nanmean(data, axis=0)
    
    plt.figure()
    plt.plot(avg)
    plt.xlabel('event')
    plt.ylabel('remaining privacy budget')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_regret(
    csv_paths: List[str],
    out_path: str,
    regret_type: str = "cumulative",
    theory_kwargs: Optional[Dict[str, float]] = None,
) -> None:
    """Plot mean cumulative or average regret across multiple CSVs.

    Args:
        csv_paths: List of CSV file paths containing regret columns.
        out_path: Where to save the resulting figure.
        regret_type: "cumulative" or "average" to select column prefix.
        theory_kwargs: Optional parameters to overlay theoretical bound using
            :mod:`code.memory_pair.src.theory` functions. Expected keys are
            ``G``, ``D``, ``c``, ``C`` and optionally ``S`` (array of S_T).
    """

    prefix = "cum_regret" if regret_type == "cumulative" else "avg_regret"

    emp_curves: List[np.ndarray] = []
    th_curves: List[np.ndarray] = []
    max_len = 0

    for p in csv_paths:
        df = pd.read_csv(p)

        emp_col = next(
            (c for c in df.columns if c.startswith(prefix) and "emp" in c),
            None,
        )
        if emp_col is None:
            continue

        emp_vals = df[emp_col].to_numpy()
        emp_curves.append(emp_vals)
        max_len = max(max_len, len(emp_vals))

        th_col = next(
            (c for c in df.columns if c.startswith(prefix) and "theory" in c),
            None,
        )
        if th_col is not None:
            th_curves.append(df[th_col].to_numpy())

    if not emp_curves:
        print("Warning: No regret columns found for plotting")
        return

    data_emp = np.full((len(emp_curves), max_len), np.nan)
    for i, arr in enumerate(emp_curves):
        data_emp[i, : len(arr)] = arr
    avg_emp = np.nanmean(data_emp, axis=0)

    plt.figure()
    plt.plot(avg_emp, label="empirical")

    if th_curves:
        data_th = np.full((len(th_curves), max_len), np.nan)
        for i, arr in enumerate(th_curves):
            data_th[i, : len(arr)] = arr
        avg_th = np.nanmean(data_th, axis=0)
        plt.plot(avg_th, label="theory")
    elif theory_kwargs is not None:
        t = np.arange(1, max_len + 1)
        G = theory_kwargs.get("G", 1.0)
        D = theory_kwargs.get("D", 1.0)
        c_val = theory_kwargs.get("c", 1.0)
        C_val = theory_kwargs.get("C", 1.0)
        S = theory_kwargs.get("S")
        if S is None:
            S = t * (G ** 2)
        bound = theory.regret_insert_bound(S, G, D, c_val, C_val)
        if regret_type == "average":
            bound = bound / t
        plt.plot(bound, linestyle="--", label="theory")

    plt.xlabel("event")
    plt.ylabel("cumulative regret" if prefix == "cum_regret" else "average regret")
    if th_curves or theory_kwargs is not None:
        plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accountant_comparison(
    data_dir: str,
    out_path: str,
    metric: str = "avg_regret_empirical",
) -> None:
    """Plot comparison of different accountant types.

    Args:
        data_dir: Directory containing CSV files organized by accountant type.
        out_path: Output path for the plot.
        metric: Column name to compare (e.g. "avg_regret_empirical",
            "cum_regret_empirical", "capacity_remaining", "sigma_step_theory").
    """
    accountant_types = ["eps_delta", "zcdp", "relaxed"]
    colors = {"eps_delta": "blue", "zcdp": "red", "relaxed": "green"}
    
    plt.figure(figsize=(12, 8))
    
    for acc_type in accountant_types:
        # Find CSV files for this accountant type
        csv_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv') and acc_type in root:
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print(f"Warning: No CSV files found for accountant type {acc_type}")
            continue
        
        # Aggregate data across all CSV files for this accountant type
        all_curves = []
        max_len = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if metric in df.columns:
                    vals = df[metric].to_numpy()
                else:
                    print(f"Warning: Metric {metric} not found in {csv_file}")
                    continue
                
                all_curves.append(vals)
                max_len = max(max_len, len(vals))
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        if not all_curves:
            continue
        
        # Create matrix and compute statistics
        data_matrix = np.full((len(all_curves), max_len), np.nan)
        for i, curve in enumerate(all_curves):
            data_matrix[i, :len(curve)] = curve
        
        # Compute mean and confidence intervals
        mean_curve = np.nanmean(data_matrix, axis=0)
        std_curve = np.nanstd(data_matrix, axis=0)
        n_curves = np.sum(~np.isnan(data_matrix), axis=0)
        ci_curve = 1.96 * std_curve / np.sqrt(np.maximum(n_curves, 1))
        
        # Plot
        x = np.arange(len(mean_curve))
        plt.plot(x, mean_curve, color=colors.get(acc_type, "black"), 
                label=f"{acc_type} (n={len(all_curves)})", linewidth=2)
        plt.fill_between(x, mean_curve - ci_curve, mean_curve + ci_curve, 
                        color=colors.get(acc_type, "black"), alpha=0.2)
    
    plt.xlabel("Event")
    if metric.startswith("avg_regret"):
        ylabel = "Average Regret"
    elif metric.startswith("cum_regret"):
        ylabel = "Cumulative Regret"
    else:
        ylabel = metric.replace("_", " ").title()
    plt.ylabel(ylabel)
    plt.title(f"Accountant Comparison: {ylabel}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accountant_summary_stats(summary_file: str, out_path: str) -> None:
    """
    Plot summary statistics comparison between accountant types.
    
    Args:
        summary_file: Path to aggregated summary CSV with accountant_type column
        out_path: Output path for the plot
    """
    try:
        df = pd.read_csv(summary_file)
        
        if "accountant_type" not in df.columns:
            print("Warning: No accountant_type column found in summary file")
            return
        
        # Define metrics to compare
        metrics_to_plot = [
            ("final_regret", "Final Regret"),
            ("m_theory", "Deletion Capacity"),
            ("sigma_step_theory", "Noise Scale"),
            ("deletions_count", "Deletions Performed")
        ]
        
        available_metrics = [(col, title) for col, title in metrics_to_plot if col in df.columns]
        
        if not available_metrics:
            print("Warning: No metrics available for plotting")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (metric_col, metric_title) in enumerate(available_metrics):
            if i >= 4:  # Only plot first 4 metrics
                break
                
            ax = axes[i]
            
            # Box plot by accountant type
            accountant_order = ["eps_delta", "zcdp", "relaxed"]
            available_accountants = [acc for acc in accountant_order if acc in df["accountant_type"].values]
            
            if available_accountants:
                sns.boxplot(data=df, x="accountant_type", y=metric_col, 
                           order=available_accountants, ax=ax)
                ax.set_title(metric_title)
                ax.set_xlabel("Accountant Type")
                ax.set_ylabel(metric_title)
                
                # Add mean values as text
                for j, acc_type in enumerate(available_accountants):
                    subset = df[df["accountant_type"] == acc_type]
                    if not subset.empty and metric_col in subset.columns:
                        mean_val = subset[metric_col].mean()
                        if not np.isnan(mean_val):
                            ax.text(j, ax.get_ylim()[1] * 0.9, f"μ={mean_val:.3f}", 
                                   ha='center', va='top', fontsize=10)
        
        # Hide unused subplots
        for j in range(len(available_metrics), 4):
            axes[j].set_visible(False)
        
        plt.suptitle("Accountant Type Comparison - Summary Statistics", fontsize=16)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating summary plot: {e}")


def plot_capacity_vs_noise_tradeoff(summary_file: str, out_path: str) -> None:
    """
    Plot the capacity vs noise tradeoff for different accountant types.
    
    Args:
        summary_file: Path to aggregated summary CSV
        out_path: Output path for the plot
    """
    try:
        df = pd.read_csv(summary_file)
        
        required_cols = ["accountant_type", "m_theory", "sigma_step_theory"]
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Required columns {required_cols} not found in summary file")
            return
        
        # Filter out rows with missing data
        df_clean = df.dropna(subset=required_cols)
        
        if df_clean.empty:
            print("Warning: No complete data available for capacity vs noise plot")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Color map for accountant types
        colors = {"eps_delta": "blue", "zcdp": "red", "relaxed": "green"}
        markers = {"eps_delta": "o", "zcdp": "s", "relaxed": "^"}
        
        for acc_type in df_clean["accountant_type"].unique():
            subset = df_clean[df_clean["accountant_type"] == acc_type]
            
            plt.scatter(subset["sigma_step_theory"], subset["m_theory"], 
                       color=colors.get(acc_type, "black"),
                       marker=markers.get(acc_type, "o"),
                       s=100, alpha=0.7, label=acc_type)
        
        plt.xlabel("Noise Scale (σ)")
        plt.ylabel("Deletion Capacity (m)")
        plt.title("Capacity vs Noise Tradeoff by Accountant Type")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text annotations for interesting points
        for acc_type in df_clean["accountant_type"].unique():
            subset = df_clean[df_clean["accountant_type"] == acc_type]
            if not subset.empty:
                mean_noise = subset["sigma_step_theory"].mean()
                mean_capacity = subset["m_theory"].mean()
                plt.annotate(f"{acc_type}\n(μ_σ={mean_noise:.1f}, μ_m={mean_capacity:.1f})",
                           xy=(mean_noise, mean_capacity), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=colors.get(acc_type, "white"), alpha=0.7),
                           fontsize=9)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating capacity vs noise plot: {e}")


# M10 New Plotting Functions

def plot_nstar_m_live(csv_paths: List[str], out_path: str) -> None:
    """Plot live N* and m capacity estimates over time."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        for p in csv_paths:
            df = pd.read_csv(p)
            
            # Plot N_star_live if available
            if 'N_star_live' in df.columns:
                events = np.arange(len(df))
                ax1.plot(events, df['N_star_live'], alpha=0.7, label=os.path.basename(p))
                
            # Plot m_theory_live if available
            if 'm_theory_live' in df.columns:
                events = np.arange(len(df))
                ax2.plot(events, df['m_theory_live'], alpha=0.7, label=os.path.basename(p))
        
        ax1.set_ylabel('N* (Live)')
        ax1.set_title('Live Sample Complexity Estimate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Event')
        ax2.set_ylabel('m (Live)')
        ax2.set_title('Live Deletion Capacity Estimate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating N*/m live plot: {e}")


def plot_regret_decomp(csv_paths: List[str], out_path: str) -> None:
    """Plot regret decomposition using logged oracle metrics."""
    try:
        static_curves = []
        path_curves = []
        dyn_curves = []
        max_len = 0

        for p in csv_paths:
            df = pd.read_csv(p)
            if all(col in df.columns for col in ["regret_static_term", "regret_path_term", "regret_dynamic"]):
                static_vals = df["regret_static_term"].to_numpy()
                path_vals = df["regret_path_term"].to_numpy()
                dyn_vals = df["regret_dynamic"].to_numpy()

                static_curves.append(static_vals)
                path_curves.append(path_vals)
                dyn_curves.append(dyn_vals)
                max_len = max(max_len, len(dyn_vals))

        if not dyn_curves:
            print("Warning: No regret decomposition columns found.")
            return

        def pad_and_mean(curves):
            data = np.full((len(curves), max_len), np.nan)
            for i, arr in enumerate(curves):
                data[i, : len(arr)] = arr
            return np.nanmean(data, axis=0)

        events = np.arange(max_len)
        plt.figure(figsize=(12, 8))
        plt.plot(events, pad_and_mean(static_curves), label="Static Term", alpha=0.8)
        plt.plot(events, pad_and_mean(path_curves), label="Path Term", alpha=0.8)
        plt.plot(events, pad_and_mean(dyn_curves), label="Dynamic Regret", alpha=0.8, linestyle="--")

        plt.xlabel("Event")
        plt.ylabel("Regret")
        plt.title("Regret Decomposition")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        print(f"Error creating regret decomposition plot: {e}")


def plot_S_T(csv_paths: List[str], out_path: str) -> None:
    """Plot S_T (cumulative gradient squared energy) over time."""
    try:
        plt.figure(figsize=(10, 6))
        
        for p in csv_paths:
            df = pd.read_csv(p)
            
            if 'S_scalar' in df.columns:
                events = np.arange(len(df))
                plt.plot(events, df['S_scalar'], alpha=0.7, label=os.path.basename(p))
                
        plt.xlabel('Event')
        plt.ylabel('S_T (Cumulative ||g_t||²)')
        plt.title('Gradient Squared Energy Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating S_T plot: {e}")


def plot_delete_bins(csv_paths: List[str], out_path: str) -> None:
    """Plot deletion statistics binned by blocked reason."""
    try:
        blocked_reasons = {}
        
        for p in csv_paths:
            df = pd.read_csv(p)
            
            if 'blocked_reason' in df.columns:
                for reason in df['blocked_reason'].dropna().unique():
                    if reason not in blocked_reasons:
                        blocked_reasons[reason] = 0
                    blocked_reasons[reason] += (df['blocked_reason'] == reason).sum()
        
        if blocked_reasons:
            reasons = list(blocked_reasons.keys())
            counts = list(blocked_reasons.values())
            
            plt.figure(figsize=(8, 6))
            plt.bar(reasons, counts, alpha=0.7)
            plt.xlabel('Blocked Reason')
            plt.ylabel('Count')
            plt.title('Deletion Blocking Reasons')
            plt.xticks(rotation=45)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("No blocked_reason data found for plotting")
            
    except Exception as e:
        print(f"Error creating delete bins plot: {e}")


def plot_drift_overlay(csv_paths: List[str], out_path: str) -> None:
    """Plot metrics with segment_id shading to show drift periods."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        regret_label = None

        for p in csv_paths:
            df = pd.read_csv(p)
            events = np.arange(len(df))

            regret_col = next(
                (c for c in df.columns if c.startswith("cum_regret")),
                None,
            )
            if regret_col is None:
                regret_col = next(
                    (c for c in df.columns if c.startswith("avg_regret")),
                    None,
                )
            if regret_col is not None:
                ax1.plot(events, df[regret_col], alpha=0.7, label=os.path.basename(p))
                if regret_label is None:
                    regret_label = (
                        "cumulative regret"
                        if regret_col.startswith("cum_regret")
                        else "average regret"
                    )

            # Plot accuracy or other metric
            metric_col = 'acc' if 'acc' in df.columns else 'abs_error' if 'abs_error' in df.columns else None
            if metric_col:
                ax2.plot(events, df[metric_col], alpha=0.7, label=os.path.basename(p))
            
            # Add segment shading
            if 'segment_id' in df.columns:
                segments = df['segment_id'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
                
                for i, seg_id in enumerate(segments):
                    seg_mask = df['segment_id'] == seg_id
                    seg_events = events[seg_mask]
                    if len(seg_events) > 0:
                        ax1.axvspan(seg_events.min(), seg_events.max(), alpha=0.2, color=colors[i])
                        ax2.axvspan(seg_events.min(), seg_events.max(), alpha=0.2, color=colors[i])
        
        if regret_label is None:
            regret_label = "regret"
        ax1.set_ylabel(regret_label.title())
        ax1.set_title(f"{regret_label.title()} with Drift Segments")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        ax2.set_xlabel('Event')
        ax2.set_ylabel(metric_col or 'Metric')
        ax2.set_title(f'{metric_col or "Metric"} with Drift Segments')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating drift overlay plot: {e}")
