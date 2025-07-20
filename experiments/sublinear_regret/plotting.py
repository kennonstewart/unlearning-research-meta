import numpy as np
import matplotlib.pyplot as plt
import ast  # For safely evaluating the string representation of lists
from scipy.stats import linregress


def plot_regret(csv_path, png_path):
    # Read the raw data as strings
    data_raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)

    # Convert the array-string columns to actual numbers
    rows, cols = data_raw.shape
    data = np.zeros((rows, cols), dtype=float)

    for i in range(rows):
        for j in range(cols):
            cell = data_raw[i, j]
            if cell.startswith("[") and cell.endswith("]"):
                # This is an array representation - extract the first number
                try:
                    value = ast.literal_eval(cell)[
                        0
                    ]  # Safely evaluate the string as a Python expression
                    data[i, j] = value
                except (ValueError, SyntaxError, IndexError):
                    print(f"Warning: Could not convert {cell} at row {i}, column {j}")
                    data[i, j] = np.nan
            else:
                # Regular number
                data[i, j] = float(cell)

    # Standard cumulative regret curve
    iterations = data[:, 0]
    regret = data[:, 1]  # Assuming column 1 contains regret values

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, regret)
    plt.title("Regret vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Regret")
    plt.grid(True)
    plt.savefig(png_path)
    plt.close()

    # Log-log transformed curve
    # Avoid log(0) by filtering out zero or negative values
    valid = (iterations > 0) & (regret > 0)
    log_iterations = np.log10(iterations[valid])
    log_regret = np.log10(regret[valid])

    plt.figure(figsize=(10, 6))
    plt.plot(log_iterations, log_regret)
    plt.title("Log-Log Regret vs Iterations")
    plt.xlabel("log10(Iterations)")
    plt.ylabel("log10(Cumulative Regret)")
    plt.grid(True)
    loglog_png_path = png_path.replace(".png", "_loglog.png")
    plt.savefig(loglog_png_path)
    plt.close()

    # Locate the knee: where the local slope falls below 0.6
    # Use a sliding window to estimate local slope
    window = 10
    slopes = np.zeros(len(log_iterations) - window)
    for i in range(len(slopes)):
        xw = log_iterations[i : i + window]
        yw = log_regret[i : i + window]
        res = linregress(xw, yw)
        slopes[i] = res.slope

    # Find the first index where slope < 0.6
    knee_idx = np.argmax(slopes < 0.6)
    if slopes[knee_idx] >= 0.6:
        # If no slope < 0.6 found, use halfway point
        knee_idx = len(slopes) // 2

    # Fit regression to asymptotic region (after knee)
    asymptotic_x = log_iterations[knee_idx + window :]
    asymptotic_y = log_regret[knee_idx + window :]
    if len(asymptotic_x) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            asymptotic_x, asymptotic_y
        )
        print(
            f"Asymptotic slope: {slope:.4f} (from log-log region after knee at idx {knee_idx + window})"
        )
        # Optionally, plot regression line on log-log plot
        plt.figure(figsize=(10, 6))
        plt.plot(log_iterations, log_regret, label="log-log curve")
        plt.plot(
            asymptotic_x,
            slope * asymptotic_x + intercept,
            "r--",
            label=f"Fit: slope={slope:.4f}",
        )
        plt.title("Log-Log Regret vs Iterations with Asymptotic Fit")
        plt.xlabel("log10(Iterations)")
        plt.ylabel("log10(Cumulative Regret)")
        plt.grid(True)
        plt.legend()
        loglog_fit_png_path = png_path.replace(".png", "_loglog_fit.png")
        plt.savefig(loglog_fit_png_path)
        plt.close()
    else:
        print("Not enough points in asymptotic region to fit regression.")
