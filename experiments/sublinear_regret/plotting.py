import numpy as np
import matplotlib.pyplot as plt
import ast  # For safely evaluating the string representation of lists

def plot_regret(csv_path, png_path):
    # Read the raw data as strings
    data_raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
    
    # Convert the array-string columns to actual numbers
    rows, cols = data_raw.shape
    data = np.zeros((rows, cols), dtype=float)
    
    for i in range(rows):
        for j in range(cols):
            cell = data_raw[i, j]
            if cell.startswith('[') and cell.endswith(']'):
                # This is an array representation - extract the first number
                try:
                    value = ast.literal_eval(cell)[0]  # Safely evaluate the string as a Python expression
                    data[i, j] = value
                except (ValueError, SyntaxError, IndexError):
                    print(f"Warning: Could not convert {cell} at row {i}, column {j}")
                    data[i, j] = np.nan
            else:
                # Regular number
                data[i, j] = float(cell)
    
    # Continue with your existing plotting code
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