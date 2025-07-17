import matplotlib.pyplot as plt
import numpy as np

def plot_regret(csv_path, png_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    steps, regret = data[:,0], data[:,1]
    plt.figure()
    plt.loglog(steps, regret, label="regret")
    plt.loglog(steps, np.sqrt(steps), linestyle="--", label="sqrt(T)")
    plt.xlabel("step")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
