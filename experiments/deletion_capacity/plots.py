import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_capacity_curve(csv_paths: List[str], out_path: str) -> None:
    curves = []
    max_len = 0
    for p in csv_paths:
        df = pd.read_csv(p)
        vals = df['capacity_remaining'].to_numpy()
        curves.append(vals)
        max_len = max(max_len, len(vals))
    data = np.full((len(curves), max_len), np.nan)
    for i, arr in enumerate(curves):
        data[i, : len(arr)] = arr
    avg = np.nanmean(data, axis=0)
    plt.figure()
    plt.plot(avg)
    plt.xlabel('event')
    plt.ylabel('capacity_remaining')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_regret(csv_paths: List[str], out_path: str) -> None:
    curves = []
    max_len = 0
    for p in csv_paths:
        df = pd.read_csv(p)
        vals = df['regret'].to_numpy()
        curves.append(vals)
        max_len = max(max_len, len(vals))
    data = np.full((len(curves), max_len), np.nan)
    for i, arr in enumerate(curves):
        data[i, : len(arr)] = arr
    avg = np.nanmean(data, axis=0)
    plt.figure()
    plt.plot(avg)
    plt.xlabel('event')
    plt.ylabel('cumulative_regret')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
