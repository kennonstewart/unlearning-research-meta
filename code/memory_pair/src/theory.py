"""
Centralized theory functions for memory pair deletion capacity analysis.

Provides unified regret bound computations and sample complexity formulas
that were previously scattered across the codebase. This eliminates duplicate
algebra and serves as a single source of truth for theoretical calculations.
"""
import numpy as np
import math

def N_star(G: float, D: float, c: float, C: float, gamma: float) -> int:
    val = (G * D * (c * C) ** 0.5 / max(gamma, 1e-12)) ** 2
    return int(np.ceil(min(val, 1e6)))

def regret_insert_bound(S_T: float, G: float, D: float, c: float, C: float) -> float:
    # matches the existing bound shape used elsewhere: L·D·√(c C T) with L≈G, T≈S_T/G^2
    if G <= 1e-12:
        return G * D * math.sqrt(c * C * S_T)  # safe fallback
    t_est = max(S_T / (G * G), 1.0)
    return G * D * math.sqrt(c * C * t_est)

def regret_delete_bound(m: int, L: float, lambda_: float, sigma: float, delta_b: float) -> float:
    # High-probability norm bound for Gaussian noise
    noise_norm_bound = sigma * math.sqrt(2 * math.log(1 / max(delta_b, 1e-12)))
    return m * (L / max(lambda_, 1e-12)) * noise_norm_bound