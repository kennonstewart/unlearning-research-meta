import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from memory_pair.src.metrics import loss_half_mse

# Backwards compatibility alias
regret = loss_half_mse


def abs_error(pred: float, y: float) -> float:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return abs(pred_scalar - y_scalar)


def mape(pred: float, y: float, eps: float = 1e-8) -> float:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return abs(pred_scalar - y_scalar) / (abs(y_scalar) + eps)


def smape(pred: float, y: float, eps: float = 1e-8) -> float:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return 2 * abs(pred_scalar - y_scalar) / (abs(pred_scalar) + abs(y_scalar) + eps)


# If you still want a 0–1 “accuracy” notion:
def tol_accuracy(pred: float, y: float, tau: float = 0.1) -> int:
    # Convert to scalar values to handle numpy arrays/tensors
    pred_scalar = float(pred.item() if hasattr(pred, 'item') else pred)
    y_scalar = float(y.item() if hasattr(y, 'item') else y)
    return int(abs(pred_scalar - y_scalar) <= tau)


# M10 Evaluation Suite Metrics

def compute_regret_decomposition(
    cumulative_regret: float,
    G_hat: float,
    D_hat: float,
    c_hat: float,
    C_hat: float,
    S_T: float,
    P_T: float,
    T: int,
    lambda_est: float
) -> Dict[str, float]:
    """
    Decompose dynamic regret into static, adaptive, and path components.
    
    Returns:
        Dictionary with regret components
    """
    # Static regret component
    if lambda_est > 0:
        R_static = (G_hat ** 2) / (lambda_est * c_hat) * (1 + np.log(T))
    else:
        R_static = 0.0
        
    # Adaptive regret component  
    R_adaptive = G_hat * D_hat * np.sqrt(c_hat * C_hat * S_T)
    
    # Path regret component
    R_path = G_hat * P_T
    
    return {
        "R_static": R_static,
        "R_adaptive": R_adaptive,
        "R_path": R_path,
        "R_theory_total": R_static + R_adaptive + R_path,
        "R_empirical": cumulative_regret
    }


def estimate_S_T_slope(S_T_values: List[float], window_K: int = 100) -> float:
    """
    Estimate the slope of S_T over the last K steps using linear regression.
    
    Args:
        S_T_values: List of cumulative S_T values
        window_K: Window size for slope estimation
        
    Returns:
        Estimated slope (gradient squared energy rate)
    """
    if len(S_T_values) < 2:
        return 0.0
        
    # Use last K values
    recent_values = S_T_values[-window_K:] if len(S_T_values) >= window_K else S_T_values
    
    if len(recent_values) < 2:
        return 0.0
        
    # Simple linear regression
    n = len(recent_values)
    x = np.arange(n)
    y = np.array(recent_values)
    
    if n == 1:
        return 0.0
        
    slope = np.polyfit(x, y, 1)[0]
    return max(slope, 0.0)  # S_T should be non-decreasing


def sensitivity_bin_aggregation(
    sensitivities: List[float],
    metrics: List[Dict[str, Any]],
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Aggregate metrics by sensitivity bins for analysis.
    
    Args:
        sensitivities: List of sensitivity values
        metrics: List of metric dictionaries
        n_bins: Number of bins for aggregation
        
    Returns:
        Dictionary with binned aggregations
    """
    if len(sensitivities) != len(metrics):
        raise ValueError("Sensitivities and metrics must have same length")
        
    if len(sensitivities) == 0:
        return {}
        
    # Create bins
    sens_array = np.array(sensitivities)
    bin_edges = np.linspace(sens_array.min(), sens_array.max(), n_bins + 1)
    bin_indices = np.digitize(sens_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Aggregate by bins
    result = {"bin_edges": bin_edges.tolist()}
    
    # Find all metric keys
    all_keys = set()
    for m in metrics:
        all_keys.update(m.keys())
    
    # Aggregate each metric
    for key in all_keys:
        bin_values = [[] for _ in range(n_bins)]
        
        for i, metric_dict in enumerate(metrics):
            if key in metric_dict:
                bin_idx = bin_indices[i]
                bin_values[bin_idx].append(metric_dict[key])
        
        # Compute bin means
        bin_means = []
        for bin_vals in bin_values:
            if bin_vals:
                bin_means.append(np.mean(bin_vals))
            else:
                bin_means.append(0.0)
                
        result[f"{key}_binned"] = bin_means
    
    return result


def track_live_capacity(
    events_processed: int,
    S_T: float,
    odometer_state: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Optional[float]]:
    """
    Track live N* and m capacity estimates.
    
    Args:
        events_processed: Number of events processed so far
        S_T: Cumulative gradient squared energy
        odometer_state: State dictionary from odometer
        config: Configuration dictionary
        
    Returns:
        Dictionary with live capacity estimates
    """
    result = {
        "N_star_live": None,
        "m_theory_live": None
    }
    
    # Extract required parameters
    G_hat = odometer_state.get("G_hat")
    D_hat = odometer_state.get("D_hat") 
    c_hat = odometer_state.get("c_hat")
    C_hat = odometer_state.get("C_hat")
    gamma_ins = config.get("gamma_insert")
    gamma_del = config.get("gamma_delete")
    sigma_step = odometer_state.get("sigma_step", 1.0)
    delta_B = config.get("delta_b", 0.05)
    
    # Compute N_star_live if parameters available
    if all(v is not None for v in [G_hat, D_hat, c_hat, C_hat, gamma_ins]):
        try:
            from odometer import N_star_live
            result["N_star_live"] = N_star_live(S_T, G_hat, D_hat, c_hat, C_hat, gamma_ins)
        except ImportError:
            # Fallback implementation
            coeff = D_hat * np.sqrt(c_hat * C_hat) / max(gamma_ins, 1e-12)
            avg_sq = S_T / max(events_processed, 1.0) if events_processed > 0 else S_T
            result["N_star_live"] = int(np.ceil(coeff ** 2 * avg_sq))
    
    # Compute m_theory_live if parameters available  
    if all(v is not None for v in [G_hat, D_hat, c_hat, C_hat, gamma_del]):
        try:
            from odometer import m_theory_live
            result["m_theory_live"] = m_theory_live(
                S_T, events_processed, G_hat, D_hat, c_hat, C_hat, 
                gamma_del, sigma_step, delta_B
            )
        except ImportError:
            # Fallback implementation
            insertion_regret = D_hat * np.sqrt(c_hat * C_hat * S_T)
            coeff = (G_hat * D_hat / max(sigma_step, 1e-12)) * np.sqrt(2 * np.log(1 / max(delta_B, 1e-12)))
            remaining = gamma_del * events_processed - insertion_regret
            if remaining <= 0:
                result["m_theory_live"] = 0
            else:
                m = int(np.floor(remaining / max(coeff, 1e-12)))
                result["m_theory_live"] = max(m, 0)
    
    return result
