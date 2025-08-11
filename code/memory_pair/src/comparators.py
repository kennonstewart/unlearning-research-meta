# /code/memory_pair/src/comparators.py
"""
Dynamic and static comparators for measuring regret decomposition.

Implements both rolling oracle w_t* (dynamic) and fixed oracle w_0* (static) 
for regret decomposition as per Definition 5.8. Dynamic mode tracks 
path-length P_T for decomposition: O(G²/λ log T + G P_T).
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    from .lbfgs import LimitedMemoryBFGS
except ImportError:
    from lbfgs import LimitedMemoryBFGS


@dataclass
class OracleState:
    """State of the rolling oracle."""

    w_star: np.ndarray
    last_refresh_step: int
    last_objective: float
    window_size: int
    path_length_norm: str = "L2"


class StaticOracle:
    """
    Static oracle that uses a fixed comparator w_0* from calibration phase.
    
    Implements static regret decomposition where the comparator does not change
    over time, providing a baseline for measuring adaptation vs optimization error.
    """
    
    def __init__(
        self,
        dim: int,
        lambda_reg: float = 0.0,
        cfg: Optional[Any] = None,
    ):
        """
        Initialize static oracle.
        
        Args:
            dim: Parameter dimension
            lambda_reg: Regularization parameter
            cfg: Configuration object
        """
        self.dim = dim
        self.lambda_reg = lambda_reg
        self.cfg = cfg
        
        # Static oracle state (set during calibration)
        self.w_star_fixed: Optional[np.ndarray] = None
        self.is_calibrated = False
        
        # Regret tracking against fixed oracle
        self.regret_static = 0.0
        self.events_seen = 0
        
    def calibrate_with_initial_data(self, data_buffer: List[Tuple[np.ndarray, float]]) -> None:
        """
        Calibrate static oracle using initial data from calibration phase.
        
        Args:
            data_buffer: List of (x, y) training examples from calibration
        """
        if len(data_buffer) == 0:
            self.w_star_fixed = np.zeros(self.dim)
            self.is_calibrated = True
            return
            
        # Solve ERM on calibration data to get fixed oracle
        self.w_star_fixed = self._solve_erm_on_data(data_buffer)
        self.is_calibrated = True
        
    def _solve_erm_on_data(self, data: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Solve ERM on given data using simple gradient descent.
        
        Args:
            data: List of (x, y) training examples
            
        Returns:
            Optimized parameters
        """
        if len(data) == 0:
            return np.zeros(self.dim)
            
        w = np.zeros(self.dim)
        
        # Simple gradient descent
        learning_rate = 0.01
        num_steps = 20
        
        for step in range(num_steps):
            grad = np.zeros(self.dim)
            
            # Compute batch gradient
            for x, y in data:
                pred = float(w @ x)
                residual = pred - y
                grad += residual * x
                
            # Average gradient and add regularization
            grad /= len(data)
            grad += self.lambda_reg * w
            
            # Gradient descent step
            w = w - learning_rate * grad
            
        return w
        
    def update_regret_accounting(
        self, x: np.ndarray, y: float, current_theta: np.ndarray
    ) -> None:
        """
        Update static regret accounting against fixed oracle.
        
        Args:
            x: Feature vector
            y: Target value
            current_theta: Current model parameters
        """
        if not self.is_calibrated:
            return
            
        # Compute current loss with regularization
        pred_current = float(current_theta @ x)
        loss_current = self._compute_regularized_loss(pred_current, y, current_theta)
        
        # Compute static oracle loss
        pred_oracle = float(self.w_star_fixed @ x)
        loss_oracle = self._compute_regularized_loss(pred_oracle, y, self.w_star_fixed)
        
        # Update static regret
        regret_increment = loss_current - loss_oracle
        self.regret_static += regret_increment
        self.events_seen += 1
        
    def _compute_regularized_loss(self, pred: float, y: float, w: np.ndarray) -> float:
        """Compute regularized loss for single point."""
        base_loss = 0.5 * (pred - y) ** 2
        reg_term = 0.5 * self.lambda_reg * float(np.dot(w, w))
        return base_loss + reg_term
        
    def get_oracle_metrics(self) -> Dict[str, Any]:
        """Get current static oracle metrics for logging."""
        metrics = {
            "regret_static": self.regret_static,
            "is_calibrated": self.is_calibrated,
            "events_seen": self.events_seen,
            "comparator_type": "static",
        }
        
        if self.w_star_fixed is not None:
            metrics.update({
                "static_oracle_norm": float(np.linalg.norm(self.w_star_fixed)),
            })
            
        return metrics
        
    def get_current_oracle(self) -> Optional[np.ndarray]:
        """Get current (fixed) oracle parameters."""
        if self.is_calibrated and self.w_star_fixed is not None:
            return self.w_star_fixed.copy()
        return None


class RollingOracle:
    """
    Rolling oracle that maintains w_t* by solving ERM on a sliding window.

    Tracks path-length P_T and enables dynamic regret decomposition into
    drift vs optimization components.
    """

    def __init__(
        self,
        dim: int,
        window_W: int = 512,
        oracle_steps: int = 15,
        oracle_stride: Optional[int] = None,
        oracle_tol: float = 1e-6,
        oracle_warmstart: bool = True,
        path_length_norm: str = "L2",
        lambda_reg: float = 0.0,
        cfg: Optional[Any] = None,
    ):
        """
        Initialize rolling oracle.

        Args:
            dim: Parameter dimension
            window_W: Window size for ERM (default 512)
            oracle_steps: Max optimization steps per refresh (default 15)
            oracle_stride: Steps between refreshes (default W//2)
            oracle_tol: Convergence tolerance (default 1e-6)
            oracle_warmstart: Whether to warm-start from previous w_star
            path_length_norm: Norm for path-length ("L2" or "L1")
            lambda_reg: Regularization parameter for ERM
            cfg: Configuration object for additional settings
        """
        self.dim = dim
        self.window_W = window_W
        self.oracle_steps = oracle_steps
        self.oracle_stride = oracle_stride or (window_W // 2)
        self.oracle_tol = oracle_tol
        self.oracle_warmstart = oracle_warmstart
        self.path_length_norm = path_length_norm
        self.lambda_reg = lambda_reg
        self.cfg = cfg

        # Window buffer for insert events (x, y)
        self.window_buffer: List[Tuple[np.ndarray, float]] = []

        # Oracle state
        self.oracle_state: Optional[OracleState] = None
        self.events_seen = 0
        self.last_refresh_event = 0

        # Path-length and regret tracking
        self.P_T_est = 0.0
        self.regret_dynamic = 0.0
        self.regret_static_term = 0.0
        self.regret_path_term = 0.0

        # First window oracle for static term baseline
        self.w_star_first: Optional[np.ndarray] = None

        # Optimizer for oracle
        self.oracle_optimizer = LimitedMemoryBFGS(m_max=5, cfg=cfg)

        # Diagnostics
        self.oracle_refreshes = 0
        self.oracle_stalled_count = 0

        # Drift detection attributes  
        self.drift_threshold = getattr(cfg, "drift_threshold", 0.1) if cfg else 0.1
        self.P_T_history: List[float] = []  # Track P_T history for drift detection
        self.drift_episodes: List[int] = []  # Track when drift was detected
        self.drift_detected = False

    def maybe_update(self, x: np.ndarray, y: float, current_theta: np.ndarray) -> bool:
        """
        Maybe update oracle based on new event.

        Args:
            x: Feature vector of new insert event
            y: Target value
            current_theta: Current model parameters

        Returns:
            True if oracle was refreshed, False otherwise
        """
        # Only track insert events in window
        self.window_buffer.append((x.copy(), y))

        # Maintain window size
        if len(self.window_buffer) > self.window_W:
            self.window_buffer.pop(0)

        self.events_seen += 1

        # Check if it's time to refresh
        events_since_refresh = self.events_seen - self.last_refresh_event
        should_refresh = (
            events_since_refresh >= self.oracle_stride or self.oracle_state is None
        )

        if should_refresh and len(self.window_buffer) >= 2:
            self._refresh_oracle(current_theta)
            self.last_refresh_event = self.events_seen
            return True

        return False

    def _refresh_oracle(self, current_theta: np.ndarray) -> None:
        """Refresh oracle by solving ERM on current window."""
        if len(self.window_buffer) == 0:
            return

        # Initialize oracle parameters
        w_prev = None
        if self.oracle_state is not None and self.oracle_warmstart:
            w_prev = self.oracle_state.w_star.copy()
        elif self.oracle_warmstart:
            w_prev = current_theta.copy()
        else:
            w_prev = np.zeros(self.dim)

        # Run ERM optimization
        w_star_new, objective = self._solve_erm_on_window(w_prev)

        # Update path-length if we have a previous oracle
        if self.oracle_state is not None:
            path_increment = self._compute_path_increment(
                self.oracle_state.w_star, w_star_new
            )
            self.P_T_est += path_increment
            
            # Track P_T history for drift detection
            self.P_T_history.append(self.P_T_est)
            
            # Check for drift (sudden spike in P_T)
            self._check_drift_episode()

        # Store first window oracle for static term baseline
        if self.w_star_first is None:
            self.w_star_first = w_star_new.copy()

        # Update oracle state
        self.oracle_state = OracleState(
            w_star=w_star_new,
            last_refresh_step=self.events_seen,
            last_objective=objective,
            window_size=len(self.window_buffer),
            path_length_norm=self.path_length_norm,
        )

        self.oracle_refreshes += 1

    def _solve_erm_on_window(self, w_init: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve ERM on current window using SGD or L-BFGS.

        Args:
            w_init: Initial parameters for warm-start

        Returns:
            (optimized_w, final_objective)
        """
        if len(self.window_buffer) == 0:
            return w_init.copy(), float("inf")

        w = w_init.copy()

        # Reset optimizer for this ERM solve
        self.oracle_optimizer = LimitedMemoryBFGS(m_max=5, cfg=self.cfg)

        prev_obj = float("inf")
        stalled_count = 0

        for step in range(self.oracle_steps):
            # Compute gradient on full window (batch gradient)
            grad = self._compute_window_gradient(w)

            # Compute current objective
            current_obj = self._compute_window_objective(w)

            # Check convergence
            if abs(prev_obj - current_obj) < self.oracle_tol:
                break

            # Check for stalling
            if current_obj >= prev_obj:
                stalled_count += 1
                if stalled_count >= 3:
                    self.oracle_stalled_count += 1
                    break
            else:
                stalled_count = 0

            prev_obj = current_obj

            # L-BFGS step
            direction = self.oracle_optimizer.direction(grad)

            # Line search with simple backtracking
            step_size = self._line_search(w, direction, grad)

            # Update parameters
            s = step_size * direction
            w_new = w + s

            # Update L-BFGS with curvature pair
            grad_new = self._compute_window_gradient(w_new)
            y_vec = grad_new - grad
            self.oracle_optimizer.add_pair(s, y_vec)

            w = w_new

        final_obj = self._compute_window_objective(w)
        return w, final_obj

    def _compute_window_gradient(self, w: np.ndarray) -> np.ndarray:
        """Compute gradient of regularized loss on window."""
        if len(self.window_buffer) == 0:
            return np.zeros_like(w)

        grad = np.zeros_like(w)

        for x, y in self.window_buffer:
            pred = float(w @ x)
            residual = pred - y
            grad += residual * x

        # Average gradient
        grad /= len(self.window_buffer)

        # Add regularization
        grad += self.lambda_reg * w

        return grad

    def _compute_window_objective(self, w: np.ndarray) -> float:
        """Compute regularized loss objective on window."""
        if len(self.window_buffer) == 0:
            return 0.0

        total_loss = 0.0

        for x, y in self.window_buffer:
            pred = float(w @ x)
            loss = 0.5 * (pred - y) ** 2
            total_loss += loss

        # Average loss
        avg_loss = total_loss / len(self.window_buffer)

        # Add regularization
        reg_term = 0.5 * self.lambda_reg * float(np.dot(w, w))

        return avg_loss + reg_term

    def _line_search(
        self,
        w: np.ndarray,
        direction: np.ndarray,
        grad: np.ndarray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c1: float = 1e-4,
    ) -> float:
        """Simple backtracking line search."""
        alpha = alpha_init
        obj_current = self._compute_window_objective(w)
        grad_norm_sq = float(np.dot(grad, grad))

        # Armijo condition with backtracking
        for _ in range(10):  # Max 10 backtracking steps
            w_new = w + alpha * direction
            obj_new = self._compute_window_objective(w_new)

            armijo_bound = obj_current + c1 * alpha * grad_norm_sq
            if obj_new <= armijo_bound:
                return alpha

            alpha *= rho

        return alpha  # Return last alpha even if Armijo not satisfied

    def _check_drift_episode(self) -> None:
        """Check if a drift episode is occurring based on P_T spike."""
        if len(self.P_T_history) < 2:
            return
            
        # Simple drift detection: significant jump in path length
        current_P_T = self.P_T_history[-1]
        prev_P_T = self.P_T_history[-2] if len(self.P_T_history) >= 2 else 0.0
        
        # Detect spike in path length
        if prev_P_T > 0:
            relative_increase = (current_P_T - prev_P_T) / prev_P_T
            if relative_increase > self.drift_threshold:
                self.drift_detected = True
                self.drift_episodes.append(self.events_seen)
                print(f"[Oracle] Drift detected at event {self.events_seen}, "
                      f"P_T jump: {prev_P_T:.4f} -> {current_P_T:.4f} "
                      f"(+{relative_increase:.1%})")
        else:
            # For very early stages, use absolute threshold
            if current_P_T > self.drift_threshold:
                self.drift_detected = True
                self.drift_episodes.append(self.events_seen)

    def _compute_path_increment(self, w_prev: np.ndarray, w_curr: np.ndarray) -> float:
        """Compute path-length increment between oracle iterates."""
        diff = w_curr - w_prev

        if self.path_length_norm == "L2":
            return float(np.linalg.norm(diff, 2))
        elif self.path_length_norm == "L1":
            return float(np.linalg.norm(diff, 1))
        else:
            # Default to L2
            return float(np.linalg.norm(diff, 2))

    def update_regret_accounting(
        self, x: np.ndarray, y: float, current_theta: np.ndarray
    ) -> None:
        """
        Update dynamic regret accounting after each event.

        Args:
            x: Feature vector
            y: Target value
            current_theta: Current model parameters
        """
        if self.oracle_state is None:
            return

        # Compute current loss with regularization
        pred_current = float(current_theta @ x)
        loss_current = self._compute_regularized_loss(pred_current, y, current_theta)

        # Compute oracle loss with regularization
        pred_oracle = float(self.oracle_state.w_star @ x)
        loss_oracle = self._compute_regularized_loss(
            pred_oracle, y, self.oracle_state.w_star
        )

        # Update dynamic regret
        regret_increment = loss_current - loss_oracle
        self.regret_dynamic += regret_increment

        # Update static term (vs first window oracle)
        if self.w_star_first is not None:
            pred_first = float(self.w_star_first @ x)
            loss_first = self._compute_regularized_loss(
                pred_first, y, self.w_star_first
            )
            static_increment = loss_current - loss_first
            self.regret_static_term += static_increment

        # Path term is difference
        self.regret_path_term = self.regret_dynamic - self.regret_static_term

    def _compute_regularized_loss(self, pred: float, y: float, w: np.ndarray) -> float:
        """Compute regularized loss for single point."""
        base_loss = 0.5 * (pred - y) ** 2
        reg_term = 0.5 * self.lambda_reg * float(np.dot(w, w))
        return base_loss + reg_term

    def get_oracle_metrics(self) -> Dict[str, Any]:
        """Get current oracle metrics for logging."""
        metrics = {
            "P_T_est": self.P_T_est,
            "regret_dynamic": self.regret_dynamic,
            "regret_static_term": self.regret_static_term,
            "regret_path_term": self.regret_path_term,
            "oracle_refreshes": self.oracle_refreshes,
            "oracle_stalled_count": self.oracle_stalled_count,
            "window_size": len(self.window_buffer),
            "comparator_type": "dynamic",  # Rolling oracle is dynamic type
            "drift_detected": self.drift_detected,
            "drift_episodes_count": len(self.drift_episodes),
            "drift_threshold": self.drift_threshold,
        }

        if self.oracle_state is not None:
            metrics.update(
                {
                    "oracle_objective": self.oracle_state.last_objective,
                    "oracle_w_norm": float(np.linalg.norm(self.oracle_state.w_star)),
                    "oracle_refresh_step": self.oracle_state.last_refresh_step,
                }
            )

        return metrics

    def get_current_oracle(self) -> Optional[np.ndarray]:
        """Get current oracle parameters."""
        if self.oracle_state is not None:
            return self.oracle_state.w_star.copy()
        return None

    def is_drift_detected(self) -> bool:
        """Check if drift has been detected recently."""
        return self.drift_detected
        
    def reset_drift_flag(self) -> None:
        """Reset drift detection flag after handling."""
        self.drift_detected = False
