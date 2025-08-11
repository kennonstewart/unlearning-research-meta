# /code/memory_pair/src/memory_pair.py

import numpy as np
from enum import Enum
from typing import Optional, Union, Tuple, Dict, Any

try:
    from .odometer import PrivacyOdometer, RDPOdometer
    from .lbfgs import LimitedMemoryBFGS
    from .metrics import regret
    from .calibrator import Calibrator
    from .comparators import RollingOracle
    from .accountant_strategies import StrategyAccountantAdapter
except ModuleNotFoundError:
    from odometer import PrivacyOdometer, RDPOdometer
    from lbfgs import LimitedMemoryBFGS
    from metrics import regret
    from calibrator import Calibrator
    from comparators import RollingOracle
    from accountant_strategies import StrategyAccountantAdapter


class Phase(Enum):
    """
    Enumeration of the three phases in the MemoryPair state machine.

    CALIBRATION: Bootstrap phase to estimate constants G, D, c, C
    LEARNING: Insert-only phase until ready_to_predict (inserts >= N*)
    INTERLEAVING: Normal operation with both inserts and deletes allowed
    """

    CALIBRATION = 1
    LEARNING = 2
    INTERLEAVING = 3


class MemoryPair:
    """
    Online learning algorithm with unlearning capabilities and privacy accounting.

    MemoryPair implements a three-phase state machine for calibrated online learning:

    1. CALIBRATION: Bootstrap phase to estimate theoretical constants (G, D, c, C)
       and compute sample complexity N*
    2. LEARNING: Insert-only phase until the model is ready to make predictions
       (when inserts_seen >= N*)
    3. INTERLEAVING: Normal operation allowing both insertions and deletions
       with privacy accounting

    The algorithm uses L-BFGS for second-order optimization and maintains cumulative
    regret tracking. Deletions are performed with differential privacy guarantees
    managed by the PrivacyOdometer.

    Attributes:
        theta (np.ndarray): Current parameter vector
        lbfgs (LimitedMemoryBFGS): L-BFGS optimizer instance
        odometer (PrivacyOdometer): Privacy budget tracking for deletions
        phase (Phase): Current phase of the state machine
        calibrator (Calibrator): Helper for collecting calibration statistics
        cumulative_regret (float): Total regret accumulated over all events
        events_seen (int): Total number of events (inserts + deletes) processed
        inserts_seen (int): Number of insertions processed
        deletes_seen (int): Number of deletions processed
        N_star (Optional[int]): Sample complexity threshold for ready_to_predict
        ready_to_predict (bool): Whether model is ready for predictions
        last_grad (Optional[np.ndarray]): Last computed gradient (for external access)
    """

    def __init__(
        self,
        dim: int,
        odometer: Union[PrivacyOdometer, RDPOdometer] = None,
        calibrator: Optional[Calibrator] = None,
        recal_window: Optional[int] = None,
        recal_threshold: float = 0.3,
        cfg: Optional[Any] = None,
    ):
        """
        Initialize MemoryPair algorithm.

        Args:
            dim: Dimensionality of the parameter space
            odometer: Privacy odometer for deletion tracking (creates default if None)
            calibrator: Calibrator for bootstrap phase (creates default if None)
            recal_window: Events between recalibration checks (None = disabled)
            recal_threshold: Relative threshold for drift detection
            cfg: Configuration object with feature flags (for future use)
        """
        self.theta = np.zeros(dim)
        m_max = getattr(cfg, "m_max", 10) if cfg else 10
        self.lbfgs = LimitedMemoryBFGS(m_max=m_max, cfg=cfg)
        self.odometer = odometer or RDPOdometer()

        # Store config for feature flags (no behavior change yet)
        self.cfg = cfg

        # State machine attributes
        self.phase = Phase.CALIBRATION
        self.calibrator = calibrator or Calibrator()  # Use provided calibrator

        # Tracking attributes
        self.cumulative_regret = 0.0
        self.events_seen = 0
        self.inserts_seen = 0
        self.deletes_seen = 0
        self.N_star: Optional[int] = None
        self.ready_to_predict = False
        self.calibration_stats: Optional[dict] = None

        # Adaptive recalibration attributes
        self.recal_window = recal_window
        self.recal_threshold = recal_threshold
        self.last_recal_event = 0
        self.recalibrations_count = 0

        # For external gradient access
        self.last_grad: Optional[np.ndarray] = None

        # Strong convexity tracking for new implementation
        self.lambda_raw: Optional[float] = None
        self.sc_stable: int = 0
        self.pair_admitted: bool = True
        self.pair_damped: bool = False
        self.d_norm: float = 0.0

        # Adaptive geometry tracking
        # S_scalar accumulates squared gradients from **insert** events only.
        # Deletes are tracked separately in S_delete for diagnostics but do not
        # influence the step-size policy.
        self.S_scalar: float = 0.0
        self.S_delete: float = 0.0
        self.t: int = 0  # counts insert steps feeding S_scalar
        self.lambda_est: Optional[float] = None
        self.eta_t: float = 0.0
        self.lambda_stability_counter: int = 0
        self.sc_active: bool = False
        self.lambda_estimator = LambdaEstimator(
            ema_beta=getattr(cfg, "ema_beta", 0.9) if cfg is not None else 0.9,
            floor=getattr(cfg, "lambda_floor", 1e-6) if cfg is not None else 1e-6,
            cap=getattr(cfg, "lambda_cap", 1e3) if cfg is not None else 1e3,
        )

        # Oracle for dynamic regret tracking (optional)
        self.oracle: Optional[Union[RollingOracle, "StaticOracle"]] = None
        if cfg and getattr(cfg, "enable_oracle", False):
            comparator_type = getattr(cfg, "comparator_type", "dynamic")
            
            if comparator_type == "static":
                # Import here to avoid circular imports
                from comparators import StaticOracle
                lambda_reg = getattr(cfg, "lambda_reg", 0.0)
                self.oracle = StaticOracle(dim=dim, lambda_reg=lambda_reg, cfg=cfg)
            else:
                # Dynamic (rolling) oracle
                oracle_window_W = getattr(cfg, "oracle_window_W", 512)
                oracle_steps = getattr(cfg, "oracle_steps", 15)
                oracle_stride = getattr(cfg, "oracle_stride", None)
                oracle_tol = getattr(cfg, "oracle_tol", 1e-6)
                oracle_warmstart = getattr(cfg, "oracle_warmstart", True)
                path_length_norm = getattr(cfg, "path_length_norm", "L2")
                lambda_reg = getattr(cfg, "lambda_reg", 0.0)

                self.oracle = RollingOracle(
                    dim=dim,
                    window_W=oracle_window_W,
                    oracle_steps=oracle_steps,
                    oracle_stride=oracle_stride,
                    oracle_tol=oracle_tol,
                    oracle_warmstart=oracle_warmstart,
                    path_length_norm=path_length_norm,
                    lambda_reg=lambda_reg,
                    cfg=cfg,
                )
                
        # Drift-responsive rate adaptation
        self.drift_adaptation_enabled = getattr(cfg, "drift_adaptation", False) if cfg else False
        self.drift_rate_boost = getattr(cfg, "drift_rate_boost", 1.5) if cfg else 1.5

    def _compute_regularized_loss(self, pred: float, y: float) -> float:
        """Compute regularized loss: l(pred, y) + (lambda_reg/2) * ||theta||^2"""
        base_loss = 0.5 * (pred - y) ** 2  # squared loss
        lambda_reg = getattr(self.cfg, "lambda_reg", 0.0) if self.cfg else 0.0
        reg_term = 0.5 * lambda_reg * float(np.dot(self.theta, self.theta))
        return base_loss + reg_term

    def _compute_regularized_gradient(
        self, x: np.ndarray, pred: float, y: float
    ) -> np.ndarray:
        """Compute regularized gradient: grad_l + lambda_reg * theta"""
        base_grad = (pred - y) * x  # gradient of squared loss
        lambda_reg = getattr(self.cfg, "lambda_reg", 0.0) if self.cfg else 0.0
        reg_grad = lambda_reg * self.theta

        # Add bounds to prevent extreme gradients
        total_grad = base_grad + reg_grad
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 100.0:  # Clamp very large gradients
            total_grad = total_grad * (100.0 / grad_norm)

        return total_grad

    def _update_lambda_estimate(
        self,
        g_old: np.ndarray,
        g_new: np.ndarray,
        theta_old: np.ndarray,
        theta_new: np.ndarray,
    ) -> None:
        """Update online lambda estimate using secant method + EMA"""
        # Compute raw secant estimate
        diff_w = theta_new - theta_old
        diff_g = g_new - g_old

        denom = float(np.dot(diff_w, diff_w))
        if denom <= 1e-12:
            self.lambda_raw = None
            return

        num = float(np.dot(diff_g, diff_w))
        lambda_raw = max(num / denom, 0.0)

        # Apply bounds
        if self.cfg:
            bounds = getattr(self.cfg, "lambda_est_bounds", [1e-8, 1e6])
            lambda_raw = float(np.clip(lambda_raw, bounds[0], bounds[1]))

        self.lambda_raw = lambda_raw

        # EMA smoothing
        if self.cfg:
            beta = getattr(self.cfg, "lambda_est_beta", 0.1)
        else:
            beta = 0.1

        if self.lambda_est is None:
            self.lambda_est = lambda_raw
        else:
            self.lambda_est = (1 - beta) * self.lambda_est + beta * lambda_raw

        # Update stability counter
        threshold = (
            getattr(self.cfg, "lambda_min_threshold", 1e-6) if self.cfg else 1e-6
        )
        K = getattr(self.cfg, "lambda_stability_K", 100) if self.cfg else 100

        if self.lambda_est > threshold:
            self.sc_stable += 1
        else:
            self.sc_stable = 0

    def calibrate_step(self, x: np.ndarray, y: float) -> float:
        """
        Perform one calibration step during the bootstrap phase.

        This method runs a standard insert-like update but logs the gradient
        and parameter values to the Calibrator instead of updating regret gates.
        Should only be called during the CALIBRATION phase.

        Args:
            x: Input feature vector
            y: Target value

        Returns:
            Prediction made before the parameter update

        Raises:
            RuntimeError: If called outside CALIBRATION phase
        """
        if self.phase != Phase.CALIBRATION:
            raise RuntimeError(
                f"calibrate_step() can only be called during CALIBRATION phase, current phase: {self.phase}"
            )

        # 1. Prediction before update
        pred = float(self.theta @ x)

        # 2. Compute regularized gradient and track statistics
        g_old = self._compute_regularized_gradient(x, pred, y)
        self.S_scalar += float(np.dot(g_old, g_old))
        self.t += 1

        # 3. Step-size policy
        self._update_step_size()

        # 4. Compute L-BFGS direction with step-size
        direction = self.lbfgs.direction(g_old, calibrator=self.calibrator)
        self.d_norm = float(np.linalg.norm(direction))

        # Apply trust region clipping if needed
        d_max = getattr(self.cfg, "d_max", float("inf")) if self.cfg else float("inf")
        if self.d_norm > d_max:
            direction = direction * (d_max / self.d_norm)
            self.d_norm = d_max

        s = self.eta_t * direction
        theta_prev = self.theta
        theta_new = theta_prev + s

        # 5. Update L-BFGS with new information (using regularized gradients)
        pred_new = float(theta_new @ x)
        g_new = self._compute_regularized_gradient(x, pred_new, y)
        y_vec = g_new - g_old

        # Track pair admission (will be implemented in lbfgs.py)
        self.pair_admitted, self.pair_damped = self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

        # Add parameter bounds to prevent extreme values
        theta_norm = np.linalg.norm(self.theta)
        if theta_norm > 10.0:  # Reasonable bound for parameters
            self.theta = self.theta * (10.0 / theta_norm)

        # 6. Update lambda estimator with new implementation
        self._update_lambda_estimate(g_old, g_new, theta_prev, theta_new)

        # 4. Log to calibrator (key difference from insert)
        self.calibrator.observe(g_old, self.theta)

        # 5. Update counters
        self.events_seen += 1
        self.inserts_seen += 1
        self.last_grad = g_old

        return pred

    def finalize_calibration(self, gamma: float) -> None:
        """
        Finalize the calibration phase and transition to LEARNING.

        Computes the sample complexity N* from collected statistics but
        does NOT finalize the odometer. The odometer should be finalized
        separately after the warmup phase completes.

        Args:
            gamma: Target average regret per step for theoretical bounds (gamma_learn)

        Raises:
            RuntimeError: If called outside CALIBRATION phase
        """
        if self.phase != Phase.CALIBRATION:
            raise RuntimeError(
                f"finalize_calibration() can only be called during CALIBRATION phase, current phase: {self.phase}"
            )

        # Get calibration statistics
        stats = self.calibrator.finalize(gamma, self)
        self.N_star = stats["N_star"]

        # Store stats for later access
        self.calibration_stats = stats

        # Calibrate static oracle if enabled
        if self.oracle is not None and hasattr(self.oracle, 'calibrate_with_initial_data'):
            # For static oracle, use calibration data collected during bootstrap
            # In a real implementation, we'd collect the calibration data
            # For now, we'll mark it as calibrated (the calibrator should provide this data)
            if hasattr(self.calibrator, 'get_calibration_data'):
                calibration_data = self.calibrator.get_calibration_data()
                self.oracle.calibrate_with_initial_data(calibration_data)
            else:
                # Fallback: mark as calibrated with empty data
                self.oracle.calibrate_with_initial_data([])

        # DO NOT finalize odometer here - it should be done after warmup

        # Transition to LEARNING phase
        self.phase = Phase.LEARNING

        print(
            f"[MemoryPair] Calibration complete. N* = {self.N_star}, transitioning to LEARNING phase."
        )
        print("[MemoryPair] Odometer will be finalized after warmup completes.")

    @property
    def can_predict(self) -> bool:
        """
        Check if the model is ready to make reliable predictions.

        Returns True when the model has seen enough data (inserts_seen >= N*)
        and is past the calibration phase.

        Returns:
            True if ready to predict, False otherwise
        """
        return self.ready_to_predict

    def insert(
        self,
        x: np.ndarray,
        y: float,
        *,
        return_grad: bool = False,
        log_to_odometer: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Insert a data point and update the model.

        Performs a standard online learning update during LEARNING or INTERLEAVING
        phases. Updates cumulative regret, parameter vector via L-BFGS, and handles
        state transitions. Returns the prediction made before the update.

        Args:
            x: Input feature vector
            y: Target value
            return_grad: If True, return (prediction, gradient) tuple
            log_to_odometer: If True, log gradient/theta to odometer (legacy parameter)

        Returns:
            Prediction before update, or (prediction, gradient) if return_grad=True

        Raises:
            RuntimeError: If called during CALIBRATION phase
        """
        if self.phase == Phase.CALIBRATION:
            raise RuntimeError(
                "Use calibrate_step() during CALIBRATION phase, not insert()"
            )

        # 1. Prediction before update
        pred = float(self.theta @ x)

        # 2. Update regret counters
        self.cumulative_regret += regret(pred, y)
        self.events_seen += 1
        self.inserts_seen += 1

        # 3. Compute regularized gradient and track S_T
        g_old = self._compute_regularized_gradient(x, pred, y)
        self.S_scalar += float(np.dot(g_old, g_old))
        self.t += 1

        # 4. Step-size policy
        self._update_step_size()

        # 5. Compute L-BFGS direction with step-size
        direction = self.lbfgs.direction(g_old, calibrator=self.calibrator)
        self.d_norm = float(np.linalg.norm(direction))

        # Apply trust region clipping if needed
        d_max = getattr(self.cfg, "d_max", float("inf")) if self.cfg else float("inf")
        if self.d_norm > d_max:
            direction = direction * (d_max / self.d_norm)
            self.d_norm = d_max

        s = self.eta_t * direction
        theta_prev = self.theta
        theta_new = theta_prev + s

        # 6. Update L-BFGS with new information (using regularized gradients)
        pred_new = float(theta_new @ x)
        g_new = self._compute_regularized_gradient(x, pred_new, y)
        y_vec = g_new - g_old

        # Track pair admission (will be implemented in lbfgs.py)
        self.pair_admitted, self.pair_damped = self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

        # Add parameter bounds to prevent extreme values
        theta_norm = np.linalg.norm(self.theta)
        if theta_norm > 10.0:  # Reasonable bound for parameters
            self.theta = self.theta * (10.0 / theta_norm)

        # 7. Update lambda estimator with new implementation
        self._update_lambda_estimate(g_old, g_new, theta_prev, theta_new)

        # Store gradient for external access
        self.last_grad = g_old

        # Update EMA tracking for drift detection (if past calibration phase)
        if self.phase in [Phase.LEARNING, Phase.INTERLEAVING]:
            self.calibrator.observe_ongoing(g_old)

            # Check for recalibration trigger
            self._check_recalibration_trigger()

        # Optional: push stats to odometer during bootstrap/warmup (legacy)
        if log_to_odometer and hasattr(self.odometer, "observe"):
            self.odometer.observe(g_old, self.theta)

        # Update oracle if enabled (only for insert events)
        if self.oracle is not None and self.phase in [
            Phase.LEARNING,
            Phase.INTERLEAVING,
        ]:
            # Handle dynamic oracle (RollingOracle)
            if hasattr(self.oracle, 'maybe_update'):
                oracle_refreshed = self.oracle.maybe_update(x, y, self.theta)
                if oracle_refreshed:
                    print(
                        f"[Oracle] Refreshed at event {self.events_seen}, P_T_est = {self.oracle.P_T_est:.4f}"
                    )
            
            # Update regret accounting for both static and dynamic oracles
            self.oracle.update_regret_accounting(x, y, self.theta)

        # 4. Handle state transitions
        if (
            self.phase == Phase.LEARNING
            and self.N_star is not None
            and self.inserts_seen >= self.N_star
        ):
            self.ready_to_predict = True
            self.phase = Phase.INTERLEAVING
            print(
                f"[MemoryPair] Reached N* = {self.N_star} inserts. Ready to predict, transitioning to INTERLEAVING phase."
            )

        if return_grad:
            return pred, g_old
        return pred

    def delete(self, x: np.ndarray, y: float) -> None:
        """
        Delete a data point using differentially private unlearning.

        Computes the influence of the data point, checks privacy budget capacity,
        and applies the deletion update with appropriate noise injection for
        differential privacy guarantees.

        Args:
            x: Input feature vector of point to delete
            y: Target value of point to delete

        Raises:
            RuntimeError: If odometer is not ready or capacity is exceeded
        """
        if self.phase != Phase.INTERLEAVING:
            raise RuntimeError("Deletions are only allowed during INTERLEAVING phase")

        if not self.odometer.ready_to_delete:
            raise RuntimeError("Odometer not finalized or capacity depleted.")

        # Compute regularized gradient and track diagnostics (no S_scalar update)
        pred = float(self.theta @ x)
        g = self._compute_regularized_gradient(x, pred, y)
        self.S_delete += float(np.dot(g, g))

        # Step-size policy (no lambda update during deletes)
        tiny = 1e-12
        self._update_step_size()

        influence = self.lbfgs.direction(g, calibrator=self.calibrator)
        sensitivity = np.linalg.norm(influence)
        sigma = self.odometer.noise_scale(float(sensitivity))

        # Debug: Validate parameters before calling spend
        if sensitivity is None or np.isnan(sensitivity) or np.isinf(sensitivity):
            raise RuntimeError(f"Invalid sensitivity value: {sensitivity}")
        if sigma is None or np.isnan(sigma) or np.isinf(sigma):
            raise RuntimeError(f"Invalid sigma value: {sigma}")

        # Handle different odometer types
        if (isinstance(self.odometer, RDPOdometer) or
            isinstance(self.odometer, StrategyAccountantAdapter)):
            # For RDP and new accountant strategies: spend budget with actual sensitivity and sigma
            try:
                self.odometer.spend(float(sensitivity), float(sigma))
            except Exception as e:
                raise RuntimeError(f"Odometer spend failed with sensitivity={sensitivity}, sigma={sigma}: {e}")
        else:
            # For legacy PrivacyOdometer: spend without parameters
            self.odometer.spend()

        # Apply noisy deletion with step-size
        noise = np.random.normal(0, sigma, self.theta.shape)
        self.theta = self.theta - self.eta_t * influence + noise

        # Update counters
        self.events_seen += 1
        self.deletes_seen += 1

    def _update_step_size(self) -> None:
        """Compute step size based on AdaGrad or strong-convexity schedule."""
        tiny = 1e-12
        eps = getattr(self.cfg, "adagrad_eps", 1e-12) if self.cfg else 1e-12
        D_bound = getattr(self.calibrator, "D_hat_t", None)
        if D_bound is None:
            D_bound = getattr(self.cfg, "D_bound", 1.0) if self.cfg else 1.0
        eta_max = getattr(self.cfg, "eta_max", 1.0) if self.cfg else 1.0

        if self._lambda_is_stable():
            # Strong convexity step size: 1/(lambda * t) but with safeguards
            lambda_safe = max(self.lambda_est, 1e-6)  # Prevent division by tiny numbers
            t_safe = max(self.t, 1)
            self.eta_t = 1.0 / (lambda_safe * t_safe)
            self.sc_active = True
        else:
            self.eta_t = D_bound / np.sqrt(self.S_scalar + eps)
            self.sc_active = False

        self.eta_t = min(self.eta_t, eta_max)
        
        # Apply drift-responsive rate adaptation
        if (self.drift_adaptation_enabled and 
            self.oracle is not None and 
            hasattr(self.oracle, 'is_drift_detected') and
            self.oracle.is_drift_detected()):
            # Boost learning rate when drift is detected
            self.eta_t *= self.drift_rate_boost
            self.eta_t = min(self.eta_t, eta_max)  # Still respect max limit
            # Reset drift flag after adaptation
            self.oracle.reset_drift_flag()

    def _lambda_is_stable(self) -> bool:
        """Check whether strong-convexity estimate is reliable."""
        if not self.cfg or not getattr(self.cfg, "strong_convexity", False):
            return False

        threshold = getattr(self.cfg, "lambda_min_threshold", 1e-6)
        K = getattr(self.cfg, "lambda_stability_K", 100)

        return (
            self.lambda_est is not None
            and self.lambda_est > threshold
            and self.sc_stable >= K
        )

    def get_average_regret(self) -> float:
        """Calculates the average regret over all seen events."""
        if self.events_seen == 0:
            return float("inf")
        return self.cumulative_regret / self.events_seen

    def get_current_loss_reg(self, x: np.ndarray, y: float) -> float:
        """Get the current regularized loss value for logging."""
        pred = float(self.theta @ x)
        return self._compute_regularized_loss(pred, y)

    def get_metrics_dict(self) -> dict:
        """Get dictionary of current metrics for logging."""
        metrics = {
            "lambda_est": self.lambda_est,
            "lambda_raw": self.lambda_raw,
            "sc_stable": self.sc_stable,
            "pair_admitted": self.pair_admitted,
            "pair_damped": self.pair_damped,
            "d_norm": self.d_norm,
            "eta_t": self.eta_t,
            "sc_active": self.sc_active,
        }

        # Add oracle metrics if oracle is enabled
        if self.oracle is not None:
            oracle_metrics = self.oracle.get_oracle_metrics()
            metrics.update(oracle_metrics)

        return metrics

    def _check_recalibration_trigger(self) -> None:
        """Check if recalibration should be triggered based on drift detection."""
        if (
            self.recal_window is None
            or self.phase != Phase.INTERLEAVING
            or not hasattr(self.odometer, "supports_recalibration")
            or not self.odometer.supports_recalibration()
        ):
            return

        # Check if it's time for a recalibration check
        events_since_last_recal = self.events_seen - self.last_recal_event
        if events_since_last_recal < self.recal_window:
            return

        # Check for drift
        if self.calibrator.check_drift(self.recal_threshold):
            print(
                f"[MemoryPair] Drift detected at event {self.events_seen}. Triggering recalibration."
            )
            self._perform_recalibration()

        self.last_recal_event = self.events_seen

    def _perform_recalibration(self) -> None:
        """Perform adaptive recalibration with updated statistics."""
        try:
            # Get updated statistics from calibrator
            new_stats = self.calibrator.get_updated_stats(self)

            # Estimate remaining events
            remaining_T = max(1000, self.events_seen)  # Conservative estimate

            # Recalibrate the odometer
            self.odometer.recalibrate_with(new_stats, remaining_T)

            self.recalibrations_count += 1
            print(f"[MemoryPair] Recalibration #{self.recalibrations_count} completed.")

        except Exception as e:
            print(f"[MemoryPair] Recalibration failed: {e}")
            # Continue without recalibration on failure

    def get_recalibration_stats(self) -> Dict[str, Any]:
        """Get statistics about recalibration events."""
        return {
            "recalibrations_count": self.recalibrations_count,
            "last_recal_event": self.last_recal_event,
            "current_G_ema": getattr(self.calibrator, "G_ema", None),
            "finalized_G": getattr(self.calibrator, "finalized_G", None),
        }


class LambdaEstimator:
    def __init__(self, ema_beta: float = 0.9, floor: float = 1e-6, cap: float = 1e3):
        self.beta = ema_beta
        self.floor = floor
        self.cap = cap
        self.ema: Optional[float] = None

    def update(
        self,
        g_prev: np.ndarray,
        g_curr: np.ndarray,
        w_prev: np.ndarray,
        w_curr: np.ndarray,
    ) -> Optional[float]:
        diff_w = w_curr - w_prev
        denom = float(np.dot(diff_w, diff_w))
        if denom <= 1e-12:
            return self.ema
        num = float(np.dot(g_curr - g_prev, diff_w))
        lam = max(num / denom, 0.0)
        if self.ema is None:
            self.ema = lam
        else:
            self.ema = self.beta * self.ema + (1 - self.beta) * lam
        self.ema = float(np.clip(self.ema, self.floor, self.cap))
        return self.ema
