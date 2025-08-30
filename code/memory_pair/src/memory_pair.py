# /code/memory_pair/src/memory_pair.py

import numpy as np
from enum import Enum
from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from .odometer import N_star_live, m_theory_live
    from .lbfgs import LimitedMemoryBFGS
    from .metrics import loss_half_mse
    from .calibrator import Calibrator
    from .comparators import RollingOracle
    from .accountant import get_adapter
    from .accountant.types import Accountant
except (ModuleNotFoundError, ImportError):
    from odometer import N_star_live, m_theory_live
    from lbfgs import LimitedMemoryBFGS
    from metrics import loss_half_mse
    from calibrator import Calibrator
    from comparators import RollingOracle
    from accountant import get_adapter
    from accountant.types import Accountant


@dataclass
class CalibStats:
    G: float
    D: float
    c: float
    C: float
    N_star: int


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
    Online learning algorithm with unlearning capabilities and zCDP privacy accounting.

    MemoryPair implements a three-phase state machine for calibrated online learning:

    1. CALIBRATION: Bootstrap phase to estimate theoretical constants (G, D, c, C)
       and compute sample complexity N*
    2. LEARNING: Insert-only phase until the model is ready to make predictions
       (when inserts_seen >= N*)
    3. INTERLEAVING: Normal operation allowing both insertions and deletions
       with privacy accounting

    The algorithm uses L-BFGS for second-order optimization and maintains cumulative
    regret tracking. When an oracle/comparator is enabled, regret is computed against
    the oracle's optimal solution. When no oracle is available, regret is computed
    against a zero prediction baseline to enable basic regret tracking. Deletions 
    are performed with differential privacy guarantees managed by the zCDP-only 
    Accountant interface.

    Attributes:
        theta (np.ndarray): Current parameter vector
        lbfgs (LimitedMemoryBFGS): L-BFGS optimizer instance
        accountant (Accountant): zCDP privacy accounting interface for deletions
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
        accountant: Optional[Accountant] = None,
        calibrator: Optional[Calibrator] = None,
        recal_window: Optional[int] = None,
        recal_threshold: float = 0.3,
        cfg: Optional[Any] = None,
    ):
        """
        Initialize MemoryPair algorithm.

        Args:
            dim: Dimensionality of the parameter space
            accountant: zCDP accountant adapter for unified privacy accounting
            calibrator: Calibrator for bootstrap phase (creates default if None)
            recal_window: Events between recalibration checks (None = disabled)
            recal_threshold: Relative threshold for drift detection
            cfg: Configuration object with feature flags (for future use)
        """
        self.theta = np.zeros(dim)
        m_max = getattr(cfg, "m_max", 10) if cfg else 10
        self.lbfgs = LimitedMemoryBFGS(m_max=m_max, cfg=cfg)

        # zCDP-only accountant
        if accountant is not None:
            self.accountant = accountant
        else:
            acct_kwargs = {
                "rho_total": getattr(cfg, "rho_total", 1.0),
                "delta_total": getattr(cfg, "delta_total", 1e-5),
                "T": getattr(cfg, "T", getattr(cfg, "max_events", 10000) if cfg else 10000),
                "gamma": getattr(cfg, "gamma_delete", getattr(cfg, "gamma", 0.5) if cfg else 0.5),
                "lambda_": getattr(cfg, "lambda_", 0.1),
                "delta_b": getattr(cfg, "delta_b", 0.05),
                "m_max": getattr(cfg, "m_max", None),
            }
            self.accountant = get_adapter("zcdp", **acct_kwargs)

        self.cfg = cfg
        self.lambda_reg = getattr(cfg, "lambda_reg", 0.0) if cfg else 0.0

        # State machine attributes
        self.phase = Phase.CALIBRATION
        self.calibrator = calibrator or Calibrator()  # Use provided calibrator

        # Tracking attributes
        self.cumulative_regret = 0.0
        # Last-step regret increments (from comparator accounting)
        self.regret_increment = 0.0
        self.static_regret_increment = 0.0
        self.path_regret_increment = 0.0
        # Noise-induced regret tracking
        self.noise_regret_cum = 0.0
        self.noise_regret_inc = 0.0

        self.events_seen = 0
        self.inserts_seen = 0
        self.deletes_seen = 0
        self.N_star: Optional[int] = None
        self.N_gamma: Optional[int] = None
        self.ready_to_predict = False
        self.calibration_stats: Optional[dict] = None

        # Frozen snapshot of calibrator stats after calibration
        self.calib_stats: Optional[CalibStats] = None

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
            comparator_type = getattr(cfg, "comparator", "dynamic")

            if comparator_type == "static":
                # Import here to avoid circular imports
                from .comparators import StaticOracle

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
        self.drift_adaptation_enabled = (
            getattr(cfg, "drift_adaptation", False) if cfg else False
        )
        self.drift_kappa = (
            getattr(cfg, "drift_kappa", 0.5) if cfg else 0.5
        )  # (1 + kappa) factor
        self.drift_window = (
            getattr(cfg, "drift_window", 10) if cfg else 10
        )  # Duration in steps
        self.drift_boost_remaining = 0  # Steps remaining for current boost
        self.base_eta_t = 0.0  # Store base learning rate for boost calculation

    def _compute_regularized_loss(self, pred: float, y: float) -> float:
        """Compute regularized loss: l(pred, y) + (lambda_reg/2) * ||theta||^2"""
        base_loss = loss_half_mse(pred, y)  # squared loss
        reg_term = 0.5 * self.lambda_reg * float(np.dot(self.theta, self.theta))
        return base_loss + reg_term

    def _compute_regularized_gradient(
        self, x: np.ndarray, pred: float, y: float
    ) -> np.ndarray:
        """Compute regularized gradient: grad_l + lambda_reg * theta"""
        base_grad = (pred - y) * x  # gradient of squared loss
        reg_grad = self.lambda_reg * self.theta

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

        # Store frozen snapshot of calibrator stats
        self.calib_stats = CalibStats(
            G=stats["G"],
            D=stats["D"],
            c=stats["c"],
            C=stats["C"],
            N_star=stats["N_star"],
        )

        # Calibrate static oracle if enabled
        if self.oracle is not None and hasattr(
            self.oracle, "calibrate_with_initial_data"
        ):
            # For static oracle, use calibration data collected during bootstrap
            # In a real implementation, we'd collect the calibration data
            # For now, we'll mark it as calibrated (the calibrator should provide this data)
            if hasattr(self.calibrator, "get_calibration_data"):
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

        # 2. Update counters (do NOT add loss to regret; regret is comparator-based)
        base_loss_t = self._compute_regularized_loss(pred, y)
        self.events_seen += 1
        self.inserts_seen += 1

        # Reset per-event regret increment until comparator updates it
        self.regret_increment = float("nan")
        self.static_regret_increment = 0.0
        self.path_regret_increment = 0.0

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

        # Update oracle if enabled (only for insert events)
        if self.oracle is not None and self.phase in [
            Phase.LEARNING,
            Phase.INTERLEAVING,
        ]:
            # Handle dynamic oracle (RollingOracle)
            if hasattr(self.oracle, "maybe_update"):
                oracle_refreshed = self.oracle.maybe_update(x, y, self.theta)

            # Update regret accounting for both static and dynamic oracles
            incs = self.oracle.update_regret_accounting(x, y, self.theta)
            if isinstance(incs, dict):
                self.regret_increment = incs.get("regret_increment", 0.0)
                self.static_regret_increment = incs.get("static_increment", 0.0)
                self.path_regret_increment = incs.get("path_increment", 0.0)
        else:
            # Fallback regret tracking when oracle is disabled
            # Compute simple regret against zero prediction for basic tracking
            pred = float(self.theta @ x)
            zero_pred_loss = loss_half_mse(0.0, y)
            current_loss = loss_half_mse(pred, y)
            self.regret_increment = current_loss - zero_pred_loss

        # 7b. Accumulate comparator-based regret if present; otherwise do not change cumulative_regret
        if self.regret_increment is not None:
            try:
                # add only if it is a finite number
                if not (
                    isinstance(self.regret_increment, float)
                    and np.isnan(self.regret_increment)
                ):
                    self.cumulative_regret += float(self.regret_increment)
            except Exception:
                # If comparator unavailable, leave cumulative_regret unchanged for this event
                pass

        # 4. Handle state transitions
        if (
            self.phase == Phase.LEARNING
            and self.N_star is not None
            and self.inserts_seen >= self.N_star
            and self.N_gamma is None
        ):
            self.phase = Phase.INTERLEAVING
            print(
                f"[MemoryPair] Reached N* = {self.N_star} inserts. Transitioning to INTERLEAVING phase."
            )
            print("[Finalize] Finalizing accountant...")

            # Finalize accountant (zCDP-only)
            if self.accountant is not None:
                self.accountant.finalize(
                    {
                        "G": self.calib_stats.G,
                        "D": self.calib_stats.D,
                        "c": self.calib_stats.c,
                        "C": self.calib_stats.C,
                    },
                    T_estimate=self.calib_stats.N_star or self.events_seen or 1,
                )

            # Compute deferred inference threshold N_gamma
            try:
                from .theory import N_gamma
            except (ModuleNotFoundError, ImportError):
                from theory import N_gamma

            m_cap = 0
            if self.accountant is not None:
                m_cap = self.accountant.metrics().get("m_capacity", 0)
            gamma = getattr(self.cfg, "gamma", 0.5)
            self.N_gamma = N_gamma(
                self.calib_stats.G,
                self.calib_stats.D,
                self.calib_stats.c,
                self.calib_stats.C,
                m_cap,
                gamma,
            )
            print(
                f"[MemoryPair] N_gamma = {self.N_gamma} events required before predictions."
            )
            self._maybe_enable_predictions()

        # Check if predictions can be enabled
        self._maybe_enable_predictions()

        if return_grad:
            return pred, g_old
        return pred

    def delete(self, x: np.ndarray, y: float) -> Optional[str]:
        """
        Delete a data point using differentially private unlearning with gating.

        Computes the influence of the data point, checks both regret and privacy
        gates, and applies the deletion update with appropriate noise injection
        if both gates pass.

        Args:
            x: Input feature vector of point to delete
            y: Target value of point to delete

        Returns:
            None if deletion succeeded, or blocked reason string if gates failed

        Raises:
            RuntimeError: If accountant is not ready or invalid parameters
        """
        if self.phase != Phase.INTERLEAVING:
            raise RuntimeError("Deletions are only allowed during INTERLEAVING phase")

        # Always use accountant interface (no more odometer branching)
        return self._delete_with_accountant(x, y)

    def _delete_with_accountant(self, x: np.ndarray, y: float) -> Optional[str]:
        """Delete using the new accountant interface."""
        if not self.accountant.ready():
            return "privacy_gate"

        # Compute influence for gating checks
        pred = float(self.theta @ x)
        g = self._compute_regularized_gradient(x, pred, y)

        # Get step size for deletion cost estimation
        self._update_step_size()
        influence = self.lbfgs.direction(g, calibrator=self.calibrator)
        sensitivity = np.linalg.norm(influence)

        # Check privacy gate via accountant
        ok, sigma, reason = self.accountant.pre_delete(sensitivity)
        if not ok:
            return reason

        # Check regret gate using new theory functions
        if hasattr(self.cfg, "gamma_delete") and self.cfg.gamma_delete is not None:
            try:
                from .theory import regret_insert_bound, regret_delete_bound
            except (ModuleNotFoundError, ImportError):
                from theory import regret_insert_bound, regret_delete_bound

            L = self.calib_stats.G
            ins_reg = regret_insert_bound(
                self.S_scalar,
                self.calib_stats.G,
                self.calib_stats.D,
                self.calib_stats.c,
                self.calib_stats.C,
            )

            # Projected avg regret with one more delete (m_used + 1)
            m_used = self.accountant.metrics().get("m_used", 0)
            del_reg = regret_delete_bound(
                m_used + 1,
                L,
                self.lambda_reg or 1e-12,
                sigma,
                getattr(self.cfg, "delta_b", 0.05),
            )
            proj_avg = (ins_reg + del_reg) / max(self.events_seen or 1, 1)
            if proj_avg > getattr(self.cfg, "gamma_delete", float("inf")):
                return "regret_gate"

        # Both gates passed - proceed with deletion

        # Spend budget
        self.accountant.spend(sensitivity, sigma)

        # Track diagnostics (no S_scalar update for deletes)
        self.S_delete += float(np.dot(g, g))

        # Apply noisy deletion with step-size
        noise = np.random.normal(0, sigma, self.theta.shape)
        self.theta = self.theta - self.eta_t * influence + noise

        # Track noise-induced regret bound
        delta_b = getattr(self.cfg, "delta_b", 0.05)
        lambda_safe = max(self.lambda_reg, 1e-12)
        delta_reg = (
            (self.calib_stats.G / lambda_safe)
            * sigma
            * np.sqrt(2 * np.log(1 / max(delta_b, 1e-12)))
        )
        self.noise_regret_inc = float(delta_reg)
        self.noise_regret_cum += float(delta_reg)

        # Update counters
        self.events_seen += 1
        self.deletes_seen += 1

        # Check if predictions can be enabled after deletion
        self._maybe_enable_predictions()

        return None  # Success

    def _update_step_size(self) -> None:
        """Compute step size based on AdaGrad or strong-convexity schedule."""
        tiny = 1e-12
        eps = getattr(self.cfg, "adagrad_eps", 1e-12) if self.cfg else 1e-12
        D_bound = getattr(self.calibrator, "D_hat_t", None)
        if D_bound is None:
            D_bound = getattr(self.cfg, "D_bound", 1.0) if self.cfg else 1.0
        eta_max = getattr(self.cfg, "eta_max", 1.0) if self.cfg else 1.0

        # Base step size: strongly convex η_t = 1/(λ * t) or AdaGrad
        if self._lambda_is_stable():
            # Strong convexity step size: 1/(lambda * t) but with safeguards
            lambda_safe = max(self.lambda_est, 1e-6)  # Prevent division by tiny numbers
            t_safe = max(self.t, 1)
            self.base_eta_t = 1.0 / (lambda_safe * t_safe)
            self.sc_active = True
        else:
            self.base_eta_t = D_bound / np.sqrt(self.S_scalar + eps)
            self.sc_active = False

        self.base_eta_t = min(self.base_eta_t, eta_max)

        # Check for drift detection and apply LR nudge
        if (
            self.drift_adaptation_enabled
            and self.oracle is not None
            and hasattr(self.oracle, "is_drift_detected")
            and self.oracle.is_drift_detected()
        ):
            # Start new drift boost: temporarily multiply η_t by (1 + κ) for K steps
            self.drift_boost_remaining = self.drift_window
            self.oracle.reset_drift_flag()
            print(
                f"[MemoryPair] Drift detected at t={self.t}, applying LR boost: "
                f"η_t *= (1 + {self.drift_kappa}) for {self.drift_window} steps"
            )

        # Apply drift boost if active
        if self.drift_boost_remaining > 0:
            self.eta_t = self.base_eta_t * (1.0 + self.drift_kappa)
            self.drift_boost_remaining -= 1
        else:
            self.eta_t = self.base_eta_t

        # Final cap to stability
        self.eta_t = min(self.eta_t, eta_max)

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

    def _maybe_enable_predictions(self) -> None:
        """Enable predictions once theoretical sample complexity is met."""
        if (
            not self.ready_to_predict
            and self.N_gamma is not None
            and self.events_seen >= self.N_gamma
        ):
            self.ready_to_predict = True
            print(
                f"[MemoryPair] Reached N_gamma = {self.N_gamma} events. Predictions enabled."
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

    def get_stepsize_policy(self) -> Dict[str, Any]:
        """Get step-size policy information for logging."""
        if hasattr(self, 'sc_active') and self.sc_active:
            policy = "strongly-convex"
            params = {
                "lambda": self.lambda_est,
                "t": self.t,
                "eta_formula": "1/(λ*t)"
            }
        else:
            policy = "adagrad"
            D_bound = getattr(self.calibrator, "D_hat_t", None)
            if D_bound is None:
                D_bound = getattr(self.cfg, "D_bound", 1.0) if self.cfg else 1.0
            params = {
                "D": D_bound,
                "S_t": self.S_scalar,
                "eta_formula": "D/√S_t"
            }
        
        return {
            "stepsize_policy": policy,
            "stepsize_params": params
        }

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
            # Drift-responsive fields
            "drift_boost_remaining": getattr(self, "drift_boost_remaining", 0),
            "base_eta_t": getattr(self, "base_eta_t", self.eta_t),
        }

        # Regret tracking fields
        metrics.update(
            {
                "regret_increment": self.regret_increment,
                "static_regret_increment": self.static_regret_increment,
                "path_regret_increment": self.path_regret_increment,
                "cum_regret": self.cumulative_regret,
                "avg_regret": self.get_average_regret(),
                "noise_regret_increment": self.noise_regret_inc,
                "noise_regret_cum": self.noise_regret_cum,
                "cum_regret_with_noise": self.cumulative_regret + self.noise_regret_cum,
                "avg_regret_with_noise": (
                    self.cumulative_regret + self.noise_regret_cum
                )
                / max(self.events_seen, 1),
                "N_gamma": self.N_gamma,
            }
        )

        # Add accountant metrics if accountant is available
        if self.accountant is not None:
            acc_metrics = self.accountant.metrics()
            metrics.update(
                {
                    "accountant": acc_metrics.get("accountant"),
                    "m_capacity": acc_metrics.get("m_capacity"),
                    "m_used": acc_metrics.get("m_used"),
                    "sigma_step": acc_metrics.get("sigma_step"),
                    "eps_spent": acc_metrics.get("eps_spent"),
                    "eps_remaining": acc_metrics.get("eps_remaining"),
                    "rho_spent": acc_metrics.get("rho_spent"),
                    "rho_remaining": acc_metrics.get("rho_remaining"),
                    "delta_total": acc_metrics.get("delta_total"),
                }
            )
        else:
            # Default values when accountant is not available
            metrics.update(
                {
                    "accountant": None,
                    "m_capacity": None,
                    "m_used": None,
                    "sigma_step": None,
                    "eps_spent": None,
                    "eps_remaining": None,
                    "rho_spent": None,
                    "rho_remaining": None,
                    "delta_total": None,
                }
            )

        # Add oracle metrics if oracle is enabled
        if self.oracle is not None:
            try:
                oracle_metrics = self.oracle.get_oracle_metrics()
            except Exception:
                # If oracle isn't ready (e.g., during calibration/warmup), provide safe defaults
                oracle_metrics = {
                    "P_T": 0.0,
                    "P_T_est": 0.0,
                    "drift_flag": False,
                    "regret_dynamic": 0.0,
                    "regret_static_term": 0.0,
                    "regret_path_term": 0.0,
                }

            # Map oracle metrics to expected field names
            oracle_metrics["P_T"] = oracle_metrics.get(
                "P_T", oracle_metrics.get("P_T_est", 0.0)
            )
            oracle_metrics["P_T_est"] = oracle_metrics.get(
                "P_T_est", oracle_metrics["P_T"]
            )
            oracle_metrics["drift_flag"] = oracle_metrics.get("drift_detected", False)

            # Normalize regret field names
            if (
                "regret_static" in oracle_metrics
                and "regret_static_term" not in oracle_metrics
            ):
                oracle_metrics["regret_static_term"] = oracle_metrics.pop(
                    "regret_static"
                )
            oracle_metrics.setdefault(
                "regret_path_term", oracle_metrics.get("regret_path", 0.0)
            )
            oracle_metrics.setdefault(
                "regret_dynamic",
                oracle_metrics.get("regret_static_term", 0.0)
                + oracle_metrics.get("regret_path_term", 0.0),
            )

            if hasattr(self.oracle, "__class__"):
                oracle_metrics["comparator_type"] = (
                    "static"
                    if "Static" in self.oracle.__class__.__name__
                    else "dynamic"
                )

            metrics.update(oracle_metrics)
        else:
            # Default values when oracle is disabled
            metrics.update(
                {
                    "P_T": 0.0,
                    "regret_dynamic": 0.0,
                    "regret_static_term": 0.0,
                    "regret_path_term": 0.0,
                    "drift_flag": False,
                    "comparator_type": "none",
                }
            )

        # Add step-size policy information
        stepsize_info = self.get_stepsize_policy()
        metrics.update(stepsize_info)

        return metrics

    def get_live_diagnostics(self) -> Dict[str, Any]:
        """Get live M9 diagnostics for deletion gating."""
        diagnostics = {}

        # Get gamma values from config
        if hasattr(self, "cfg") and self.cfg is not None:
            diagnostics["gamma_bar"] = getattr(self.cfg, "gamma_bar", None)
            diagnostics["gamma_split"] = getattr(self.cfg, "gamma_split", None)
            diagnostics["gamma_ins"] = getattr(self.cfg, "gamma_insert", None)
            diagnostics["gamma_del"] = getattr(self.cfg, "gamma_delete", None)

        # Get live capacity estimates if odometer is available
        if hasattr(self.odometer, "G_hat") and hasattr(self.odometer, "D_hat"):
            G_hat = getattr(self.odometer, "G_hat", None)
            D_hat = getattr(self.odometer, "D_hat", None)
            c_hat = getattr(self.odometer, "c_hat", None)
            C_hat = getattr(self.odometer, "C_hat", None)
            gamma_ins = diagnostics.get("gamma_ins", None)
            gamma_del = diagnostics.get("gamma_del", None)

            if all(v is not None for v in [G_hat, D_hat, c_hat, C_hat, gamma_ins]):
                diagnostics["N_star_live"] = N_star_live(
                    self.S_scalar, G_hat, D_hat, c_hat, C_hat, gamma_ins
                )

            if all(v is not None for v in [G_hat, D_hat, c_hat, C_hat, gamma_del]):
                # Get noise scale estimate
                sigma_step = getattr(self.odometer, "sigma_step", 1.0)
                delta_B = (
                    getattr(self.cfg, "delta_b", 0.05) if hasattr(self, "cfg") else 0.05
                )

                diagnostics["m_theory_live"] = m_theory_live(
                    self.S_scalar,
                    self.inserts_seen,
                    G_hat,
                    D_hat,
                    c_hat,
                    C_hat,
                    gamma_del,
                    sigma_step,
                    delta_B,
                )

        return diagnostics

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
