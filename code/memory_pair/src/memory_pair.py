import numpy as np
from enum import Enum
from typing import Optional, Union, Tuple, Dict, Any
from .lbfgs import LimitedMemoryBFGS
from .odometer import PrivacyOdometer, RDPOdometer
from .metrics import regret
from .calibrator import Calibrator


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
    ):
        """
        Initialize MemoryPair algorithm.

        Args:
            dim: Dimensionality of the parameter space
            odometer: Privacy odometer for deletion tracking (creates default if None)
            calibrator: Calibrator for bootstrap phase (creates default if None)
            recal_window: Events between recalibration checks (None = disabled)
            recal_threshold: Relative threshold for drift detection
        """
        self.theta = np.zeros(dim)
        self.lbfgs = LimitedMemoryBFGS(dim)
        self.odometer = odometer or RDPOdometer()

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

        # 2. Compute gradient and L-BFGS direction
        g_old = (pred - y) * x
        direction = self.lbfgs.direction(g_old)
        alpha = 1.0
        s = alpha * direction
        theta_new = self.theta + s

        # 3. Update L-BFGS with new information
        g_new = (float(theta_new @ x) - y) * x
        y_vec = g_new - g_old
        self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

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

        # 3. Compute gradients and L-BFGS direction
        g_old = (pred - y) * x
        direction = self.lbfgs.direction(g_old)
        alpha = 1.0
        s = alpha * direction
        theta_new = self.theta + s

        g_new = (float(theta_new @ x) - y) * x
        y_vec = g_new - g_old

        self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

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

        # Compute influence and sensitivity
        g = (float(self.theta @ x) - y) * x
        influence = self.lbfgs.direction(g)
        sensitivity = np.linalg.norm(influence)
        sigma = self.odometer.noise_scale(float(sensitivity))

        # Handle different odometer types
        if isinstance(self.odometer, RDPOdometer):
            # For RDP: spend budget with actual sensitivity and sigma
            self.odometer.spend(sensitivity, sigma)
        else:
            # For legacy PrivacyOdometer: spend without parameters
            self.odometer.spend()

        # Apply noisy deletion
        noise = np.random.normal(0, sigma, self.theta.shape)
        self.theta = self.theta - influence + noise

        # Update counters
        self.events_seen += 1
        self.deletes_seen += 1

    def get_average_regret(self) -> float:
        """Calculates the average regret over all seen events."""
        if self.events_seen == 0:
            return float("inf")
        return self.cumulative_regret / self.events_seen

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
