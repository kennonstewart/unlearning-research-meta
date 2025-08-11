# /code/memory_pair/src/odometer.py
# Privacy odometer to track budget consumption for deletions.
# This module implements a privacy odometer to manage privacy budgets for unlearning operations.
import numpy as np
from typing import Dict, Any, Optional, Tuple


def N_star_live(S_T, G_hat, D_hat, c_hat, C_hat, gamma_insert) -> int:
    """Live sample complexity using cumulative squared gradients."""
    tiny = 1e-12
    if D_hat is None or c_hat is None or C_hat is None or gamma_insert is None:
        return 0
    coeff = D_hat * np.sqrt(c_hat * C_hat) / max(gamma_insert, tiny)
    # Estimate average gradient squared using S_T and G_hat bound
    if G_hat is None or abs(G_hat) <= tiny:
        avg_sq = S_T
    else:
        t_est = S_T / (G_hat ** 2)
        avg_sq = S_T / max(t_est, 1.0)
    return int(np.ceil(coeff ** 2 * avg_sq))


def m_theory_live(
    S_T,
    N,
    G_hat,
    D_hat,
    c_hat,
    C_hat,
    gamma_delete,
    sigma_step,
    delta_B: float = 0.05,
) -> int:
    """Theoretical deletion capacity with live gradient statistics."""
    tiny = 1e-12
    insertion_regret = D_hat * np.sqrt(c_hat * C_hat * S_T)
    coeff = (G_hat * D_hat / max(sigma_step, tiny)) * np.sqrt(2 * np.log(1 / max(delta_B, tiny)))
    remaining = gamma_delete * N - insertion_regret
    if remaining <= 0:
        return 0
    m = int(np.floor(remaining / max(coeff, tiny)))
    return max(m, 0)


class PrivacyOdometer:
    """
    Adaptive Privacy Odometer for differentially private machine unlearning.

    This class manages privacy budget allocation for deletion operations by:
    1. Collecting statistics during a warmup/calibration phase
    2. Computing optimal deletion capacity based on regret constraints
    3. Tracking privacy budget consumption during deletions
    4. Providing noise scales for differential privacy guarantees

    The odometer solves an optimization problem to maximize deletion capacity m
    subject to a total regret constraint, then allocates privacy budget uniformly
    across all deletions (ε_step = ε_total/m, δ_step = δ_total/m).

    Attributes:
        eps_total (float): Total privacy budget (ε)
        delta_total (float): Total failure probability (δ)
        T (int): Estimated total number of events
        gamma (float): Target average regret per event
        lambda_ (float): Strong convexity parameter
        delta_b (float): Regret bound failure probability
        deletion_capacity (Optional[int]): Maximum number of deletions allowed
        L (Optional[float]): Lipschitz constant (gradient bound)
        D (Optional[float]): Hypothesis diameter
        eps_step (Optional[float]): Privacy budget per deletion
        delta_step (Optional[float]): Failure probability per deletion
        sigma_step (Optional[float]): Noise standard deviation per deletion
        eps_spent (float): Privacy budget consumed so far
        deletions_count (int): Number of deletions performed
        ready_to_delete (bool): Whether odometer is finalized and ready
    """

    def __init__(
        self,
        *,
        eps_total: float = 1.0,
        delta_total: float = 1e-5,
        T: int = 10000,
        gamma: float = 0.5,
        lambda_: float = 0.1,
        delta_b: float = 0.05,
    ):
        """
        Initialize PrivacyOdometer with experiment parameters.

        Args:
            eps_total: Total privacy budget (ε)
            delta_total: Total failure probability (δ)
            T: Estimated total number of events
            gamma: Target average regret per event
            lambda_: Strong convexity parameter
            delta_b: Regret bound failure probability
        """
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta_b = delta_b

        # Statistics collected during warmup (for backward compatibility)
        self._grad_norms = []
        self._theta_traj = []

        # Computed after finalization
        self.deletion_capacity: int = 0
        self.L: Optional[float] = None
        self.D: Optional[float] = None
        self.eps_step: Optional[float] = None
        self.delta_step: Optional[float] = None
        self.sigma_step: Optional[float] = None

        # Budget tracking
        self.eps_spent = 0.0
        self.deletions_count = 0
        self.ready_to_delete = False
        self._status = "unfinalized"  # Track odometer status

    def observe(self, grad: np.ndarray, theta: np.ndarray) -> None:
        """
        Record gradient and parameter during warmup phase (legacy method).

        This method is kept for backward compatibility with existing code
        that calls odometer.observe() directly.

        Args:
            grad: Gradient vector
            theta: Parameter vector
        """
        self._grad_norms.append(np.linalg.norm(grad))
        self._theta_traj.append(np.copy(theta))

    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """
        Finalize odometer configuration using calibration statistics.

        This is the primary finalization method that uses statistics from
        the Calibrator to compute deletion capacity and privacy parameters.

        Args:
            stats: Dictionary containing calibration results with keys:
                  'G': Gradient bound, 'D': Hypothesis diameter,
                  'c': Lower curvature bound, 'C': Upper curvature bound
            T_estimate: Estimated total number of events (typically N*)
        """
        self.L = stats["G"]  # Use gradient bound as Lipschitz constant
        self.D = stats["D"]
        self.c = stats.get("c", 1.0)
        self.C = stats.get("C", 1.0)
        self.T = T_estimate

        # Solve for optimal deletion capacity
        m = self._solve_capacity()
        self.deletion_capacity = max(1, m)

        # Compute per-deletion privacy parameters
        self.eps_step = self.eps_total / self.deletion_capacity
        self.delta_step = self.delta_total / self.deletion_capacity
        self.sigma_step = (
            (self.L / self.lambda_)
            * np.sqrt(2 * np.log(1.25 / self.delta_step))
            / self.eps_step
        )

        # Mark as ready for deletions
        self.ready_to_delete = True

        print(
            f"[Odometer] Finalized with deletion capacity m = {self.deletion_capacity}"
        )
        print(f"[Odometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(
            f"[Odometer] ε_step = {self.eps_step:.6f}, δ_step = {self.delta_step:.2e}, σ = {self.sigma_step:.4f}"
        )

    def finalize(self) -> None:
        """
        Legacy finalization method using collected warmup statistics.

        This method maintains backward compatibility by using statistics
        collected via observe() calls. For new code, prefer finalize_with().
        """
        if not self._grad_norms or not self._theta_traj:
            print("[Odometer] Warning: No observations collected, using default values")
            self.L = 1.0
            self.D = 1.0
        else:
            self.L = max(self._grad_norms)
            theta_0 = self._theta_traj[0]
            self.D = max(np.linalg.norm(theta - theta_0) for theta in self._theta_traj)

        m = self._solve_capacity()
        self.deletion_capacity = max(1, m)
        self.eps_step = self.eps_total / self.deletion_capacity
        self.delta_step = self.delta_total / self.deletion_capacity
        self.sigma_step = (
            (self.L / self.lambda_)
            * np.sqrt(2 * np.log(1.25 / self.delta_step))
            / self.eps_step
        )
        self.ready_to_delete = True

        print(
            f"[Odometer] Finalized with deletion capacity m = {self.deletion_capacity}"
        )
        print(f"[Odometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(
            f"[Odometer] ε_step = {self.eps_step:.6f}, δ_step = {self.delta_step:.2e}, σ = {self.sigma_step:.4f}"
        )

    def _solve_capacity(self) -> int:
        """
        Solve for maximum deletion capacity subject to regret constraint.

        Binary search for the largest m such that:
        (R_ins + R_del(m)) / T ≤ γ

        where:
        R_ins = L·D·√(c·C·T) (insertion regret)
        R_del(m) = (m·L/λ)·√((2ln(1.25m/δ_tot)/ε_tot)·(2ln(1/δ_B))) (deletion regret)

        Returns:
            Maximum feasible deletion capacity
        """

        def regret_bound(m):
            """Compute total regret bound for capacity m."""
            # Insertion regret term
            c_val = getattr(self, "c", 1.0)
            C_val = getattr(self, "C", 1.0)
            insertion_regret = self.L * self.D * np.sqrt(c_val * C_val * self.T)

            # Deletion regret term
            if m <= 0:
                deletion_regret = 0
            else:
                deletion_regret = (m * self.L / self.lambda_) * np.sqrt(
                    (2 * np.log(1.25 * max(m, 1) / self.delta_total) / self.eps_total)
                    * (2 * np.log(1 / self.delta_b))
                )

            return (insertion_regret + deletion_regret) / self.T

        # If even m=1 exceeds regret budget, set status and return minimal capacity
        if regret_bound(1) > self.gamma:
            self._status = "degenerate"
            print(
                f"[Odometer] Warning: Even m=1 exceeds regret budget γ={self.gamma:.4f}"
            )
            print(f"[Odometer] Regret bound for m=1: {regret_bound(1):.4f}")
            print("[Odometer] Setting capacity to 1; next delete forces retrain.")
            return 1

        # Binary search for maximum feasible capacity
        lo, hi = 1, self.T
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if regret_bound(mid) <= self.gamma:
                lo = mid
            else:
                hi = mid - 1

        self._status = "normal"
        return lo

    def spend(self) -> None:
        """
        Consume privacy budget for one deletion operation.

        Raises:
            RuntimeError: If odometer is not finalized or capacity is exceeded
        """
        if not self.ready_to_delete:
            raise RuntimeError(
                "Odometer not finalized. Call finalize() or finalize_with() before spending."
            )
        if self.deletions_count >= self.deletion_capacity:
            if (
                self.deletion_capacity == 1
                and hasattr(self, "_status")
                and self._status == "degenerate"
            ):
                raise RuntimeError(
                    "Deletion capacity is 1 and exhausted. Next delete requires retraining."
                )
            else:
                raise RuntimeError(
                    f"Deletion capacity {self.deletion_capacity} exceeded. Retraining required."
                )
        self.eps_spent += self.eps_step
        self.deletions_count += 1

    def remaining(self) -> float:
        """
        Get remaining privacy budget.

        Returns:
            Remaining privacy budget (ε_total - ε_spent)
        """
        return self.eps_total - self.eps_spent

    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """
        Get standard deviation for Gaussian noise injection.

        Args:
            sensitivity: L2 sensitivity of the deletion operation (unused in current implementation)

        Returns:
            Standard deviation for Gaussian noise

        Raises:
            ValueError: If odometer is not finalized
        """
        if self.sigma_step is None:
            raise ValueError(
                "Call finalize() or finalize_with() to compute noise scale."
            )
        return self.sigma_step


ALPHAS = [1.5, 2, 3, 4, 8, 16, 32, 64, float("inf")]


import math
import warnings


def rho_to_epsilon(rho: float, delta: float) -> float:
    """
    Convert zCDP parameter ρ to (ε, δ)-DP.
    
    Args:
        rho: zCDP parameter
        delta: Target failure probability
        
    Returns:
        Corresponding ε value for (ε, δ)-DP
    """
    if delta <= 0:
        return float("inf")
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


class ZCDPOdometer:
    """
    zCDP-based Privacy Odometer for differentially private machine unlearning.

    This class manages privacy budget allocation using zero-Concentrated Differential 
    Privacy (zCDP) accounting instead of the traditional (ε, δ)-DP per-step allocation.
    zCDP budget adds linearly; we convert to (ε, δ) only once for reporting.

    The odometer:
    1. Collects statistics during calibration phase
    2. Computes joint optimal deletion capacity m and noise scale σ based on zCDP + regret constraints
    3. Tracks zCDP budget consumption during deletions via per-delete sensitivity accounting
    4. Provides adaptive recalibration with EMA drift detection

    Attributes:
        rho_spent (float): Cumulative ρ budget used so far
        rho_total (float): Total zCDP budget (ρ)
        delta_total (float): Total failure probability (δ)
        T (int): Estimated total number of events
        gamma (float): Target average regret per event
        lambda_ (float): Strong convexity parameter
        delta_b (float): Regret bound failure probability
        deletion_capacity (Optional[int]): Maximum number of deletions allowed
        L (Optional[float]): Lipschitz constant (gradient bound)
        D (Optional[float]): Hypothesis diameter
        sigma_step (Optional[float]): Computed noise standard deviation per deletion
        deletions_count (int): Number of deletions performed
        ready_to_delete (bool): Whether odometer is finalized and ready
        m_max (Optional[int]): Upper bound for binary search on deletion capacity
        sens_bound (Optional[float]): Sensitivity upper bound used in optimization
        actual_sensitivities (list): Track actual per-delete sensitivities
    """

    def __init__(
        self,
        *,
        rho_total: float = 1.0,
        delta_total: float = 1e-5,
        T: int = 10000,
        gamma: float = 0.5,
        lambda_: float = 0.1,
        delta_b: float = 0.05,
        m_max: Optional[int] = None,
    ):
        """
        Initialize ZCDPOdometer with experiment parameters.

        Args:
            rho_total: Total zCDP budget (ρ)
            delta_total: Total failure probability (δ)
            T: Estimated total number of events
            gamma: Target average regret per event
            lambda_: Strong convexity parameter
            delta_b: Regret bound failure probability
            m_max: Upper bound for deletion capacity binary search
        """
        self.rho_spent = 0.0
        self.rho_total = rho_total
        self.delta_total = delta_total
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta_b = delta_b
        self.m_max = m_max

        # Computed after finalization
        self.deletion_capacity: int = 0
        self.L: Optional[float] = None
        self.D: Optional[float] = None
        self.sigma_step: Optional[float] = None
        self.sens_bound: Optional[float] = (
            None  # Sensitivity bound used in optimization
        )

        # Per-delete sensitivity tracking
        self.actual_sensitivities = []

        # Budget tracking
        self.deletions_count = 0
        self.ready_to_delete = False
        self._status = "unfinalized"  # Track odometer status

    def rho_cost_gaussian(self, sensitivity: float, sigma: float) -> float:
        """Return ρ for one Gaussian mechanism call."""
        return (sensitivity ** 2) / (2 * sigma ** 2)

    def spend(self, sensitivity: float, sigma: float):
        """
        Consume zCDP budget for one deletion operation with per-delete sensitivity tracking.

        Args:
            sensitivity: L2 sensitivity of the deletion operation (actual ||d|| for this delete)
            sigma: Standard deviation of Gaussian noise

        Raises:
            RuntimeError: If odometer is not finalized, capacity is exceeded, or budget overflow
        """
        if not self.ready_to_delete:
            raise RuntimeError(
                "Odometer not finalized. Call finalize() or finalize_with() before spending."
            )
        if self.deletions_count >= self.deletion_capacity:
            if (
                self.deletion_capacity == 1
                and hasattr(self, "_status")
                and self._status == "degenerate"
            ):
                raise RuntimeError(
                    "Deletion capacity is 1 and exhausted. Next delete requires retraining."
                )
            else:
                raise RuntimeError(
                    f"Deletion capacity {self.deletion_capacity} exceeded. Retraining required."
                )

        # Track actual sensitivity for future recalibration
        self.actual_sensitivities.append(sensitivity)

        # Add zCDP cost for this deletion using actual sensitivity
        self.rho_spent += self.rho_cost_gaussian(sensitivity, sigma)
        
        # Check budget overflow
        if self.rho_spent > self.rho_total:
            raise RuntimeError(
                f"zCDP budget exceeded: ρ_spent={self.rho_spent:.6f} > ρ_total={self.rho_total:.6f}"
            )

        self.deletions_count += 1

    def to_eps_delta(self, delta: float) -> float:
        """
        Convert current zCDP spending to (ε, δ)-DP.

        Args:
            delta: Target failure probability

        Returns:
            Current ε guarantee for the given δ
        """
        return rho_to_epsilon(self.rho_spent, delta)

    def over_budget(self) -> bool:
        """Check if current zCDP spending exceeds global ρ budget."""
        return self.rho_spent > self.rho_total

    def remaining_rho_delta(self) -> Tuple[float, float]:
        """Get remaining zCDP budget and δ."""
        return self.rho_total - self.rho_spent, self.delta_total

    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """
        Finalize odometer configuration using calibration statistics with joint m-σ optimization.

        This method uses statistics from the Calibrator to compute optimal deletion capacity
        and noise scale using zCDP-based joint optimization that:
        1. Binary searches on deletion capacity m (up to m_max if specified)
        2. For each m, computes minimum σ satisfying zCDP constraints
        3. Checks regret constraint using the computed σ
        4. Selects the largest feasible m with corresponding optimal σ

        Args:
            stats: Dictionary containing calibration results with keys:
                  'G': Gradient bound, 'D': Hypothesis diameter,
                  'c': Lower curvature bound, 'C': Upper curvature bound
            T_estimate: Estimated total number of events (typically max_events)
        """
        self.L = stats["G"]  # Use gradient bound as Lipschitz constant
        self.D = stats["D"]
        self.c = stats.get("c", 1.0)
        self.C = stats.get("C", 1.0)
        self.T = T_estimate

        # Use empirical sensitivity upper bound if available, otherwise fall back to L/λ
        if self.actual_sensitivities:
            # Use high quantile of observed sensitivities as upper bound
            self.sens_bound = float(np.quantile(self.actual_sensitivities, 0.95))
        else:
            # Fall back to theoretical bound
            self.sens_bound = self.L / self.lambda_

        # Joint m-σ optimization: find largest m with smallest feasible σ
        m, sigma = self._joint_optimize_m_sigma()
        self.deletion_capacity = max(1, m)
        self.sigma_step = sigma

        # Mark as ready for deletions
        self.ready_to_delete = True

        print(
            f"[ZCDPOdometer] Joint optimization: m = {self.deletion_capacity}, σ = {self.sigma_step:.4f}"
        )
        print(f"[ZCDPOdometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(f"[ZCDPOdometer] Sensitivity bound = {self.sens_bound:.4f}")
        if self.actual_sensitivities:
            print(
                f"[ZCDPOdometer] Based on {len(self.actual_sensitivities)} observed sensitivities"
            )
        else:
            print("[ZCDPOdometer] Using theoretical sensitivity bound L/λ")

    def _joint_optimize_m_sigma(self) -> Tuple[int, float]:
        """
        Joint optimization of deletion capacity m and noise scale σ using zCDP.

        Binary search for the largest m such that:
        1. Privacy constraint: m · ρ_step ≤ ρ_total with ρ_step = Δ²/(2σ²)
        2. Regret constraint: R_total(m, σ) / T ≤ γ

        For each candidate m:
        - Compute minimum σ satisfying zCDP constraint: σ = sens_bound / sqrt(2 * rho_step)
        - Check if regret bound with this σ is feasible
        - Return largest feasible m and its corresponding σ

        Returns:
            Tuple of (optimal_m, optimal_sigma)
        """

        def compute_min_sigma_for_m(m: int) -> float:
            """
            Compute minimum σ for m deletions to satisfy zCDP constraint.

            From: m · ρ_step ≤ ρ_total with ρ_step = sens_bound²/(2σ²)
            Solve: σ ≥ sens_bound / sqrt(2 * ρ_total / m)
            """
            if m <= 0:
                return 0.0

            rho_step = self.rho_total / m
            sigma_required = self.sens_bound / math.sqrt(2 * rho_step)
            return sigma_required

        def regret_bound_with_sigma(m: int, sigma: float) -> float:
            """Compute total regret bound for capacity m and noise scale σ."""
            # Insertion regret term
            insertion_regret = self.L * self.D * np.sqrt(self.c * self.C * self.T)

            # Deletion regret term using provided σ
            # From Theorem 5.5: R_del ≈ m * (L/λ) * noise_contribution
            # where noise_contribution comes from injected Gaussian noise
            if m <= 0:
                deletion_regret = 0
            else:
                # High-probability bound on ||η||: σ * sqrt(2 * log(1/δ_B))
                noise_norm_bound = sigma * np.sqrt(2 * np.log(1 / self.delta_b))
                deletion_regret = m * (self.L / self.lambda_) * noise_norm_bound

            return (insertion_regret + deletion_regret) / self.T

        # Binary search for largest feasible m
        max_m = self.m_max if self.m_max is not None else min(self.T, 10000)

        best_m = 1
        best_sigma = compute_min_sigma_for_m(1)

        # Check if even m=1 is feasible
        if (
            np.isfinite(best_sigma)
            and regret_bound_with_sigma(1, best_sigma) <= self.gamma
        ):
            # Binary search for larger m
            lo, hi = 1, max_m

            while lo <= hi:
                mid = (lo + hi) // 2
                sigma_required = compute_min_sigma_for_m(mid)

                if (
                    np.isfinite(sigma_required)
                    and regret_bound_with_sigma(mid, sigma_required) <= self.gamma
                ):
                    # This m is feasible
                    best_m = mid
                    best_sigma = sigma_required
                    lo = mid + 1
                else:
                    # This m is too large
                    hi = mid - 1

            self._status = "normal"
        else:
            # Even m=1 is not feasible
            self._status = "degenerate"
            print("[ZCDPOdometer] Warning: Even m=1 exceeds constraints")
            print(f"[ZCDPOdometer] Required σ for m=1: {best_sigma:.4f}")
            print(
                f"[ZCDPOdometer] Regret bound: {regret_bound_with_sigma(1, best_sigma):.4f} > γ={self.gamma:.4f}"
            )

        print(
            f"[ZCDPOdometer] Joint optimization selected m={best_m}, σ={best_sigma:.4f}"
        )
        print(
            f"[ZCDPOdometer] Final regret bound: {regret_bound_with_sigma(best_m, best_sigma):.4f}"
        )

        return best_m, best_sigma

    def noise_scale(self, sensitivity: Optional[float] = None) -> float:
        """
        Get standard deviation for Gaussian noise injection.

        Args:
            sensitivity: L2 sensitivity of the deletion operation (unused, we use fixed σ)

        Returns:
            Standard deviation for Gaussian noise

        Raises:
            ValueError: If odometer is not finalized
        """
        if self.sigma_step is None:
            raise ValueError(
                "Call finalize() or finalize_with() to compute noise scale."
            )
        return self.sigma_step

    def supports_recalibration(self) -> bool:
        """Check if this odometer supports mid-stream recalibration."""
        return True

    def recalibrate_with(self, new_stats: Dict[str, Any], remaining_T: int) -> None:
        """
        Recalibrate odometer with updated statistics and remaining budget.

        Args:
            new_stats: Updated calibration statistics
            remaining_T: Remaining events to process
        """
        if not self.ready_to_delete:
            raise RuntimeError("Cannot recalibrate an unfinalized odometer.")

        print(f"[ZCDPOdometer] Recalibrating with remaining T = {remaining_T}")
        print(
            f"[ZCDPOdometer] Current deletions: {self.deletions_count}/{self.deletion_capacity}"
        )

        # Update statistics
        self.L = new_stats["G"]
        self.D = new_stats["D"]
        self.c = new_stats.get("c", 1.0)
        self.C = new_stats.get("C", 1.0)
        self.T = remaining_T

        # Update sensitivity bound with latest observations
        if self.actual_sensitivities:
            self.sens_bound = float(np.quantile(self.actual_sensitivities, 0.95))
        else:
            self.sens_bound = self.L / self.lambda_

        # Recompute capacity with remaining budget
        # First, check how much zCDP budget is remaining
        rho_remaining, delta_remaining = self.remaining_rho_delta()

        # Temporarily adjust budget for reoptimization
        original_rho = self.rho_total
        original_delta = self.delta_total
        self.rho_total = max(0.01, rho_remaining)  # Ensure some budget remains
        self.delta_total = delta_remaining

        # Reoptimize with remaining budget
        m_new, sigma_new = self._joint_optimize_m_sigma()

        # Update capacity (total = already used + newly computed)
        self.deletion_capacity = self.deletions_count + m_new
        self.sigma_step = sigma_new

        # Restore original total budgets
        self.rho_total = original_rho
        self.delta_total = original_delta

        print(
            f"[ZCDPOdometer] Recalibration complete: new capacity = {self.deletion_capacity}"
        )
        print(f"[ZCDPOdometer] Updated σ = {self.sigma_step:.4f}")

    def get_sensitivity_stats(self) -> Dict[str, float]:
        """Get statistics about observed sensitivities."""
        if not self.actual_sensitivities:
            return {"count": 0}

        sensitivities = np.array(self.actual_sensitivities)
        return {
            "count": len(sensitivities),
            "mean": float(np.mean(sensitivities)),
            "std": float(np.std(sensitivities)),
            "min": float(np.min(sensitivities)),
            "max": float(np.max(sensitivities)),
            "q95": float(np.quantile(sensitivities, 0.95)),
            "q99": float(np.quantile(sensitivities, 0.99)),
        }


# Backward compatibility shim
class RDPOdometer(ZCDPOdometer):
    """
    Deprecated: Use ZCDPOdometer instead.
    
    Backward compatibility shim that converts RDP parameters to zCDP.
    """
    def __init__(self, eps_total: float = 1.0, **kw):
        warnings.warn(
            "RDPOdometer is deprecated → use ZCDPOdometer", 
            DeprecationWarning,
            stacklevel=2
        )
        # Convert (ε, δ)-DP to zCDP: ρ ≈ ε²/(2*log(1/δ))
        delta_total = kw.get('delta_total', 1e-5)
        rho_total = eps_total**2 / (2*math.log(1/delta_total))
        super().__init__(rho_total=rho_total, **kw)
