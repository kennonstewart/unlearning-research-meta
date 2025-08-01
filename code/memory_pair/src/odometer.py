# /code/memory_pair/src/odometer.py
# Privacy odometer to track budget consumption for deletions.
# This module implements a privacy odometer to manage privacy budgets for unlearning operations.
import numpy as np
from typing import Dict, Any, Optional, Tuple


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
        self.deletion_capacity: Optional[int] = None
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
        print(f"[PrivacyOdometer] finalize_with called with stats: {stats}")
        print(f"[PrivacyOdometer] T_estimate: {T_estimate}")
        
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
            f"[PrivacyOdometer] Finalized with deletion capacity m = {self.deletion_capacity}"
        )
        print(f"[PrivacyOdometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(
            f"[PrivacyOdometer] ε_step = {self.eps_step:.6f}, δ_step = {self.delta_step:.2e}, σ = {self.sigma_step:.4f}"
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


class RDPOdometer:
    """
    RDP-based Privacy Odometer for differentially private machine unlearning.

    This class manages privacy budget allocation using Rényi Differential Privacy (RDP)
    accounting instead of the traditional (ε, δ)-DP per-step allocation. RDP provides
    tighter composition bounds by tracking privacy loss at multiple orders α.

    The odometer:
    1. Collects statistics during calibration phase
    2. Computes joint optimal deletion capacity m and noise scale σ based on RDP + regret constraints
    3. Tracks RDP budget consumption during deletions via per-delete sensitivity accounting
    4. Provides adaptive recalibration with EMA drift detection

    Attributes:
        alphas (list): RDP orders to track (default: [1.5, 2, 3, 4, 8, 16, 32, 64, inf])
        eps_alpha_spent (dict): Privacy budget spent at each RDP order
        eps_total (float): Total privacy budget (ε)
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
        eps_total: float = 1.0,
        delta_total: float = 1e-5,
        T: int = 10000,
        gamma: float = 0.5,
        lambda_: float = 0.1,
        delta_b: float = 0.05,
        alphas: Optional[list] = None,
        m_max: Optional[int] = None,
    ):
        """
        Initialize RDPOdometer with experiment parameters.

        Args:
            eps_total: Total privacy budget (ε)
            delta_total: Total failure probability (δ)
            T: Estimated total number of events
            gamma: Target average regret per event
            lambda_: Strong convexity parameter
            delta_b: Regret bound failure probability
            alphas: RDP orders to track (default: ALPHAS)
            m_max: Upper bound for deletion capacity binary search
        """
        self.alphas = alphas or ALPHAS
        self.eps_alpha_spent = {a: 0.0 for a in self.alphas}
        self.eps_total = eps_total
        self.delta_total = delta_total
        self.T = T
        self.gamma = gamma
        self.lambda_ = lambda_
        self.delta_b = delta_b
        self.m_max = m_max

        # Computed after finalization
        self.deletion_capacity: Optional[int] = None
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

    def step_cost_gaussian(self, alpha, sensitivity, sigma):
        """Compute RDP cost for one Gaussian mechanism step."""
        if alpha == float("inf"):
            # For α = ∞, RDP cost is sensitivity²/(2σ²)
            return (sensitivity**2) / (2 * sigma**2)
        return alpha * (sensitivity**2) / (2 * sigma**2)

    def spend(self, sensitivity, sigma):
        """
        Consume RDP budget for one deletion operation with per-delete sensitivity tracking.

        Args:
            sensitivity: L2 sensitivity of the deletion operation (actual ||d|| for this delete)
            sigma: Standard deviation of Gaussian noise

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

        # Track actual sensitivity for future recalibration
        self.actual_sensitivities.append(sensitivity)

        # Add RDP cost for this deletion using actual sensitivity
        for a in self.alphas:
            self.eps_alpha_spent[a] += self.step_cost_gaussian(a, sensitivity, sigma)

        self.deletions_count += 1

    def to_eps_delta(self, delta):
        """
        Convert current RDP curve to (ε, δ)-DP.

        Args:
            delta: Target failure probability

        Returns:
            Tightest ε guarantee for the given δ
        """
        if delta <= 0:
            return float("inf")

        return min(
            self.eps_alpha_spent[a] + np.log(1 / delta) / (a - 1)
            for a in self.alphas
            if a > 1 and np.isfinite(a)
        )

    def over_budget(self):
        """Check if current RDP spending exceeds global (ε, δ) budget."""
        if self.eps_total and self.delta_total:
            return self.to_eps_delta(self.delta_total) > self.eps_total
        return False

    def finalize_with(self, stats: Dict[str, Any], T_estimate: int) -> None:
        """
        Finalize odometer configuration using calibration statistics with joint m-σ optimization.

        This method uses statistics from the Calibrator to compute optimal deletion capacity
        and noise scale using enhanced RDP-based joint optimization that:
        1. Binary searches on deletion capacity m (up to m_max if specified)
        2. For each m, computes minimum σ satisfying RDP constraints across all α orders
        3. Checks regret constraint using the computed σ
        4. Selects the largest feasible m with corresponding optimal σ

        Args:
            stats: Dictionary containing calibration results with keys:
                  'G': Gradient bound, 'D': Hypothesis diameter,
                  'c': Lower curvature bound, 'C': Upper curvature bound
            T_estimate: Estimated total number of events (typically max_events)
        """
        print(f"[RDPOdometer] finalize_with called with stats: {stats}")
        print(f"[RDPOdometer] T_estimate: {T_estimate}")
        
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
            f"[RDPOdometer] Joint optimization: m = {self.deletion_capacity}, σ = {self.sigma_step:.4f}"
        )
        print(f"[RDPOdometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(f"[RDPOdometer] Sensitivity bound = {self.sens_bound:.4f}")
        if self.actual_sensitivities:
            print(
                f"[RDPOdometer] Based on {len(self.actual_sensitivities)} observed sensitivities"
            )
        else:
            print("[RDPOdometer] Using theoretical sensitivity bound L/λ")

    def _joint_optimize_m_sigma(self) -> Tuple[int, float]:
        """
        Joint optimization of deletion capacity m and noise scale σ.

        Binary search for the largest m such that:
        1. Privacy constraint: There exists σ such that m deletions with sensitivity
           sens_bound satisfy RDP→(ε,δ) conversion
        2. Regret constraint: R_total(m, σ) / T ≤ γ

        For each candidate m:
        - Compute minimum σ satisfying RDP constraints across all α orders
        - Check if regret bound with this σ is feasible
        - Return largest feasible m and its corresponding σ

        Returns:
            Tuple of (optimal_m, optimal_sigma)
        """

        def compute_min_sigma_for_m(m: int) -> float:
            """
            Compute minimum σ for m deletions to satisfy RDP→(ε,δ) constraint.

            For Gaussian mechanism: εα(m steps) = m * α * sens_bound² / (2σ²)
            RDP→(ε,δ) conversion: ε ≥ min_α [εα + log(1/δ)/(α-1)]

            Solve: σ ≥ max_α sqrt(m * α * sens_bound² / (2 * εα_budget[α]))
            where εα_budget[α] comes from inverting the conversion constraint.
            """
            if m <= 0:
                return 0.0

            # For each α, compute required εα budget to achieve final (ε,δ)
            # Approximate by allocating budget proportionally across orders
            min_sigma_candidates = []

            for alpha in self.alphas:
                if alpha <= 1 or not np.isfinite(alpha):
                    continue

                # For this alpha, what's the maximum εα we can spend?
                # From RDP→(ε,δ): ε ≥ εα + log(1/δ)/(α-1)
                # So: εα ≤ ε - log(1/δ)/(α-1)
                eps_alpha_budget = self.eps_total - np.log(1 / self.delta_total) / (
                    alpha - 1
                )

                if eps_alpha_budget <= 0:
                    continue

                # From Gaussian RDP: εα = m * α * sens_bound² / (2σ²)
                # Solve for σ: σ² ≥ m * α * sens_bound² / (2 * εα_budget)
                sigma_squared = (
                    m * alpha * (self.sens_bound**2) / (2 * eps_alpha_budget)
                )
                if sigma_squared > 0:
                    min_sigma_candidates.append(np.sqrt(sigma_squared))

            # Return the maximum σ needed across all constraints
            return max(min_sigma_candidates) if min_sigma_candidates else float("inf")

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
            print("[RDPOdometer] Warning: Even m=1 exceeds constraints")
            print(f"[RDPOdometer] Required σ for m=1: {best_sigma:.4f}")
            print(
                f"[RDPOdometer] Regret bound: {regret_bound_with_sigma(1, best_sigma):.4f} > γ={self.gamma:.4f}"
            )

        print(
            f"[RDPOdometer] Joint optimization selected m={best_m}, σ={best_sigma:.4f}"
        )
        print(
            f"[RDPOdometer] Final regret bound: {regret_bound_with_sigma(best_m, best_sigma):.4f}"
        )

        return best_m, best_sigma

    def _solve_capacity_rdp(self) -> int:
        """
        Legacy method: Solve for maximum deletion capacity subject to regret constraint using RDP.

        Note: This method is kept for backward compatibility. New code should use
        _joint_optimize_m_sigma() which provides better joint optimization.

        Returns:
            Maximum feasible deletion capacity
        """

        def regret_bound_rdp(m):
            """Compute total regret bound for capacity m using RDP accounting."""
            # Insertion regret term
            c_val = getattr(self, "c", 1.0)
            C_val = getattr(self, "C", 1.0)
            insertion_regret = self.L * self.D * np.sqrt(c_val * C_val * self.T)

            # Deletion regret term using RDP
            if m <= 0:
                deletion_regret = 0
            else:
                # Compute required σ for m deletions under RDP constraint
                sensitivity = self.L / self.lambda_
                sigma_required = self._compute_sigma(sensitivity, m)

                deletion_regret = (m * self.L / self.lambda_) * np.sqrt(
                    (sigma_required**2) * (2 * np.log(1 / self.delta_b))
                )

            return (insertion_regret + deletion_regret) / self.T

        # If even m=1 exceeds regret budget, set status and return minimal capacity
        if regret_bound_rdp(1) > self.gamma:
            self._status = "degenerate"
            print(
                f"[RDPOdometer] Warning: Even m=1 exceeds regret budget γ={self.gamma:.4f}"
            )
            print(f"[RDPOdometer] Regret bound for m=1: {regret_bound_rdp(1):.4f}")
            print("[RDPOdometer] Setting capacity to 1; next delete forces retrain.")
            return 1

        # Binary search for maximum feasible capacity
        lo, hi = 1, self.T
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if regret_bound_rdp(mid) <= self.gamma:
                lo = mid
            else:
                hi = mid - 1

        self._status = "normal"
        return lo

    def _compute_sigma(self, sensitivity: float, m: int) -> float:
        """
        Compute required noise σ for m deletions to satisfy RDP→(ε,δ) constraint.

        Args:
            sensitivity: L2 sensitivity per deletion
            m: Number of deletions

        Returns:
            Required standard deviation σ
        """

        # Binary search for minimum σ such that m deletions stay within budget
        def rdp_cost_for_sigma(sigma):
            """Compute RDP cost for m deletions with given σ."""
            return min(
                m * self.step_cost_gaussian(a, sensitivity, sigma)
                + np.log(1 / self.delta_total) / (a - 1)
                for a in self.alphas
                if a > 1 and np.isfinite(a)
            )

        # Binary search for σ
        sigma_lo, sigma_hi = 1e-6, 1000.0
        for _ in range(50):  # Limit iterations
            sigma_mid = (sigma_lo + sigma_hi) / 2
            if rdp_cost_for_sigma(sigma_mid) <= self.eps_total:
                sigma_hi = sigma_mid
            else:
                sigma_lo = sigma_mid

            if sigma_hi - sigma_lo < 1e-8:
                break

        return sigma_hi

    def remaining_rdp(self) -> Dict[float, float]:
        """Get remaining RDP budget at each order."""
        return {
            a: max(0, self.eps_total - self.eps_alpha_spent[a]) for a in self.alphas
        }

    def remaining_eps_delta(self) -> Tuple[float, float]:
        """Get remaining (ε, δ) budget after RDP conversion."""
        eps_used = self.to_eps_delta(self.delta_total)
        return max(0, self.eps_total - eps_used), self.delta_total

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

        print(f"[RDPOdometer] Recalibrating with remaining T = {remaining_T}")
        print(
            f"[RDPOdometer] Current deletions: {self.deletions_count}/{self.deletion_capacity}"
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
        # First, check how much RDP budget is remaining
        eps_remaining, delta_remaining = self.remaining_eps_delta()

        # Temporarily adjust budget for reoptimization
        original_eps = self.eps_total
        original_delta = self.delta_total
        self.eps_total = max(0.01, eps_remaining)  # Ensure some budget remains
        self.delta_total = delta_remaining

        # Reoptimize with remaining budget
        m_new, sigma_new = self._joint_optimize_m_sigma()

        # Update capacity (total = already used + newly computed)
        self.deletion_capacity = self.deletions_count + m_new
        self.sigma_step = sigma_new

        # Restore original total budgets
        self.eps_total = original_eps
        self.delta_total = original_delta

        print(
            f"[RDPOdometer] Recalibration complete: new capacity = {self.deletion_capacity}"
        )
        print(f"[RDPOdometer] Updated σ = {self.sigma_step:.4f}")

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
