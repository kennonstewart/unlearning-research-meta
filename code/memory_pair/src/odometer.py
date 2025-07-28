# /code/memory_pair/src/odometer.py
# Privacy odometer to track budget consumption for deletions.
# This module implements a privacy odometer to manage privacy budgets for unlearning operations.
import numpy as np
from typing import Dict, Any, Optional


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
        self._status = 'unfinalized'  # Track odometer status

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
        self.sigma_step = (self.L / self.lambda_) * np.sqrt(2 * np.log(1.25 / self.delta_step)) / self.eps_step
        
        # Mark as ready for deletions
        self.ready_to_delete = True
        
        print(f"[Odometer] Finalized with deletion capacity m = {self.deletion_capacity}")
        print(f"[Odometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(f"[Odometer] ε_step = {self.eps_step:.6f}, δ_step = {self.delta_step:.2e}, σ = {self.sigma_step:.4f}")

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

        print(f"[Odometer] Finalized with deletion capacity m = {self.deletion_capacity}")
        print(f"[Odometer] L = {self.L:.4f}, D = {self.D:.4f}")
        print(f"[Odometer] ε_step = {self.eps_step:.6f}, δ_step = {self.delta_step:.2e}, σ = {self.sigma_step:.4f}")

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
            c_val = getattr(self, 'c', 1.0)
            C_val = getattr(self, 'C', 1.0)
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
            self._status = 'degenerate'
            print(f"[Odometer] Warning: Even m=1 exceeds regret budget γ={self.gamma:.4f}")
            print(f"[Odometer] Regret bound for m=1: {regret_bound(1):.4f}")
            print(f"[Odometer] Setting capacity to 1; next delete forces retrain.")
            return 1

        # Binary search for maximum feasible capacity
        lo, hi = 1, self.T
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if regret_bound(mid) <= self.gamma:
                lo = mid
            else:
                hi = mid - 1
        
        self._status = 'normal'
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
            if self.deletion_capacity == 1 and hasattr(self, '_status') and self._status == 'degenerate':
                raise RuntimeError(
                    f"Deletion capacity is 1 and exhausted. Next delete requires retraining."
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
            raise ValueError("Call finalize() or finalize_with() to compute noise scale.")
        return self.sigma_step
