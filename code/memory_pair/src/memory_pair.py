# /code/memory_pair/src/memory_pair.py

import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from .odometer import N_star_live, m_theory_live
    from .lbfgs import LimitedMemoryBFGS
    from .metrics import loss_half_mse
    from .comparators import RollingOracle
    from .accountant import ZCDPAccountant
    from .accountant.types import Accountant
except (ModuleNotFoundError, ImportError):
    from odometer import N_star_live, m_theory_live
    from lbfgs import LimitedMemoryBFGS
    from metrics import loss_half_mse
    from comparators import RollingOracle
    from accountant import ZCDPAccountant
    from accountant.types import Accountant


@dataclass
class CalibStats:
    G: float
    D: float
    c: float
    C: float
    N_star: int


class MemoryPair:
    """
    Online learning algorithm with unlearning capabilities and zCDP privacy accounting.
    
    Simplified theory-first implementation that accepts theoretical constants as inputs.
    """

    def __init__(
        self,
        dim: int,
        G: float = 1.0,
        D: float = 1.0,
        c: float = 1.0,
        C: float = 1.0,
        accountant: Optional[Accountant] = None,
        cfg: Optional[Any] = None,
    ):
        self.theta = np.zeros(dim)
        m_max = getattr(cfg, "m_max", 10) if cfg else 10
        m_max = self._safe_int(m_max, 10)
        self.lbfgs = LimitedMemoryBFGS(m_max=m_max, cfg=cfg)

        # Store theoretical constants
        self.G = float(G)
        self.D = float(D)
        self.c = float(c)
        self.C = float(C)

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
            self.accountant = ZCDPAccountant(**acct_kwargs)

        # Finalize accountant immediately with provided constants
        if self.accountant is not None:
            stats = {"G": self.G, "D": self.D, "c": self.c, "C": self.C}
            T_estimate = getattr(cfg, "max_events", 10000) if cfg else 10000
            self.accountant.finalize(stats, T_estimate)

        self.cfg = cfg
        self.lambda_reg = getattr(cfg, "lambda_reg", 0.0) if cfg else 0.0

        # Tracking attributes
        self.cumulative_regret = 0.0
        self.regret_increment = 0.0
        self.static_regret_increment = 0.0
        self.path_regret_increment = 0.0
        self.noise_regret_cum = 0.0
        self.noise_regret_inc = 0.0

        self.events_seen = 0
        self.inserts_seen = 0
        self.deletes_seen = 0
        
        # Compute N_gamma immediately since we have constants
        try:
            from .theory import N_gamma
        except (ModuleNotFoundError, ImportError):
            from theory import N_gamma
        
        m_cap = 0
        if self.accountant is not None:
            m_cap = self.accountant.metrics().get("m_capacity", 0)
        gamma = getattr(cfg, "gamma", 0.5) if cfg else 0.5
        self.N_gamma = N_gamma(self.G, self.D, self.c, self.C, m_cap, gamma)
        self.ready_to_predict = False  # Will be enabled after N_gamma events

        # For external gradient access
        self.last_grad: Optional[np.ndarray] = None

        # Strong convexity tracking
        self.lambda_raw: Optional[float] = None
        self.sc_stable: int = 0
        self.pair_admitted: bool = True
        self.pair_damped: bool = False
        self.d_norm: float = 0.0

        # Adaptive geometry tracking
        self.S_scalar: float = 0.0
        self.S_delete: float = 0.0
        self.t: int = 0
        self.lambda_est: Optional[float] = None
        self.eta_t: float = 0.0
        self.lambda_stability_counter: int = 0
        self.sc_active: bool = False
        self.lambda_estimator = LambdaEstimator(
            ema_beta=getattr(cfg, "ema_beta", 0.9) if cfg is not None else 0.9,
            floor=getattr(cfg, "lambda_floor", 1e-6) if cfg is not None else 1e-6,
            cap=getattr(cfg, "lambda_cap", 1e3) if cfg is not None else 1e3,
        )

        # Oracle (optional)
        self.oracle: Optional[Union[RollingOracle, "StaticOracle"]] = None
        
        # Track last event for logging oracle_objective
        self._last_event_info: Optional[Tuple[np.ndarray, float]] = None
        if cfg and getattr(cfg, "enable_oracle", False):
            comparator_type = getattr(cfg, "comparator", "dynamic")
            if comparator_type == "static":
                from .comparators import StaticOracle
                lambda_reg = getattr(cfg, "lambda_reg", 0.0)
                self.oracle = StaticOracle(dim=dim, lambda_reg=lambda_reg, cfg=cfg)
            else:
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
        self.drift_kappa = getattr(cfg, "drift_kappa", 0.5) if cfg else 0.5
        self.drift_window = getattr(cfg, "drift_window", 10) if cfg else 10
        self.drift_boost_remaining = 0
        self.base_eta_t = 0.0

    @property
    def odometer(self):
        """
        Expose the odometer from the accountant for privacy accounting diagnostics.
        
        Returns the ZCDPOdometer instance if using zCDP accountant, None otherwise.
        This allows the experiment workflow to access privacy accounting diagnostics.
        """
        if hasattr(self.accountant, 'odometer'):
            return self.accountant.odometer
        return None

    # ---- helpers to sanitize cfg values ----
    def _safe_pos_float(self, val, default):
        try:
            if val is None:
                return default
            v = float(val)
            if not np.isfinite(v) or v <= 0:
                return default
            return v
        except Exception:
            return default

    def _safe_nonneg_float(self, val, default):
        try:
            if val is None:
                return default
            v = float(val)
            if not np.isfinite(v) or v < 0:
                return default
            return v
        except Exception:
            return default

    def _safe_int(self, val, default):
        try:
            if val is None:  # Handle None explicitly
                return default
            v = int(val)
            if v < 0:
                return default
            return v
        except Exception:
            return default

    def _compute_regularized_loss(self, pred: float, y: float) -> float:
        base_loss = loss_half_mse(pred, y)
        reg_term = 0.5 * self.lambda_reg * float(np.dot(self.theta, self.theta))
        return base_loss + reg_term

    def _compute_regularized_gradient(
        self, x: np.ndarray, pred: float, y: float
    ) -> np.ndarray:
        base_grad = (pred - y) * x
        reg_grad = self.lambda_reg * self.theta
        total_grad = base_grad + reg_grad
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 100.0:
            total_grad = total_grad * (100.0 / grad_norm)
        return total_grad

    def _regret_terms(self, x: np.ndarray, y: float, theta_curr: np.ndarray, theta_comp: np.ndarray, lambda_reg: float) -> Dict[str, float]:
        """
        Compute regret terms with the SAME regularized objective for both current and comparator.
        
        Args:
            x: Feature vector
            y: Target value  
            theta_curr: Current model parameters
            theta_comp: Comparator model parameters
            lambda_reg: Regularization parameter
            
        Returns:
            Dictionary with loss_curr_reg, loss_comp_reg, and regret_inc
        """
        # compute both sides with the SAME regularized objective
        pred_curr = float(theta_curr @ x)
        pred_comp = float(theta_comp @ x)
        
        loss_curr = loss_half_mse(pred_curr, y) + 0.5 * lambda_reg * float(theta_curr @ theta_curr)
        loss_comp = loss_half_mse(pred_comp, y) + 0.5 * lambda_reg * float(theta_comp @ theta_comp)
        
        return {
            "loss_curr_reg": float(loss_curr),
            "loss_comp_reg": float(loss_comp),
            "regret_inc": float(loss_curr - loss_comp),
        }

    def _update_lambda_estimate(
        self,
        g_old: np.ndarray,
        g_new: np.ndarray,
        theta_old: np.ndarray,
        theta_new: np.ndarray,
    ) -> None:
        diff_w = theta_new - theta_old
        diff_g = g_new - g_old

        denom = float(np.dot(diff_w, diff_w))
        if denom <= 1e-12:
            self.lambda_raw = None
            return

        num = float(np.dot(diff_g, diff_w))
        lambda_raw = max(num / denom, 0.0)

        if self.cfg:
            bounds = getattr(self.cfg, "lambda_est_bounds", [1e-8, 1e6])
            lambda_raw = float(np.clip(lambda_raw, bounds[0], bounds[1]))

        self.lambda_raw = lambda_raw

        beta = getattr(self.cfg, "lambda_est_beta", 0.1) if self.cfg else 0.1
        if self.lambda_est is None:
            self.lambda_est = lambda_raw
        else:
            self.lambda_est = (1 - beta) * self.lambda_est + beta * lambda_raw

        # stability counter with safe thresholds
        threshold_raw = getattr(self.cfg, "lambda_min_threshold", 1e-6) if self.cfg else 1e-6
        K_raw = getattr(self.cfg, "lambda_stability_K", 100) if self.cfg else 100
        threshold = self._safe_pos_float(threshold_raw, 1e-6)
        K = self._safe_int(K_raw, 100)

        if self.lambda_est is not None and self.lambda_est > threshold:
            self.sc_stable += 1
        else:
            self.sc_stable = 0

    @property
    def can_predict(self) -> bool:
        return self.ready_to_predict

    def insert(
        self,
        x: np.ndarray,
        y: float,
        *,
        return_grad: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        pred = float(self.theta @ x)

        base_loss_t = self._compute_regularized_loss(pred, y)
        self.events_seen += 1
        self.inserts_seen += 1

        self.regret_increment = float("nan")
        self.static_regret_increment = 0.0
        self.path_regret_increment = 0.0

        g_old = self._compute_regularized_gradient(x, pred, y)
        self.S_scalar += float(np.dot(g_old, g_old))
        self.t += 1

        self._update_step_size()

        direction = self.lbfgs.direction(g_old)
        self.d_norm = float(np.linalg.norm(direction))

        d_max_raw = getattr(self.cfg, "d_max", float("inf")) if self.cfg else float("inf")
        d_max = self._safe_pos_float(d_max_raw, float("inf"))
        if np.isfinite(d_max) and self.d_norm > d_max:
            direction = direction * (d_max / self.d_norm)
            self.d_norm = d_max

        s = self.eta_t * direction
        theta_prev = self.theta
        theta_new = theta_prev + s

        pred_new = float(theta_new @ x)
        g_new = self._compute_regularized_gradient(x, pred_new, y)
        y_vec = g_new - g_old

        self.pair_admitted, self.pair_damped = self.lbfgs.add_pair(s, y_vec)
        self.theta = theta_new

        theta_norm = np.linalg.norm(self.theta)
        if theta_norm > 10.0:
            self.theta = self.theta * (10.0 / theta_norm)

        self._update_lambda_estimate(g_old, g_new, theta_prev, theta_new)

        self.last_grad = g_old

        if self.oracle is not None:
            if hasattr(self.oracle, "maybe_update"):
                oracle_refreshed = self.oracle.maybe_update(x, y, self.theta)

            incs = self.oracle.update_regret_accounting(x, y, self.theta)
            if isinstance(incs, dict):
                self.regret_increment = incs.get("regret_increment", 0.0)
                self.static_regret_increment = incs.get("static_increment", 0.0)
                self.path_regret_increment = incs.get("path_increment", 0.0)
        else:
            # Fallback: zero-baseline comparator using same regularized objective
            theta_comp = np.zeros(self.theta.shape)  # θ* = 0
            regret_terms = self._regret_terms(x, y, self.theta, theta_comp, self.lambda_reg)
            self.regret_increment = regret_terms["regret_inc"]

        # Track last event for oracle_objective calculation in logging
        self._last_event_info = (x.copy(), y)

        if self.regret_increment is not None:
            try:
                if not (isinstance(self.regret_increment, float) and np.isnan(self.regret_increment)):
                    self.cumulative_regret += float(self.regret_increment)
            except Exception:
                pass

        self._maybe_enable_predictions()

        if return_grad:
            return pred, g_old
        return pred

    def delete(self, x: np.ndarray, y: float) -> Optional[str]:
        return self._delete_with_accountant(x, y)

    def _delete_with_accountant(self, x: np.ndarray, y: float) -> Optional[str]:
        if not self.accountant.ready():
            return "privacy_gate"

        pred = float(self.theta @ x)
        g = self._compute_regularized_gradient(x, pred, y)

        self._update_step_size()
        influence = self.lbfgs.direction(g)
        sensitivity = np.linalg.norm(influence)

        ok, sigma, reason = self.accountant.pre_delete(sensitivity)
        if not ok:
            return reason

        if hasattr(self.cfg, "gamma_delete") and self.cfg.gamma_delete is not None:
            try:
                from .theory import regret_insert_bound, regret_delete_bound
            except (ModuleNotFoundError, ImportError):
                from theory import regret_insert_bound, regret_delete_bound

            L = self.G
            ins_reg = regret_insert_bound(
                self.S_scalar,
                self.G,
                self.D,
                self.c,
                self.C,
            )

            m_used = self.accountant.metrics().get("m_used", 0)
            del_reg = regret_delete_bound(
                m_used + 1,
                L,
                self.lambda_reg or 1e-12,
                sigma,
                getattr(self.cfg, "delta_b", 0.05),
            )
            proj_avg = (ins_reg + del_reg) / max(self.events_seen or 1, 1)
            gamma_delete = getattr(self.cfg, "gamma_delete", float("inf"))
            # Allow bypassing regret gate for debugging/testing
            if proj_avg > gamma_delete and not getattr(self.cfg, "disable_regret_gate", False):
                return "regret_gate"

        self.accountant.spend(sensitivity, sigma)

        self.S_delete += float(np.dot(g, g))

        noise = np.random.normal(0, sigma, self.theta.shape)
        self.theta = self.theta - self.eta_t * influence + noise

        delta_b = getattr(self.cfg, "delta_b", 0.05)
        lambda_safe = max(self.lambda_reg, 1e-12)
        delta_reg = (
            (self.G / lambda_safe)
            * sigma
            * np.sqrt(2 * np.log(1 / max(delta_b, 1e-12)))
        )
        self.noise_regret_inc = float(delta_reg)
        self.noise_regret_cum += float(delta_reg)

        self.events_seen += 1
        self.deletes_seen += 1

        self._maybe_enable_predictions()

        return None

    def _update_step_size(self) -> None:
        eps = getattr(self.cfg, "adagrad_eps", 1e-12) if self.cfg else 1e-12

        # Use provided D constant directly
        D_bound = self.D

        # eta_max from cfg; sanitize
        eta_max_raw = getattr(self.cfg, "eta_max", 1.0) if self.cfg else 1.0
        eta_max = self._safe_pos_float(eta_max_raw, 1.0)

        # Base step size
        if self._lambda_is_stable():
            lambda_safe = max(self.lambda_est, 1e-6)
            t_safe = max(self.t, 1)
            self.base_eta_t = 1.0 / (lambda_safe * t_safe)
            self.sc_active = True
        else:
            self.base_eta_t = D_bound / np.sqrt(self.S_scalar + eps)
            self.sc_active = False

        self.base_eta_t = min(self.base_eta_t, eta_max)

        # Apply drift boost if active (no change to comparison semantics)
        if hasattr(self, 'drift_boost_remaining') and self.drift_boost_remaining > 0:
            self.eta_t = self.base_eta_t * (1.0 + self.drift_kappa)
            self.drift_boost_remaining -= 1
        else:
            self.eta_t = self.base_eta_t

        self.eta_t = min(self.eta_t, eta_max)

    def _lambda_is_stable(self) -> bool:
        if not self.cfg or not getattr(self.cfg, "strong_convexity", False):
            return False

        threshold_raw = getattr(self.cfg, "lambda_min_threshold", 1e-6)
        K_raw = getattr(self.cfg, "lambda_stability_K", 100)

        threshold = self._safe_pos_float(threshold_raw, 1e-6)
        K = self._safe_int(K_raw, 100)

        return (
            self.lambda_est is not None
            and self.lambda_est > threshold
            and self.sc_stable >= K
        )

    def _maybe_enable_predictions(self) -> None:
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
        if self.events_seen == 0:
            return float("inf")
        return self.cumulative_regret / self.events_seen

    def get_current_loss_reg(self, x: np.ndarray, y: float) -> float:
        pred = float(self.theta @ x)
        return self._compute_regularized_loss(pred, y)

    def get_stepsize_policy(self) -> Dict[str, Any]:
        if hasattr(self, 'sc_active') and self.sc_active:
            policy = "strongly-convex"
            params = {
                "lambda": self.lambda_est,
                "t": self.t,
                "eta_formula": "1/(λ*t)"
            }
        else:
            policy = "adagrad"
            # Use provided D constant directly
            D_val = self.D
            params = {
                "D": D_val,
                "S_t": self.S_scalar,
                "eta_formula": "D/√S_t"
            }
        
        return {
            "stepsize_policy": policy,
            "stepsize_params": params
        }

    def get_metrics_dict(self) -> dict:
        metrics = {
            "lambda_est": self.lambda_est,
            "lambda_raw": self.lambda_raw,
            "sc_stable": self.sc_stable,
            "pair_admitted": self.pair_admitted,
            "pair_damped": self.pair_damped,
            "d_norm": self.d_norm,
            "eta_t": self.eta_t,
            "sc_active": self.sc_active,
            "drift_boost_remaining": getattr(self, "drift_boost_remaining", 0),
            "base_eta_t": getattr(self, "base_eta_t", self.eta_t),
        }

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

        if self.oracle is not None:
            try:
                oracle_metrics = self.oracle.get_oracle_metrics()
            except Exception:
                oracle_metrics = {
                    "P_T": 0.0,
                    "P_T_est": 0.0,
                    "drift_flag": False,
                    "regret_dynamic": 0.0,
                    "regret_static_term": 0.0,
                    "regret_path_term": 0.0,
                }

            oracle_metrics["P_T"] = oracle_metrics.get(
                "P_T", oracle_metrics.get("P_T_est", 0.0)
            )
            oracle_metrics["P_T_est"] = oracle_metrics.get(
                "P_T_est", oracle_metrics["P_T"]
            )
            oracle_metrics["drift_flag"] = oracle_metrics.get("drift_detected", False)

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

        stepsize_info = self.get_stepsize_policy()
        metrics.update(stepsize_info)

        return metrics

    def get_live_diagnostics(self) -> Dict[str, Any]:
        diagnostics = {}
        if hasattr(self, "cfg") and self.cfg is not None:
            diagnostics["gamma_bar"] = getattr(self.cfg, "gamma_bar", None)
            diagnostics["gamma_split"] = getattr(self.cfg, "gamma_split", None)
            diagnostics["gamma_ins"] = getattr(self.cfg, "gamma_insert", None)
            diagnostics["gamma_del"] = getattr(self.cfg, "gamma_delete", None)

        if self.odometer and hasattr(self.odometer, "G_hat") and hasattr(self.odometer, "D_hat"):
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

    def get_comparator_metrics(self) -> Dict[str, Any]:
        """Get current comparator metrics for logging."""
        if self.oracle is not None:
            # Oracle is enabled - get metrics from oracle
            metrics = self.oracle.get_oracle_metrics()
        else:
            # Oracle disabled - use zero proxy metrics
            metrics = {
                "comparator_type": "zero_proxy",
                "oracle_refresh_step": 0,
                "oracle_refreshes": 0,
                "regret_static": self.cumulative_regret,  # Total regret vs zero baseline
                "is_calibrated": True,  # Zero oracle is always "calibrated"
                "events_seen": self.events_seen,
            }
        return metrics

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get comprehensive metrics for logging, including regret and oracle information."""
        metrics = {
            # Core regret metrics
            "regret_increment": self.regret_increment,
            "cum_regret": self.cumulative_regret,
            "static_regret_increment": self.static_regret_increment,
            "path_regret_increment": self.path_regret_increment,
            "noise_regret_increment": self.noise_regret_inc,
            "noise_regret_cum": self.noise_regret_cum,
            "cum_regret_with_noise": self.cumulative_regret + self.noise_regret_cum,
            
            # Basic model state
            "events_seen": self.events_seen,
            "inserts_seen": self.inserts_seen,
            "deletes_seen": self.deletes_seen,
            "ready_to_predict": self.ready_to_predict,
            "N_gamma": self.N_gamma,
            
            # Theoretical constants
            "G": self.G,
            "D": self.D,
            "c": self.c,
            "C": self.C,
            
            # Lambda regularization
            "lambda_reg": self.lambda_reg,
            "lambda_raw": self.lambda_raw,
        }
        
        # Add comparator/oracle metrics
        comparator_metrics = self.get_comparator_metrics()
        metrics.update(comparator_metrics)
        
        # Set oracle_objective to comparator's regularized loss if we have regret info
        if self.oracle is not None and hasattr(self, '_last_event_info'):
            # If we tracked the last event, compute the comparator's loss
            x, y = self._last_event_info
            if self.oracle.is_calibrated:
                oracle_params = self.oracle.get_current_oracle()
                if oracle_params is not None:
                    pred_comp = float(oracle_params @ x)
                    oracle_objective = (
                        loss_half_mse(pred_comp, y) + 
                        0.5 * self.lambda_reg * float(oracle_params @ oracle_params)
                    )
                    metrics["oracle_objective"] = oracle_objective
        elif self.oracle is None and hasattr(self, '_last_event_info'):
            # Zero oracle case
            x, y = self._last_event_info
            oracle_objective = loss_half_mse(0.0, y)  # Zero prediction, no regularization term
            metrics["oracle_objective"] = oracle_objective
            
        # Add privacy/accountant metrics if available
        if hasattr(self, 'accountant') and self.accountant is not None:
            try:
                accountant_metrics = self.accountant.metrics()
                metrics.update(accountant_metrics)
            except Exception:
                pass
        
        return metrics


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