# AGENTS.md

Design spec for the **Calibrator helper**, **MemoryPair state machine**, and **PrivacyOdometer optimizer** used in the deletion-capacity experiment pipeline.

---

## 0. Goals

1. **Calibrate** theoretical constants (\(G, D, c, C\)) from a short bootstrap run.
2. **Compute** sample complexity \(N_*\) and deletion capacity \(m\), then derive per-deletion privacy costs (\(\varepsilon_{\text{step}}, \delta_{\text{step}}\)).
3. **Expose a clean developer API** with distinct phases:
   - `CALIBRATION`  → estimate constants, finalize odometer
   - `LEARNING`     → insert-only until `ready_to_predict`
   - `INTERLEAVING` → arbitrary insert/delete stream with accounting
4. Provide explicit readiness flags: `ready_to_predict` (model) and `ready_to_delete` (odometer).

---

## 1. High-Level Architecture

```
┌──────────────────┐      observe(grad, θ)      ┌─────────────────────┐
│   MemoryPair     │ ─────────────────────────▶ │     Calibrator      │
│  (state machine) │                            │ (stats & N* solver) │
└────────┬─────────┘                            └──────────┬──────────┘
         │ finalize_calibration(stats)                    │ m, ε_step, δ_step
         ▼                                                ▼
┌──────────────────┐      optimize(m, ε, δ)      ┌─────────────────────┐
│ PrivacyOdometer  │ ◀────────────────────────── │  Odometer Optimizer │
└──────────────────┘                              └─────────────────────┘
```

### Components

- **Calibrator**: Collects gradients & parameter snapshots, estimates \(G, D, c, C\). Calculates \(N_*\).
- **MemoryPair**: Implements the phase machine and algorithmic steps (`calibrate_step`, `finalize_calibration`, `insert`, `delete`). Holds `ready_to_predict`.
- **PrivacyOdometer**: After calibration, solves for \(m\), sets `ε_step`, `δ_step`, tracks spend; exposes `ready_to_delete`.

---

## 2. Phases & State Machine

```python
from enum import Enum

class Phase(Enum):
    CALIBRATION = 1
    LEARNING    = 2
    INTERLEAVING = 3
```

### Transitions

1. **Init → CALIBRATION** (default)
2. **CALIBRATION → LEARNING** once `finalize_calibration()` is called and `N_*` is determined.
3. **LEARNING → INTERLEAVING** once `ready_to_predict` is `True` (i.e., `inserts_seen >= N_*`).

Optional: allow reverting or re-calibration if drift is detected (future work).

---

## 3. Calibrator Helper

### Responsibilities

- Track:
  - Gradient norms: \(g_t = \|\nabla \ell_t(w_t)\|_2\). Store max or high-quantile → \(\hat{G}\).
  - Parameter trajectory: \(w_t\). Compute \(D = \max_t \|w_t - w_0\|_2\) (proxy for hypothesis diameter).
  - L-BFGS curvature matrix bounds: eigenvalue bounds \(c, C\) of current Hessian approximation. Fallback to (1, 1) if unavailable.
- Compute sample complexity: \(N_* = \left\lceil \left( \frac{ \hat{G} \cdot \hat{D} \cdot \sqrt{cC} }{ \gamma } \right)^2 \right\rceil\)
- Provide a stats bundle to `MemoryPair.finalize_calibration()`.

### API Sketch

```python
class Calibrator:
    def __init__(self, quantile: float = 1.0):
        self.grad_norms = []
        self.thetas = []
        self.quantile = quantile  # e.g. 0.95 to clip outliers
        self.c_hat = None
        self.C_hat = None

    def observe(self, grad: np.ndarray, theta: np.ndarray):
        self.grad_norms.append(np.linalg.norm(grad))
        self.thetas.append(theta.copy())

    def estimate_bounds(self, model) -> tuple[float, float]:
        # try model.lbfgs_bounds() or eigenvalues of B
        ...
        return c_hat, C_hat

    def finalize(self, gamma: float, model) -> dict:
        G_hat = float(np.quantile(self.grad_norms, self.quantile))
        D_hat = float(max(np.linalg.norm(th - self.thetas[0]) for th in self.thetas))
        self.c_hat, self.C_hat = self.estimate_bounds(model)
        N_star = int(np.ceil(((G_hat * D_hat * np.sqrt(self.c_hat * self.C_hat)) / gamma) ** 2))
        return {
            "G": G_hat,
            "D": D_hat,
            "c": self.c_hat,
            "C": self.C_hat,
            "N_star": max(1, N_star),
        }
```

---

## 4. MemoryPair State Machine API

### Methods

- `calibrate_step(x, y)`
  - Runs one insert-like step but **logs** grad/θ to `Calibrator` instead of updating regret gates.
- `finalize_calibration(stats)`
  - Accepts dict from `Calibrator.finalize()`, stores `N_star`, sets phase → `LEARNING`.
  - Calls `odometer.finalize_with(stats)` to compute \(m, ε_{step}, δ_{step}\).
- `insert(x, y)`
  - Standard online update; increments counters. If still < `N_star`, predictions may be withheld or flagged.
- `delete(x, y)`
  - Uses odometer to check capacity & noise scale before applying update.
- `can_predict` (property or method)
  - `return self.ready_to_predict` → `events_seen >= N_star` and phase != CALIBRATION.

### Flags

- `ready_to_predict: bool` – set when `inserts_seen >= N_star`.
- Internal: `events_seen`, `inserts_seen`, `deletes_seen`.

### Example Skeleton

```python
class MemoryPair:
    def __init__(self, dim, odometer, calibrator=None):
        self.theta = np.zeros(dim)
        self.lbfgs = ...
        self.phase = Phase.CALIBRATION
        self.calibrator = calibrator or Calibrator()
        self.odometer = odometer
        self.N_star = None
        self.ready_to_predict = False
        self.events_seen = 0
        self.inserts_seen = 0

    def calibrate_step(self, x, y):
        pred = float(self.theta @ x)
        grad_old = (pred - y) * x
        direction = self.lbfgs.direction(grad_old)
        # update theta, lbfgs as usual
        ...
        self.calibrator.observe(grad_old, self.theta)
        self.events_seen += 1
        self.inserts_seen += 1
        return pred

    def finalize_calibration(self, gamma):
        stats = self.calibrator.finalize(gamma, self)
        self.N_star = stats["N_star"]
        # finalizes odometer with stats (G, D, etc.)
        self.odometer.finalize_with(stats, T_estimate=self.N_star)  # or max_events
        self.phase = Phase.LEARNING

    def insert(self, x, y):
        pred = float(self.theta @ x)
        # update regret, theta, lbfgs
        ...
        self.events_seen += 1
        self.inserts_seen += 1
        if self.phase == Phase.LEARNING and self.inserts_seen >= self.N_star:
            self.ready_to_predict = True
            self.phase = Phase.INTERLEAVING
        return pred

    def delete(self, x, y):
        if not self.odometer.ready_to_delete:
            raise RuntimeError("Odometer not finalized or capacity depleted.")
        self.odometer.spend()
        # compute influence d, add noise N(0, σ²I)
        sigma = self.odometer.noise_scale()
        ...

    @property
    def can_predict(self) -> bool:
        return self.ready_to_predict
```

---

## 5. PrivacyOdometer Optimization

### Inputs (from `stats`)

- `G, D, c, C` (optional but useful)
- Experiment-level: `T` (total planned events), `gamma`, `eps_total`, `delta_total`, `lambda_`, `delta_B`.

### Outputs

- `m`: deletion capacity (max valid deletes)
- `eps_step`, `delta_step`
- `sigma_step`: Gaussian noise std for each deletion
- `ready_to_delete`: True after finalize

### Optimization Strategy

1. Compute insertion regret term: \(R_{ins} = GD\sqrt{cC\,T}\).
2. Binary-search for largest `m` such that total regret/T ≤ γ:
   $$
     \frac{R_{ins} + R_{del}(m)}{T} \le \gamma,
   $$
   where
   $$
     R_{del}(m) = \frac{mL}{\lambda} \sqrt{\frac{2\ln(1.25m/\delta_{tot})}{\varepsilon_{tot}}\, 2\ln(1/\delta_B)}.
   $$
3. Set \(\varepsilon_{step} = \varepsilon_{tot}/m, \delta_{step} = \delta_{tot}/m\).
4. \(\sigma = \frac{L}{\lambda} \sqrt{\frac{2\ln(1.25/\delta_{step})}{\varepsilon_{step}^2}}\).

### API Sketch

```python
class PrivacyOdometer:
    def __init__(self, eps_total, delta_total, lambda_, delta_B=0.05, T=100000, gamma=0.05):
        ...
        self.ready_to_delete = False

    def finalize_with(self, stats: dict, T_estimate: int):
        self.L = stats["G"]
        self.D = stats["D"]
        self.c = stats["c"]
        self.C = stats["C"]
        self.T = T_estimate  # or max_events
        m = self._solve_capacity()
        self.deletion_capacity = max(1, m)
        self.eps_step = self.eps_total / self.deletion_capacity
        self.delta_step = self.delta_total / self.deletion_capacity
        self.sigma_step = (self.L / self.lambda_) * np.sqrt(2*np.log(1.25/self.delta_step)) / self.eps_step
        self.ready_to_delete = True

    def _solve_capacity(self) -> int:
        def regret_bound(m):
            R_ins = self.L * self.D * np.sqrt(self.c * self.C * self.T)
            R_del = (m * self.L / self.lambda_) * np.sqrt(
                (2*np.log(1.25*max(m,1)/self.delta_total) / self.eps_total) * (2*np.log(1/self.delta_B))
            )
            return (R_ins + R_del) / self.T
        if regret_bound(1) > self.gamma:
            return 0
        lo, hi = 1, self.T
        while lo < hi:
            mid = (lo + hi + 1)//2
            if regret_bound(mid) <= self.gamma:
                lo = mid
            else:
                hi = mid - 1
        return lo
```

---

## 6. Developer Workflow

1. **Create model & odometer**
2. **Phase: CALIBRATION**
   - Loop `calibrate_step` for \~500–1000 inserts
   - Call `finalize_calibration(gamma)` → transitions to LEARNING
3. **Phase: LEARNING**
   - Insert until `ready_to_predict` is True (\(inserts \ge N_*\))
4. **Phase: INTERLEAVING**
   - Run workload stream (e.g., 10 inserts : 1 delete)
   - Odometer enforces capacity & noise

---

## 7. Logging & Testing

- Log: `phase`, `G`, `D`, `c`, `C`, `N_star`, `m`, `eps_step`, `delta_step`, `sigma_step`, `eps_spent`, `capacity_remaining`.
- Unit tests:
  - Calibrator returns positive N\*; monotone in γ
  - Odometer returns `m >= 1`; blocks at m+1 deletions
  - State transitions occur correctly

---

## 8. Error Handling & Edge Cases

- If `regret_bound(1) > γ`: degrade gracefully (set `m = 1`, warn).
- If calibration collects too few points, fall back to defaults.
- If LBFGS bounds can’t be computed, default (c, C)=(1,1).
- Clip extreme gradients to prevent absurd N\*.

---

## 9. Future Extensions

- Adaptive re-calibration on detected drift.
- Different privacy accountants (RDP, zCDP) with tighter composition.
- Phase-specific hooks for distributed/federated settings.

---

**End of AGENTS.md**

