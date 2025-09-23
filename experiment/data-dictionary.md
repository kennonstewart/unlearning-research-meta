# 📊 Data Dictionary — Memory Pair Repository

This document describes the data structures, schemas, and parameters used in the Memory Pair project. It covers:

* [1. Experiment Results STAR Schema](#1-experiment-results-star-schema)
* [2. Grid Configuration Parameters](#2-grid-configuration-parameters)
* [3. Theoretical Variables & Metrics](#3-theoretical-variables--metrics)

---

## 1. Experiment Results STAR Schema

Results are materialized in DuckDB under a **star schema** with one fact table and multiple dimension tables.

### `analytics.dim_run`

Grain: 1 row per run (`grid_id` × `seed`).

| Column     | Type    | Description                |
| ---------- | ------- | -------------------------- |
| grid\_id   | VARCHAR | Experiment grid identifier |
| seed       | BIGINT  | RNG seed                   |
| run\_id    | VARCHAR | Composite key (grid_id + seed) |

---

### `analytics.dim_event_type`

Grain: unique event types.

| Column                    | Type    | Description                          |
| ------------------------- | ------- | ------------------------------------ |
| event\_type               | VARCHAR | Event category (e.g. calibrate, insert, delete) |
| event\_type\_description  | VARCHAR | Operation label (same as event_type) |

---

### `analytics.dim_parameters`

Grain: 1 row per experiment configuration (`grid_id`).

| Column | Type | Description |
|--------|------|-------------|
| grid\_id | VARCHAR | Experiment grid identifier |
| algo | VARCHAR | Algorithm name (e.g. "memorypair") |
| dataset | VARCHAR | Dataset name (e.g. "synthetic") |
| delete\_ratio | DOUBLE | Percentage of delete events |
| eps\_total | DOUBLE | Total epsilon privacy budget |
| delta\_total | DOUBLE | Total delta privacy parameter |
| rho\_total | DOUBLE | Total rho privacy budget |
| lambda\_ | DOUBLE | Strong convexity parameter |
| target\_lambda | DOUBLE | Target strong convexity |
| target\_D | DOUBLE | Target diameter bound |
| target\_G | DOUBLE | Target gradient bound |
| target\_C | DOUBLE | Target curvature bound |
| target\_c | DOUBLE | Target curvature bound |
| strong\_convexity | BOOLEAN | Whether strong convexity is enforced |
| enable\_oracle | BOOLEAN | Whether oracle is enabled |
| regret\_comparator\_mode | VARCHAR | Comparator mode (e.g. "oracle") |
| drift\_mode | BOOLEAN | Whether drift adaptation is enabled |
| adaptive\_privacy | BOOLEAN | Whether adaptive privacy is enabled |
| adaptive\_geometry | BOOLEAN | Whether adaptive geometry is enabled |

---

### `analytics.fact_event`

Grain: 1 row per event.

**Primary Keys**
* `grid_id`, `seed`, `event_id` = natural event key

**Core Event Data**
* `event_type` (VARCHAR): Event category (calibrate, insert, delete)
* `event` (BIGINT): Event sequence number
* `op` (VARCHAR): Operation type
* `event_id` (BIGINT): Unique event identifier
* `sample_id` (VARCHAR): Sample identifier
* `segment_id` (BIGINT): Segment identifier

**Regret Metrics**
* `regret` (DOUBLE): Instantaneous regret
* `regret_increment` (DOUBLE): $\mathcal{L}_{\lambda}(\theta_{curr}) - \mathcal{L}_{\lambda}(\theta_{comp})$ on (x_t,y_t)
* `static_regret_increment` (DOUBLE): Static regret component
* `path_regret_increment` (DOUBLE): Path regret component
* `cum_regret` (DOUBLE): Cumulative regret
* `avg_regret` (DOUBLE): Average regret
* `regret_dynamic` (DOUBLE): Dynamic regret
* `regret_static_term` (DOUBLE): Static regret term
* `regret_path_term` (DOUBLE): Path regret term
* `noise_regret_increment` (DOUBLE): Noise-induced regret increment
* `noise_regret_cum` (DOUBLE): Cumulative noise regret
* `cum_regret_with_noise` (DOUBLE): Total cumulative regret including noise
* `avg_regret_with_noise` (DOUBLE): Average regret including noise

**Path Length & Energy Metrics**
* `P_T` (DOUBLE): Path length estimate
* `P_T_est` (DOUBLE): Path length estimate
* `P_T_true` (DOUBLE): True path length
* `PT_target_residual` (DOUBLE): Path length target residual
* `ST_running` (DOUBLE): Running AdaGrad energy
* `ST_target_residual` (DOUBLE): Energy target residual
* `S_scalar` (DOUBLE): Scalar energy term
* `delta_P` (DOUBLE): Path length delta

**Oracle & Comparator Metrics**
* `oracle_objective` (DOUBLE): Comparator's **regularized** loss for this event (ERM/static or rolling), used with `regret_increment` to reconstruct learner loss
* `oracle_w_norm` (DOUBLE): Oracle weight norm
* `oracle_refresh_step` (DOUBLE): Oracle refresh step
* `oracle_refreshes` (BIGINT): Number of oracle refreshes
* `oracle_stalled_count` (BIGINT): Number of oracle stalls
* `comparator_type` (VARCHAR): Comparator type (`static_oracle_erm_fullprefix`, `rolling_oracle_window{W}`, `zero_proxy`)
* `window_size` (BIGINT): Sliding window size

**Norms & Stepsize Metrics**
* `g_norm` (DOUBLE): Gradient norm
* `d_norm` (DOUBLE): Distance norm
* `x_norm` (DOUBLE): Input norm
* `w_star_norm` (DOUBLE): Optimal weight norm
* `eta_t` (DOUBLE): Current step size
* `base_eta_t` (DOUBLE): Base step size
* `lambda_est` (DOUBLE): Estimated strong convexity
* `lambda_raw` (DOUBLE): Raw lambda value
* `clip_applied` (BOOLEAN): Whether gradient clipping was applied

**Privacy & Unlearning Metrics**
* `m_capacity` (BIGINT): Deletion capacity
* `m_used` (BIGINT): Deletions used
* `deletion_capacity` (BIGINT): Total deletion capacity
* `rho_spent` (DOUBLE): Privacy budget spent
* `rho_remaining` (DOUBLE): Privacy budget remaining
* `rho_step` (DOUBLE): Privacy cost per step
* `eps_spent` (DOUBLE): Epsilon privacy budget spent
* `eps_remaining` (VARCHAR): Epsilon budget remaining
* `sigma_step` (DOUBLE): Noise scale for steps
* `sigma_delete` (DOUBLE): Noise scale for deletions
* `sigma_step_theory` (DOUBLE): Theoretical step noise scale
* `privacy_spend_running` (DOUBLE): Running privacy spend
* `sens_delete` (VARCHAR): Deletion sensitivity
* `capacity_remaining` (DOUBLE): Remaining capacity

**Drift Detection Metrics**
* `drift_flag` (BOOLEAN): Whether drift was detected
* `drift_detected` (BOOLEAN): Whether drift was detected
* `drift_episodes_count` (BIGINT): Number of drift episodes
* `drift_threshold` (DOUBLE): Drift detection threshold
* `drift_boost_remaining` (BIGINT): Remaining drift boost

**Strong Convexity & Stability**
* `sc_stable` (BIGINT): Strong convexity stability indicator
* `sc_active` (BOOLEAN): Whether strong convexity is active
* `pair_admitted` (BOOLEAN): Whether pair was admitted
* `pair_damped` (BOOLEAN): Whether pair was damped

**Theoretical Bounds & Targets**
* `G_hat` (DOUBLE): Empirical gradient bound
* `D_hat` (DOUBLE): Empirical diameter bound
* `c_hat` (DOUBLE): Empirical curvature bound
* `C_hat` (DOUBLE): Empirical curvature bound
* `theory_targets` (VARCHAR): JSON string of theoretical targets
* `eps_step_theory` (VARCHAR): Theoretical epsilon per step
* `delta_step_theory` (VARCHAR): Theoretical delta per step

**Stepsize Policy**
* `stepsize_policy` (VARCHAR): Stepsize policy name (e.g. "adagrad")
* `stepsize_params` (VARCHAR): JSON string of stepsize parameters

**Other Metrics**
* `acc` (DOUBLE): Accuracy
* `noise` (DOUBLE): Gaussian noise injected
* `accountant` (VARCHAR): Privacy accountant type
* `accountant_type` (VARCHAR): Privacy accountant type
* `delta_total` (DOUBLE): Total delta parameter
* `N_gamma` (DOUBLE): Gamma threshold count

---

### `analytics.v_run_summary`

Grain: 1 row per `(grid_id, seed)` with end-of-run stats.

| Column                          | Description              |
| ------------------------------- | ------------------------ |
| grid\_id                        | Experiment grid identifier |
| seed                            | RNG seed |
| run\_id                         | Composite run identifier |
| total\_events                   | Total number of events |
| insert\_events                  | Number of insert events |
| delete\_events                  | Number of delete events |
| avg\_regret                     | Average regret |
| min\_regret                     | Minimum regret |
| max\_regret                     | Maximum regret |
| avg\_cum\_regret                | Average cumulative regret |
| min\_cum\_regret                | Minimum cumulative regret |
| max\_cum\_regret                | Maximum cumulative regret |
| avg\_P\_T\_true                 | Average true path length |
| min\_P\_T\_true                 | Minimum true path length |
| max\_P\_T\_true                 | Maximum true path length |
| avg\_ST\_running                | Average running energy |
| min\_ST\_running                | Minimum running energy |
| max\_ST\_running                | Maximum running energy |
| avg\_rho\_spent                 | Average privacy budget spent |
| min\_rho\_spent                 | Minimum privacy budget spent |
| max\_rho\_spent                 | Maximum privacy budget spent |
| avg\_m\_used                    | Average deletions used |
| min\_m\_used                    | Minimum deletions used |
| max\_m\_used                    | Maximum deletions used |

---

### `analytics.v_events_with_params`

Grain: 1 row per event with parameter data joined.

This view combines `analytics.fact_event` with `analytics.dim_parameters` to provide all event-level data along with the corresponding experiment parameters.

---

## 2. Grid Configuration Parameters

Defined in `grids/*/params.json` files.

### Core Algorithm Parameters
| Param | Type | Meaning |
|-------|------|---------|
| `algo` | VARCHAR | Algorithm name (e.g. "memorypair") |
| `dataset` | VARCHAR | Dataset name (e.g. "synthetic") |
| `max_events` | BIGINT | Maximum number of events to process |
| `delete_ratio` | DOUBLE | Percentage of delete events (e.g. 5.0 = 5%) |

### Privacy Parameters
| Param | Type | Meaning |
|-------|------|---------|
| `accountant` | VARCHAR | Privacy accountant type (e.g. "zcdp") |
| `rho_total` | DOUBLE | Total rho privacy budget |
| `eps_total` | DOUBLE | Total epsilon privacy budget |
| `delta_total` | DOUBLE | Total delta privacy parameter |
| `delta_b` | DOUBLE | Failure probability for regret bounds |

### Strong Convexity Parameters
| Param | Type | Meaning |
|-------|------|---------|
| `lambda_` | DOUBLE | Strong convexity parameter |
| `target_lambda` | DOUBLE | Target strong convexity |
| `lambda_reg` | DOUBLE | L2 regularization parameter |
| `lambda_cap` | DOUBLE | Maximum lambda value |
| `lambda_floor` | DOUBLE | Minimum lambda value |
| `lambda_min_threshold` | DOUBLE | Minimum lambda threshold |
| `strong_convexity` | BOOLEAN | Whether strong convexity is enforced |

### Theoretical Bounds
| Param | Type | Meaning |
|-------|------|---------|
| `target_G` | DOUBLE | Target gradient bound |
| `target_D` | DOUBLE | Target diameter bound |
| `target_C` | DOUBLE | Target curvature bound |
| `target_c` | DOUBLE | Target curvature bound |
| `target_PT` | DOUBLE | Target path length |
| `target_ST` | DOUBLE | Target AdaGrad energy |

### Stepsize & Optimization
| Param | Type | Meaning |
|-------|------|---------|
| `eta_max` | DOUBLE | Maximum step size |
| `adagrad_eps` | DOUBLE | AdaGrad epsilon parameter |
| `D_bound` | DOUBLE | Diameter bound |
| `D_cap` | DOUBLE | Diameter cap |

### Memory & Capacity
| Param | Type | Meaning |
|-------|------|---------|
| `m_max` | BIGINT | L-BFGS memory size |
| `deletion_capacity` | BIGINT | Maximum deletions allowed |

### Drift Detection
| Param | Type | Meaning |
|-------|------|---------|
| `drift_mode` | BOOLEAN | Whether drift adaptation is enabled |
| `drift_threshold` | DOUBLE | Drift detection threshold |
| `drift_rate` | DOUBLE | Drift rate parameter |
| `drift_window` | BIGINT | Drift detection window size |
| `drift_kappa` | DOUBLE | Drift adaptation parameter |

### Oracle & Comparator
| Param | Type | Meaning |
|-------|------|---------|
| `enable_oracle` | BOOLEAN | Whether oracle is enabled |
| `regret_comparator_mode` | VARCHAR | Comparator mode (e.g. "oracle") |
| `comparator` | VARCHAR | Comparator type (e.g. "dynamic") |

### Adaptive Features
| Param | Type | Meaning |
|-------|------|---------|
| `adaptive_privacy` | BOOLEAN | Whether adaptive privacy is enabled |
| `adaptive_geometry` | BOOLEAN | Whether adaptive geometry is enabled |

### Other Parameters
| Param | Type | Meaning |
|-------|------|---------|
| `gamma_bar` | DOUBLE | Gamma bar threshold |
| `gamma_split` | DOUBLE | Gamma split threshold |
| `sens_calib` | DOUBLE | Sensitivity calibration parameter |
| `rotate_angle` | DOUBLE | Rotation angle for path construction |
| `relaxation_factor` | DOUBLE | Relaxation factor |
| `trim_quantile` | DOUBLE | Trim quantile for outlier handling |
| `quantile` | DOUBLE | Quantile parameter |

---

## 3. Theoretical Variables & Metrics

From the Memory Pair paper and sample calculations.

### Core Regret Metrics
| Symbol | Definition | Data Column | Example Value |
|--------|------------|-------------|---------------|
| $R_T$ | Cumulative regret | `cum_regret` | ~113.3 for T=1000 |
| $R_T^{dyn}$ | Dynamic regret vs drifting comparator | `regret_dynamic` | Bounded by $G^2/\lambda(1+\ln T) + GP_T$ |
| $R_T^{static}$ | Static regret vs fixed comparator | `regret_static_term` | $\frac{G^2}{\lambda}(1 + \ln T)$ ≈ 63.3 |
| $R_T^{path}$ | Pathwise regret component | `regret_path_term` | $G P_T$ ≈ 50 |

### Path Length & Energy
| Symbol | Definition | Data Column | Example Value |
|--------|------------|-------------|---------------|
| $P_T$ | Path length of comparator sequence | `P_T_true` | $\sum \|w^*_t - w^*_{t-1}\|$ ≈ 25 |
| $P_T^{est}$ | Estimated path length | `P_T_est` | Algorithm's estimate of $P_T$ |
| $S_T$ | AdaGrad-style cumulative squared gradient energy | `ST_running` | $\sum_{t=1}^T \|g_t\|^2$ |
| $\Delta P$ | Path length delta | `delta_P` | Change in path length |

### Privacy & Unlearning
| Symbol | Definition | Data Column | Example Value |
|--------|------------|-------------|---------------|
| $\rho_{tot}$ | Total rho privacy budget | `rho_total` | 1.0 |
| $\epsilon$ | Total epsilon privacy budget | `eps_total` | 1.0 |
| $\delta$ | Total delta privacy parameter | `delta_total` | 1e-05 |
| $\sigma_{step}$ | Noise scale for steps | `sigma_step` | Varies by privacy budget |
| $\sigma_{delete}$ | Noise scale for deletions | `sigma_delete` | Varies by privacy budget |
| $m$ | Deletion capacity | `m_capacity` | Maximum deletions allowed |

### Theoretical Bounds
| Symbol | Definition | Data Column | Example Value |
|--------|------------|-------------|---------------|
| $G$ | Gradient bound | `target_G`, `G_hat` | 2.0 |
| $D$ | Diameter bound | `target_D`, `D_hat` | 2.0 |
| $c$ | Curvature bound (lower) | `target_c`, `c_hat` | 0.05 |
| $C$ | Curvature bound (upper) | `target_C`, `C_hat` | 20.0 |
| $\lambda$ | Strong convexity parameter | `target_lambda`, `lambda_est` | 0.5 |

### Stepsize & Optimization
| Symbol | Definition | Data Column | Example Value |
|--------|------------|-------------|---------------|
| $\eta_t$ | Step size at time $t$ | `eta_t` | $D/\sqrt{S_t}$ for AdaGrad |
| $\eta_{base}$ | Base step size | `base_eta_t` | Base step size before adaptation |

### Sample Calculation Example

From the sample calculations, with $G=2.0$, $\lambda=0.5$, $T=1000$, $P_T=25$:

**Static regret term:**
$$\frac{G^2}{\lambda}(1 + \ln T) = \frac{4}{0.5} \times (1 + \ln 1000) \approx 63.3$$

**Pathwise regret term:**
$$G P_T = 2.0 \times 25 = 50$$

**Total dynamic regret bound:**
$$R_T^{dyn} \leq 63.3 + 50 = 113.3$$

**Average regret:**
$$\frac{R_T^{dyn}}{T} \leq \frac{113.3}{1000} \approx 0.113$$