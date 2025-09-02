# Deletion Capacity Experiment

**Scope**: MemoryPair + zCDP-only with synthetic data streams (theory-first preferred).

This experiment measures how many deletion requests a privacy-preserving learner can process while maintaining differential privacy guarantees and staying within regret bounds. It implements a three-phase calibrated state machine for the Memory-Pair algorithm with automatic capacity optimization.

## Overview

The deletion capacity experiment investigates the fundamental trade-off between privacy, utility, and deletion capacity in online machine unlearning. Using a regret-constrained optimization framework, the experiment determines the maximum number of deletions that can be processed while maintaining:

- **Zero-Concentrated Differential Privacy**: zCDP guarantees across all deletion operations
- **Regret Bounds**: Total regret/T ≤ γ constraint for competitive performance
- **Theoretical Soundness**: Calibrated constants based on actual algorithm behavior

## Key Features

### Three-Phase State Machine

1. **Calibration Phase**: Bootstrap run to estimate theoretical constants
   - **G**: Gradient norm upper bound (quantile-based with outlier handling)
   - **D**: Hypothesis diameter from parameter trajectory
   - **c, C**: L-BFGS curvature bounds with fallback defaults
   - **N\***: Sample complexity computed as ⌈(G·D·√(cC)/γ)²⌉

2. **Learning Phase**: Insert-only operations until ready to predict
   - Accumulates N\* samples before allowing predictions
   - Automatic transition to interleaving phase

3. **Interleaving Phase**: Full insert/delete operations with privacy accounting
   - zCDP accountant enforces capacity limits
   - Gaussian noise scaled per deletion operation

### zCDP Accountant with Joint Optimization

The zCDP accountant uses joint m-σ optimization to solve:

```
maximize m
subject to: 
1. Privacy constraint: m · ρ_step ≤ ρ_total with ρ_step = Δ²/(2σ²)
2. Regret constraint: (R_ins + R_del(m, σ))/T ≤ γ
```

Where:
- **R_ins = G·D·√(c·C·T)**: Insertion regret bound
- **R_del(m, σ) = m·(G/λ)·σ·√(2ln(1/δ_B))**: Deletion regret bound
- **m**: Deletion capacity (number of deletions allowed)
- **σ**: Gaussian noise standard deviation

The solution provides optimal deletion capacity and noise scale.

### Grid Search with Output Granularity

For systematic parameter sweeps, use the grid runner with configurable output granularity:

```bash
# Default seed-level output (one row per seed)
python agents/grid_runner.py \
    --grid-file grids.yaml \
    --seeds 10 \
    --parallel 8 \
    --base-out results/grid_$(date +%Y_%m_%d) \
    --output-granularity seed

# Event-level output (one row per insert/delete operation)  
python agents/grid_runner.py \
    --grid-file grids.yaml \
    --seeds 10 \
    --parallel 8 \
    --base-out results/grid_$(date +%Y_%m_%d) \
    --output-granularity event

# Aggregate output (one row per parameter combination)
python agents/grid_runner.py \
    --grid-file grids.yaml \
    --seeds 10 \
    --parallel 8 \
    --base-out results/grid_$(date +%Y_%m_%d) \
    --output-granularity aggregate
```

#### Output Granularity Modes

| Mode | Output Files | Schema | Use Case |
|------|-------------|---------|----------|
| `seed` | `runs/<grid_id>/seed_<seed>.csv` | One row per seed: avg_regret_empirical, N_star_emp, m_emp + mandatory fields | Statistical analysis, plotting trends |
| `event` | `runs/<grid_id>/seed_<seed>_events.csv` | One row per event: event, event_type, op, regret, acc + mandatory fields | Detailed debugging, trajectory analysis |
| `aggregate` | `runs/<grid_id>/aggregate.csv` | One row per grid-id: mean/median statistics across seeds + mandatory fields | High-level comparisons, LaTeX tables |

**Mandatory fields in all modes:** `G_hat`, `D_hat`, `sigma_step_theory` - enables recomputation of N★ and m for theoretical validation.

### Command Line Options

- `--algo`: Algorithm choice (only `memorypair` supported - MemoryPair with zCDP)
- `--schedule`: Deletion pattern (`burst`, `trickle`, `uniform`)
- `--dataset`: Data stream (synthetic-only, theory-first preferred when targets provided)
- `--T`: Total time horizon (default: 10000)
- `--gamma`: Regret constraint (default: 0.1)
- `--eps_total`: Total DP epsilon budget (default: 1.0)
- `--delta_total`: Total DP delta budget (default: 1e-5)
- `--bootstrap_steps`: Calibration phase length (default: 500)
- `--seed`: Random seed for reproducibility


## Implementation Details

### Calibrator Helper

The `Calibrator` class tracks algorithm behavior during bootstrap:

```python
from code.memory_pair.src.calibrator import Calibrator

calibrator = Calibrator(quantile=0.95)  # Clip top 5% of gradients

# During calibration phase
for x, y in bootstrap_data:
    model.calibrate_step(x, y)  # Logs to calibrator automatically

# Finalize and get statistics
stats = calibrator.finalize(gamma=0.1, model=model)
# Returns: {"G": ..., "D": ..., "c": ..., "C": ..., "N_star": ...}
```

### Enhanced MemoryPair API

```python
from code.memory_pair.src.memory_pair import MemoryPair, Phase
from code.memory_pair.src.accountant.zcdp import Adapter

# Initialize with zCDP privacy parameters (preferred approach)
accountant = Adapter(rho_total=1.0, delta_total=1e-5, gamma=0.1)
model = MemoryPair(dim=10, odometer=accountant.odometer)

# Phase 1: Calibration
assert model.phase == Phase.CALIBRATION
for x, y in bootstrap_data:
    model.calibrate_step(x, y)

# Phase 2: Finalize and transition to learning
model.finalize_calibration(gamma=0.1)
assert model.phase == Phase.LEARNING

# Phase 3: Learning until ready for deletions
while not model.can_predict:
    model.insert(x, y)  # Auto-transitions to INTERLEAVING at N*
    
# Phase 4: Interleaving with deletions
assert model.phase == Phase.INTERLEAVING
if model.odometer.ready_to_delete:
    model.delete(x, y)  # Privacy-preserving deletion
```

**Note**: The system is now zCDP-only. All privacy accounting uses zero-Concentrated Differential Privacy (ρ) with conversion to (ε,δ) for reporting.

**Example**: Create a MemoryPair model with zCDP accountant:

```python
from memory_pair.src.accountant import get_adapter
from memory_pair import MemoryPair

# Create zCDP accountant
accountant = get_adapter("zcdp", 
    rho_total=1.0, 
    delta_total=1e-5, 
    gamma=0.5, 
    lambda_=0.1
)

# Create model
model = MemoryPair(dim=20, accountant=accountant)
```

For notation and symbol definitions, see [docs/symbols.md](../../docs/symbols.md).

### Results and Metrics

Each experiment run generates comprehensive output:

```json
{
  "experiment": "deletion_capacity",
  "algorithm": "memorypair",
  "schedule": "burst", 
  "calibration": {
    "G": 2.34,
    "D": 1.87,
    "c": 0.8,
    "C": 1.2,
    "N_star": 156
  },
  "privacy": {
    "eps_total": 1.0,
    "delta_total": 1e-5,
    "deletion_capacity": 47,
    "eps_step": 0.0213,
    "delta_step": 2.13e-7,
    "noise_scale": 0.156
  },
  "performance": {
    "total_regret": 892.4,
    "regret_bound": 1000.0,
    "regret_ratio": 0.8924,
    "deletions_completed": 47,
    "accuracy_final": 0.923
  },
  "runtime": {
    "calibration_time": 12.3,
    "learning_time": 45.6,
    "interleaving_time": 78.9,
    "total_time": 136.8
  }
}
```

## Experimental Schedules

### Burst Schedule
- Front-loads deletions early in the stream
- Tests capacity under high deletion pressure
- Evaluates recovery after deletion burst

### Trickle Schedule  
- Evenly distributes deletions throughout stream
- Tests steady-state deletion performance
- Measures sustained accuracy under continuous unlearning

### Uniform Schedule
- Random deletion timing with fixed rate
- Tests robustness to deletion timing uncertainty
- Evaluates average-case performance

## Theoretical Validation

The experiment validates theoretical predictions:

1. **Capacity Formula**: Compare computed m with empirical deletion limits
2. **Regret Bounds**: Verify actual regret ≤ theoretical bounds  
3. **Privacy Guarantees**: Validate DP parameters through composition
4. **Sample Complexity**: Confirm N* provides sufficient learning


