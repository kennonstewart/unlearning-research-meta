# Experiment Runner for Theory-First Deletion Capacity Studies

This directory contains the unified experiment runner for deletion capacity experiments following the "theory-first" paradigm.

## Quick Start

### Using the Unified Runner (Recommended)

The new `run.py` script is the recommended way to run experiments with full `exp_engine` integration:

```bash
# Run a grid search from YAML
python experiment/run.py --grid-file grids/01_theory_grid.yaml --parallel 4 --seeds 5

# Run a single experiment with theory-first parameters
python experiment/run.py \
  --target-G 2.0 --target-D 2.0 --target-c 0.1 --target-C 10.0 \
  --target-lambda 0.5 --target-PT 25.0 --target-ST 50000.0 \
  --rho-total 1.0 --max-events 1000 --seeds 3

# Dry run to preview grid combinations
python experiment/run.py --grid-file grids/01_theory_grid.yaml --dry-run
```

### Legacy Runner (Deprecated)

The old `grid_runner.py` and `runner.py` scripts are deprecated and replaced by compatibility stubs.
For existing workflows, use:

```bash
# DEPRECATED - use run.py instead
python experiment/grid_runner.py \
  --grid-file grids/01_regret_decomposition.yaml \
  --parallel 4 \
  --seeds 5 \
  --base-out results/grid_$(date +%Y_%m_%d)
```

## Grid Configuration

Parameter grids are defined in YAML files in the `grids/` directory.

### Theory-First Grid Format (Recommended)

The new format emphasizes theory-first parameters that define the learning problem:

```yaml
matrix:
  # Theory-first stream targets (fundamental constants)
  target_G: [2.0]          # Gradient norm bound
  target_D: [2.0]          # Parameter/domain diameter
  target_c: [0.1]          # Min inverse-Hessian eigenvalue
  target_C: [10.0]         # Max inverse-Hessian eigenvalue
  target_lambda: [0.5]     # Strong convexity of loss
  target_PT: [25.0]        # Total path length over horizon
  target_ST: [50000.0]     # Cumulative squared gradient
  
  # Privacy parameters (grid search)
  rho_total: [0.5, 1.0, 2.0]
  
  # Execution parameters
  dim: [10]
  max_events: [1000]
  seeds: [1]
```

### Available Grid Files

1. **`grids/01_theory_grid.yaml`** (NEW - Recommended)
   - Theory-first parameter format
   - Direct integration with `exp_engine` for Parquet output
   - Content-addressed grid IDs for reproducibility

2. **`grids/test_minimal.yaml`** (NEW)
   - Minimal grid for testing and validation
   - Fast execution for CI/CD pipelines

3. **`grids/01_regret_decomposition.yaml`** (Legacy)
   - Original regret decomposition experiments
   - Uses legacy parameter format

### Legacy Grid Files (Deprecated)

The following grid files use the legacy format and are maintained for backward compatibility:
- `grids/02_privacy_sensitivity.yaml`
- `grids/03_deletion_capacity.yaml`

## Key Features

- **Theory-First Paradigm**: Experiments are defined by theoretical constants (G, D, c, C, λ, P_T, S_T) rather than implementation details
- **Content-Addressed Grid IDs**: Each parameter combination gets a unique, deterministic hash for reproducibility
- **Parquet-First Storage**: All results written to HIVE-partitioned Parquet via `exp_engine`
- **Parallel Execution**: Run multiple seeds in parallel with `--parallel N`
- **Dry Run Mode**: Preview grid combinations before running with `--dry-run`

## Architecture

The unified runner (`run.py`) directly calls:
- `get_theory_stream()` from `code/data_loader` for synthetic data generation
- `MemoryPair` from `code/memory_pair` for the learning algorithm
- `exp_engine.io.write_event_rows()` for Parquet output
- `exp_engine.cah.attach_grid_id()` for content-addressed hashing

This eliminates the fragmentation between `grid_runner.py`, `runner.py`, and `exp_engine`,
creating a single, clean execution path.

## Migration Guide

### From Old Grid Runner

**Old:**
```bash
python experiment/grid_runner.py --grid-file grids/my_grid.yaml --base-out results/
```

**New:**
```bash
python experiment/run.py --grid-file grids/my_grid.yaml --output-dir results_parquet/
```

### From Old Runner (Single Experiments)

**Old:**
```bash
python experiment/runner.py --gamma-bar 1.0 --eps-total 1.0 --max-events 1000
```

**New:**
```bash
python experiment/run.py \
  --target-G 2.0 --target-D 2.0 --target-c 0.1 --target-C 10.0 \
  --target-lambda 0.5 --target-PT 25.0 --target-ST 50000.0 \
  --rho-total 1.0 --max-events 1000
```

Note: The new runner requires theory-first parameters instead of gamma-based parameters.

## Backward Compatibility

The old `grid_runner.py` and `runner.py` files have been replaced with compatibility stubs
that import the necessary helper functions for existing tests. You will see deprecation
warnings when importing from these modules. Update your code to use `experiment/run.py`
for new work.

## Output Structure

### Parquet-First Output (Default)

The unified runner writes results directly to Parquet format using `exp_engine`:

```
results_parquet/
├── events/                          # Event-level data
│   └── grid_id=<hash>/
│       └── seed=<n>/
│           └── *.parquet
├── grids/                           # Grid parameters
│   └── grid_id=<hash>/
│       └── params.json
```

Each grid cell gets a unique content-addressed `grid_id` based on its parameters,
ensuring reproducibility and deduplication.

### Legacy CSV Output (Deprecated)

The old `grid_runner.py` script wrote CSV files to a different structure.
This format is maintained for backward compatibility with existing tests.
