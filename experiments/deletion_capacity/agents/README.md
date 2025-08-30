# Grid Search for Deletion Capacity Experiments

This directory contains the grid search automation for deletion capacity experiments as specified in `AGENTS.md`.

## Quick Start

```bash
# Run with default grid
python agents/grid_runner.py --seeds 5

# Use custom grid file
python agents/grid_runner.py --grid-file my_grid.yaml --seeds 10

# Parallel execution with 4 processes
python agents/grid_runner.py --parallel 4 --seeds 5

# Dry run to see parameter combinations
python agents/grid_runner.py --dry-run
```

## Grid Configuration

The parameter grid is defined in `grids.yaml` (or custom file with `--grid-file`). 

### Default Grid Parameters (zCDP-only)

- **Learning ↔ Privacy split**: `gamma_bar` and `gamma_split` 
  - gamma_bar: [0.5, 1.0, 2.0], gamma_split: [0.3, 0.5, 0.7]
- **Calibrator conservativeness**: `quantile` 
  - 0.90, 0.95, 0.99
- **Delete workload**: `delete_ratio`
  - 1, 5, 10 (k inserts per delete)
- **zCDP budget**: `rho_total`
  - 0.5, 1.0, 2.0

Total combinations: 3×3×3×3×3 = 243

## Output Structure

```
results/grid_YYYY_MM_DD/
├── sweep/
│   ├── gamma_1.0-split_0.5_q0.95_k10_zcdp_rho1.0/
│   │   ├── seed_000.csv
│   │   ├── seed_001.csv
│   │   └── ...
│   ├── gamma_2.0-split_0.3_q0.90_k1_zcdp_rho0.5/
│   │   └── ...
│   └── all_runs.csv               # Aggregated results
```

## Schema

The aggregated `all_runs.csv` contains all per-event logs with additional grid metadata:

### Base Columns
- `event`, `op`, `regret`, `acc` - Standard event data
- `grid_id`, `seed` - Grid cell identifier and seed number
- `gamma_learn_grid`, `gamma_priv_grid`, `quantile_grid`, `delete_ratio_grid`, `accountant_grid` - Grid parameters

### Accountant-specific Columns
- **Legacy accountant**: `eps_spent`, `capacity_remaining`, `eps_step_theory`, `delta_step_theory`
- **RDP accountant**: `eps_converted`, `eps_remaining`, `delta_total`, `sens_*`

## Example Usage

```bash
# Full grid search with 3 seeds per combination
python agents/grid_runner.py \
    --grid-file grids.yaml \
    --seeds 3 \
    --parallel 8 \
    --base-out results/grid_$(date +%Y_%m_%d)

# Small test run
python agents/grid_runner.py \
    --grid-file test_grid.yaml \
    --seeds 2 \
    --base-out /tmp/test_results
```

The script automatically:
1. Generates all parameter combinations from the grid
2. Runs experiments for each combination across all seeds
3. Aggregates results into a master CSV with grid metadata
4. Validates the output schema
5. Reports completion statistics