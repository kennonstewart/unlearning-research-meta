# Grid Search for Deletion Capacity Experiments

This directory contains the grid search automation for deletion capacity experiments as specified in `AGENTS.md`.

## Quick Start

```bash
# When running the command from the experiment directory
python grid_runner.py \
  --grid-file grids/01_regret_decomposition.yaml \
  --parallel 4 \
  --seeds 5 \
  --base-out results/grid_$(date +%Y_%m_%d)

# When running the grid configuration for the pathwise regret experiment.
python grid_runner.py \ 
  --grid-file configs/grids.regret-decomposition.yaml \
  --parallel 4 \
  --seeds 3 \ 
  --base-out results/grid_$(date +%Y_%m_%d)

# Using different grid configurations
python experiment/grid_runner.py \
  --grid-file experiment/grids/02_privacy_sensitivity.yaml \
  --seeds 3 \
  --output-granularity seed \
  --base-out experiment/results/privacy_sensitivity_$(date +%Y_%m_%d)

python experiment/grid_runner.py \
  --grid-file experiment/grids/03_deletion_capacity.yaml \
  --seeds 3 \
  --output-granularity seed \
  --base-out experiment/results/deletion_capacity_$(date +%Y_%m_%d)

python agents/grid_runner.py --grid-file agents/grids.yaml --dry-run
```

## Grid Configuration

The parameter grids are defined in the `grids/` directory. Multiple grid files are available for different experiment types:

### Available Grid Files

1. **`grids/01_regret_decomposition.yaml`** (default)
   - Focuses on regret decomposition experiments
   - Enables oracle functionality for detailed regret analysis
   - Covers core regret controls, privacy parameters, and theory-first stream targets
   - ~16K combinations (limited to 50 for focused analysis)

2. **`grids/02_privacy_sensitivity.yaml`**
   - Analyzes sensitivity to privacy budget variations
   - Tests different epsilon, delta, and rho_total values
   - Fixed regret budget with varying privacy allocations
   - ~100 combinations (limited for focused analysis)

3. **`grids/03_deletion_capacity.yaml`**
   - Focuses on deletion capacity under different parameter regimes
   - Tests various deletion ratios and regret budget splits
   - Analyzes theory parameter impact on capacity
   - ~150 combinations (limited for focused analysis)

### Legacy Grid File

The original `configs/grids.yaml` is still available for backward compatibility.

Key notes for this synthetic-only setup:

- Algorithm/accountant: `algo: memorypair`, `accountant: zcdp`.
- Theory-first data stream: any presence of `target_*` values routes the runner to the theory-first synthetic stream.
- Strong convexity: enable via `strong_convexity: true` and set `lambda_reg > 0` (in the grid).
- Gamma pairing: `gamma_bar` and `gamma_split` are paired by index (not crossed). If you want more splits for the same gamma_bar, broadcast one list, e.g. `gamma_bar: [1.0]`, `gamma_split: [0.5, 0.6, 0.7]`.

### Included Grid (agents/grids.yaml)

- Privacy: `rho_total` in `[0.5, 1.0, 2.0]`
- Delete rate: `delete_ratio` in `[1, 5, 10]`
- Regret split pairs: `(gamma_bar, gamma_split)` in `[(0.6,0.5), (1.0,0.6), (1.5,0.7)]`
- Drift regimes via `target_PT` in `[100, 250]` (with `path_style: rotating`)
- Fixed theory constants: `target_G=2.0, target_D=2.0, target_c=0.1, target_C=10.0, target_lambda=0.05, target_ST=40000`
- Strong-convexity regularization: `lambda_reg` in `[0.01, 0.05]`

This yields a compact sweep that exercises the core levers without exploding combinations.

## Output Structure

Results are written under the chosen `--base-out` directory. See the main README for schema details. The `grid_id` encodes key parameters, including theory-first targets and zCDP settings, for easy filtering in downstream analysis.

## Parquet-First Mode

- Write Parquet seed/event logs while skipping legacy CSV:
  - `--parquet-out results_parquet --parquet-write-events --no-legacy-csv`
  - Seed summaries are always saved to Parquet; event logs are saved when `--parquet-write-events` is set.
- Aggregation reads directly from `results_parquet` and materializes `all_runs.csv` for plot compatibility and `all_runs.parquet` for Parquet-native workflows.
  - DuckDB is used under the hood; install with `pip install duckdb`.
  - If Parquet aggregation fails, the runner falls back to legacy CSV aggregation.
