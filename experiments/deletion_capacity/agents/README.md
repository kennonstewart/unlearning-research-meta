# Grid Search for Deletion Capacity Experiments

This directory contains the grid search automation for deletion capacity experiments as specified in `AGENTS.md`.

## Quick Start

```bash
# Run with provided theory-first grid (synthetic-only, MemoryPair+zCDP)
python experiments/deletion_capacity/agents/grid_runner.py \
  --grid-file experiments/deletion_capacity/agents/grids.yaml \
  --parallel 4 \
  --seeds 5 \
  --base-out results/grid_$(date +%Y_%m_%d)

# Event-level output (one row per event)
python agents/grid_runner.py \
  --grid-file agents/grids.yaml \
  --seeds 3 \
  --output-granularity event \
  --base-out results/grid_$(date +%Y_%m_%d)_events

# Parallel execution with 4 processes
python agents/grid_runner.py \
  --grid-file agents/grids.yaml \
  --parallel 4 \
  --seeds 5 \
  --base-out results/grid_$(date +%Y_%m_%d)_p4

# Dry run to preview parameter combinations
python agents/grid_runner.py --grid-file agents/grids.yaml --dry-run
```

## Grid Configuration

The parameter grid is defined in `agents/grids.yaml`.

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
