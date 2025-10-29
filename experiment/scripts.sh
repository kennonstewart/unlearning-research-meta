# ---- Common knobs ----
DATE="$(date +%Y_%m_%d)"
OUT="results/$DATE"
PARALLEL=4
SEEDS=5
PY="python"  # or `uv run python` / `python3` if you prefer

# Optional: create the results root
mkdir -p "$OUT"

# Helper (optional): run one grid file with standard flags
run_grid () {
  local grid="$1"
  local name="$2"
  echo "▶ Running $name ($grid)"
  $PY grid_runner.py \
    --grid-file "$grid" \
    --parallel "$PARALLEL" \
    --seeds "$SEEDS" \
    --base-out "$OUT/$name" \
    --parquet-out "results_parquet"
}

# 1) Regret decomposition (static vs path/noise)
run_grid "experiment/grids/experiment_grids_99_sample_calc_single.yaml" "sample_calculation"

# ---- (Optional) Run them all in one go ----
# Comment any line above to skip, then:
# bash -c 'source ./run_all.sh'  # if you save this file as run_all.sh

# ---- (Optional) fire and forget single-shot examples (no helper) ----
# $PY grid_runner.py --grid-file grids/01_regret_decomposition.yaml \
#   --parallel "$PARALLEL" --seeds "$SEEDS" --base-out "$OUT/regret_decomposition"

# ---- Notes ----
# • Adjust PARALLEL and SEEDS as needed; OUT is namespaced by date for hygiene.
# • If you prefer per-seed folders, add:  --output-granularity seed
# • If your configs live elsewhere, tweak the paths accordingly.
