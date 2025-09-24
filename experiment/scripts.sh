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
run_grid "grids/1a_dynamic_regret_decomposition.yaml" "dynamic_regret_decomposition"

# 1a) Regret decomposition (dynamic vs path/noise)
run_grid "grids/1b_static_regret_decomposition.yaml" "static_regret_decomposition"

# 2) Stepsize schedule / policy ablation
# run_grid "grids/02_strong_convexity.yaml" "stepsize_schedules"

# 3) Strong convexity & sensitivity sweep (λ, G, D, curvature)
# run_grid "grids/03_path_length_sensitivity.yaml" "path_length_sensitivity"

# 4) Path length vs energy (P_T vs S_T) analysis
#run_grid "grids/04_stepsize_ablation.yaml" "stepsize_ablation"

# 5) Drift adaptation on/off and thresholds
# run_grid "grids/05_privacy_vs_capacity.yaml" "privacy_vs_capacity"

# 6) Oracle/comparator behavior (type, window)s
# run_grid "grids/06_regret_gate.yaml" "regret_gate"

# 7) Privacy odometer trade-offs (ρ_total, δ, m, σ)
# run_grid "grids/07_lbfgs_stability.yaml" "lbfgs_stability"

# 8) Deletion regret gate (γ_delete etc.)
# run_grid "grids/08_deletion_regret_gate.yaml" "regret_gate"

# 9) Noise ablation (σ_step vs theory, sens/delete)
# run_grid "grids/09_noise_ablation.yaml" "noise_ablation"

# 10) Accountant comparison (e.g., zCDP vs RDP) — if applicable
# run_grid "grids/10_accountant_comparison.yaml" "accountant_comparison"

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
