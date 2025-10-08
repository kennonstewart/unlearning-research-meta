#!/bin/bash
#SBATCH --job-name=ksstewar-mp-sweep
#SBATCH --account=<YOUR-ACCOUNT>         # e.g., coe-research / umr0xxx
#SBATCH --partition=standard              # debug for smoke tests; largemem if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                 # threads inside each task
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-<GRID_ROWS-1>           # <-- set to number of grid rows (0-indexed)
#SBATCH --output=/scratch/%u/mp_%x_%A_%a.out
#SBATCH --error=/scratch/%u/mp_%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ksstewar@umich.edu

# --- Environment ---
module purge
module load python/3.11
git clone https://github.com/kennonstewart/unlearning-research-meta.git memorypair   
cd memorypair

# --- I/O roots (scratch is faster; copy summaries back later) ---
BASE_OUT="/scratch/$USER/memorypair_$(date +%Y%m%d_%H%M)"
PARQUET_OUT="$BASE_OUT/parquet"

mkdir -p "$BASE_OUT" "$PARQUET_OUT"

# --- Files (paths relative to repo root you rsync’d to Great Lakes) ---
GRID_FILE="experiment/configs/grids.yaml" # grid file (relative to repo root)
SEEDS=3                                   # replicas per config (adjust below)
PARALLEL=8                                # <= --cpus-per-task

# --- Optional: pin threading env for BLAS/OpenMP libs ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

# --- One array index == one grid row ---
ROW="$SLURM_ARRAY_TASK_ID"

echo "[INFO] Starting row=$ROW on job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
python grid_runner.py \
  --grid-file "$GRID_FILE" \
  --row "$ROW" \
  --parallel "$PARALLEL" \
  --seeds "$SEEDS" \
  --base-out "$BASE_OUT" \
  --parquet-out "$PARQUET_OUT" \
  --parquet-write-events \
  --no-legacy-csv

EXIT=$?
echo "[INFO] Finished row=$ROW with exit code $EXIT"
exit $EXIT
