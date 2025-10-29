#!/bin/bash
#SBATCH --job-name=unlearning_exp
#SBATCH --account=your_account_name
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

# Load necessary modules
module load python/3.9.6

# Run the experiment script
python experiment/run.py --grid-file grids/01_theory_grid.yaml --parallel 4