# Sublinear Regret Experiment

This experiment evaluates whether the Memory-Pair online learner achieves sub-linear cumulative regret R_T = O(√T) on different streaming regimes. It compares the Memory-Pair algorithm against baseline methods across various data distributions and concept drift patterns.

## Overview

The sublinear regret experiment investigates the theoretical guarantees of the Memory-Pair algorithm by:

- **Testing Regret Bounds**: Verifying O(√T) cumulative regret on various data streams
- **Comparing Algorithms**: Memory-Pair vs SGD, AdaGrad, and Online Newton Step
- **Evaluating Drift Robustness**: Performance under IID, drifting, and adversarial conditions
- **Measuring Sample Efficiency**: Learning speed and adaptation capabilities

## Supported Datasets

- **Rotating-MNIST**: MNIST digits with rotation-based concept drift
- **COVTYPE**: UCI Forest CoverType dataset with label shift

## Algorithms Implemented

- **MemoryPair**: Our main method (single-pass online L-BFGS with odometer disabled)
- **OnlineSGD**: Stochastic gradient descent baseline
- **AdaGrad**: Adaptive gradient method baseline  
- **OnlineNewtonStep**: Convex optimization baseline

## Usage

### Basic Execution

```bash
cd experiments/sublinear_regret
python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42
```

### Command Line Options

- `--dataset`: Dataset choice (`rotmnist`, `covtype`)
- `--stream`: Stream type (`iid`, `drift`, `adv`)  
- `--algo`: Algorithm (`memorypair`, `sgd`, `adagrad`, `ons`)
- `--T`: Time horizon (default: 100000)
- `--seed`: Random seed for reproducibility

### Example Runs

```bash
# Test on rotating MNIST with drift
python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000

# Compare algorithms on adversarial stream
python run.py --dataset rotmnist --stream adv --algo sgd --T 50000
python run.py --dataset rotmnist --stream adv --algo memorypair --T 50000

# Evaluate on COVTYPE with IID sampling
python run.py --dataset covtype --stream iid --algo memorypair --T 20000
```

## Output and Results

Each experiment run generates:

- **CSV File**: `results/<dataset>_<stream>_<algo>.csv` with columns (step, regret)
- **Plot**: `results/<dataset>_<stream>_<algo>.png` showing log-log regret curve with √T guideline
- **Auto-commit**: Results committed with message `EXP:sublinear_regret <dataset>-<stream>-<algo> <hash>`

## Installation and Dependencies

This experiment requires the shared components:

```bash
git clone https://github.com/<USER>/memory-pair-exp.git
cd memory-pair-exp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The experiment imports:
- Data loaders from `code.data_loader`
- Memory-Pair algorithm from `code.memory_pair.src.memory_pair`

## Performance Analysis

The experiment validates:

1. **Theoretical Bounds**: Actual regret vs O(√T) theoretical bound
2. **Algorithm Comparison**: Relative performance across methods
3. **Drift Adaptation**: Speed of adaptation to concept changes
4. **Computational Efficiency**: Runtime and memory usage

Results are visualized with log-log plots showing cumulative regret over time, with reference lines for theoretical bounds.
