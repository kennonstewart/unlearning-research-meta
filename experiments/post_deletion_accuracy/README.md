# Post-Deletion Accuracy Experiment

This experiment tracks test accuracy degradation after repeated deletion operations to evaluate how well unlearning methods maintain model performance while forgetting specific data points. It compares the Memory-Pair algorithm against baseline unlearning methods.

## Overview

The post-deletion accuracy experiment investigates:

- **Accuracy Preservation**: How well models maintain performance after deletions
- **Deletion Robustness**: Stability of predictions under sequential unlearning
- **Method Comparison**: Memory-Pair vs retraining and influence-based methods
- **Scaling Analysis**: Performance degradation patterns with increasing deletions

## Research Question

*"Does test accuracy stay within 5% of a fresh-retrain model while honoring up to m⋆ deletions?"*

## Supported Datasets

- **CIFAR-10**: Image classification with torchvision loader
- **COVTYPE**: UCI Forest CoverType tabular classification

## Algorithms Implemented

- **MemoryPair**: Our main method with privacy-preserving unlearning
- **SekhariBatch**: Full retrain baseline for each deletion
- **QiaoHF**: Influence function-based deletion (no continued learning)

## Usage

### Basic Execution

```bash
cd experiments/post_deletion_accuracy
python run_accuracy.py --dataset cifar10 --algo memorypair --gamma 0.05 --seed 7
```

### Command Line Options

- `--dataset`: Dataset choice (`cifar10`, `covtype`)
- `--algo`: Algorithm (`memorypair`, `sekhari`, `qiao`)
- `--gamma`: Accuracy tolerance threshold (default: 0.05)
- `--seed`: Random seed for reproducibility

### Example Runs

```bash
# Test Memory-Pair on CIFAR-10
python run_accuracy.py --dataset cifar10 --algo memorypair --gamma 0.05

# Compare with retraining baseline
python run_accuracy.py --dataset cifar10 --algo sekhari --gamma 0.05

# Evaluate on tabular data
python run_accuracy.py --dataset covtype --algo memorypair --gamma 0.1
```

## Experimental Protocol

### Phase 1: Warm-up
- Stream samples until sample complexity threshold n_γ is detected
- Use online running-loss estimator to determine readiness
- Ensure stable baseline performance before deletions

### Phase 2: Alternating Cycles
- **Insert Phase**: 500 new samples added to model
- **Delete Phase**: 50 uniform random deletions from history
- Continue until privacy odometer ≤ 0 or baseline hits deletion limit

### Phase 3: Evaluation
- Track test accuracy throughout the experiment
- Compare against fresh-retrain model accuracy
- Monitor privacy budget consumption

## Output and Results

Each experiment generates:

- **JSON Results**: `results/accuracy_<algo>_<dataset>_<seed>.json` with detailed metrics
- **Accuracy Plot**: `figures/accuracy_vs_deletes_<dataset>.png` with m⋆ threshold marked
- **Auto-commit**: Results committed with message `EXP:post_del_acc <dataset>-<algo> <hash>`

### Key Metrics Logged

- `test_top1_acc`: Top-1 test accuracy after each deletion cycle
- `odometer_remaining`: Privacy budget remaining
- `cumulative_deletes`: Total deletions performed
- `retrain_events`: Number of full retraining operations (for baselines)
- `accuracy_drop`: Deviation from fresh-retrain baseline

## Dependencies

This experiment requires:

```bash
# Core dependencies
pip install torch torchvision scikit-learn numpy matplotlib

# Shared components
from code.data_loader import get_cifar10_stream, get_covtype_stream
from code.memory_pair.src.memory_pair import MemoryPair
```

## Performance Analysis

The experiment validates:

1. **Accuracy Bounds**: Whether test accuracy stays within γ tolerance
2. **Deletion Capacity**: Maximum deletions before accuracy violation
3. **Method Efficiency**: Computational cost vs accuracy trade-offs
4. **Privacy Cost**: ε/δ expenditure per deletion operation

Results include vertical dashed lines marking the theoretical deletion capacity m⋆ where the privacy odometer reaches zero.
