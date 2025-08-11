# Research on Online Machine Unlearning

## Introduction

This repository contains the code and experiments for our research into the performance and characteristics of online machine unlearning algorithms. It serves as a central hub to ensure the full reproducibility of our findings.

This project investigates several key questions:

- Does the Memory-Pair learner achieve sub-linear cumulative regret on drifting data streams?
- What is the deletion capacity of privacy-preserving unlearning methods?
- How does model accuracy degrade as a function of sequential deletion operations?

## 📂 Repository Structure

This repository is organized as a unified codebase with shared components and independent experiments:

```
.
├── README.md
├── AGENTS.md                    # Meta-repository guide for AI agents
├── code/                        # Canonical, installable source code
│   ├── memory_pair/            # Memory-Pair algorithm implementation
│   │   └── src/
│   │       ├── memory_pair.py  # Main algorithm with state machine
│   │       ├── calibrator.py   # Theoretical constants estimation
│   │       ├── odometer.py     # Privacy budget tracking
│   │       ├── lbfgs.py        # L-BFGS optimization
│   │       └── metrics.py      # Performance metrics
│   ├── data_loader/            # Unified dataset loaders
│   │   ├── mnist.py           # MNIST and Rotating-MNIST streams
│   │   ├── cifar10.py         # CIFAR-10 streams
│   │   ├── covtype.py         # Forest CoverType dataset
│   │   ├── linear.py          # Synthetic linear data
│   │   ├── streams.py         # Stream utilities
│   │   └── utils.py           # Common utilities
│   └── baselines/             # Baseline algorithm implementations
├── experiments/               # Independent experimental studies
│   ├── deletion_capacity/     # Deletion capacity analysis
│   ├── post_deletion_accuracy/ # Model accuracy degradation
│   └── sublinear_regret/      # Regret analysis on drifting streams
└── paper/                     # Research paper materials
```

### Key Components

- **Memory-Pair Algorithm**: Implements a three-phase state machine (calibration, learning, interleaving) with privacy-preserving deletion capabilities
- **Data Loaders**: Unified, fail-safe dataset loaders for various machine learning benchmarks
- **Experiments**: Each subdirectory contains an independent study with its own `AGENTS.md` for specific instructions
- **Paper Materials**: LaTeX source and figures for the research paper in the `paper/` directory

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kennonstewart/unlearning-research-meta.git
   cd unlearning-research-meta
   ```

2. **Set up the Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the packages in editable mode:**
   ```bash
   pip install -e code/memory_pair
   pip install -e code/data_loader
   ```

### Verification

Run the data loader sanity check to ensure everything is working:
```bash
python code/data_loader/sanity_check.py
```

## 🧪 Running Experiments

Each experiment is self-contained with its own instructions. Navigate to an experiment directory and follow its `AGENTS.md`:

### Deletion Capacity Experiment
```bash
cd experiments/deletion_capacity
python run.py --algo memorypair --schedule burst --seed 42
```

### Sublinear Regret Experiment  
```bash
cd experiments/sublinear_regret
python run.py --dataset rotmnist --stream drift --T 10000 --seed 42
```

### Post-Deletion Accuracy Experiment
```bash
cd experiments/post_deletion_accuracy
python run_accuracy.py --dataset mnist --deletions 100 --seed 42
```

## 🔬 Key Features

### Memory-Pair Algorithm
- **Three-Phase State Machine**: Calibration → Learning → Interleaving
- **Automatic Calibration**: Estimates theoretical constants (G, D, c, C) from bootstrap data
- **Privacy-Preserving Deletions**: Uses differential privacy with regret-constrained optimization
- **Sample Complexity**: Automatically computes N* for optimal learning-to-prediction transition

### Privacy Odometer
- **Regret-Constrained Optimization**: Maximizes deletion capacity under regret bounds
- **Adaptive Noise Scaling**: Computes optimal Gaussian noise for each deletion
- **Budget Tracking**: Monitors ε and δ expenditure across deletions

## 🛠️ Repository Maintenance

For information about potential repository structure improvements and refactoring opportunities, see [`REFACTORING_OPPORTUNITIES.md`](./REFACTORING_OPPORTUNITIES.md). This document identifies ways to simplify the codebase organization and improve maintainability.

## ✍️ Development Guidelines

### Import Policy
Always use the canonical imports for consistency:

```python
# Algorithms
from code.memory_pair.src.memory_pair import MemoryPair
from code.memory_pair.src.odometer import PrivacyOdometer

# Data
from code.data_loader import get_rotating_mnist_stream
```

### Experiment Structure
Each experiment must:
- Include an `AGENTS.md` with specific instructions
- Accept `--seed` parameter for reproducibility  
- Auto-commit results with standardized commit messages
- Store outputs in `experiments/<name>/results/`

### Reproducibility
All scripts must be deterministic. Use the provided seeding utility:
```python
from code.data_loader.utils import set_global_seed
set_global_seed(args.seed)
```

## 📊 Results and Analysis

Results are automatically committed to version control with structured commit messages:
- Format: `EXP:<experiment> <algorithm>-<config> <short-hash>`
- Example: `EXP:deletion_capacity memorypair-burst a1b2c3d`

Each experiment directory contains:
- `results/`: JSON files with experimental outcomes
- `figs/`: Generated plots and visualizations  
- `runs/`: Detailed run logs and intermediate outputs

## 🤝 Contributing

1. **Check experiment-specific guidelines**: Read `experiments/<name>/AGENTS.md`
2. **Follow import conventions**: Use canonical paths from `code/`
3. **Ensure reproducibility**: All scripts must accept `--seed`
4. **Test thoroughly**: Run relevant sanity checks and unit tests
5. **Document changes**: Update relevant README files

## 🆕 Advanced Features (M7-M10)

### Unified Gamma Split (M9)
The deletion capacity experiments now support unified regret budget allocation:

```bash
# New unified approach (recommended)
python cli.py --gamma-bar 2.0 --gamma-split 0.7  # 70% for learning, 30% for privacy

# Legacy approach (backward compatible)
python cli.py --gamma-learn 1.4 --gamma-priv 0.6  # equivalent to above
```

### Enhanced Data Loaders

#### CovType with Online Standardization (M7)
```bash
python cli.py --dataset covtype --online-standardize --clip-k 3.0
```
Features:
- Welford's algorithm for online feature standardization
- k-sigma clipping with configurable threshold
- Label shift tracking with sliding window
- Diagnostics: `mean_l2`, `std_l2`, `clip_rate`, `segment_id`

#### Linear with Spectrum Control (M8)
```bash
python cli.py --dataset linear --eigs "1.0,0.5,0.1" --path-type rotating
```
Features:
- Configurable covariance matrix with eigenvalue control
- Controlled parameter path evolution
- Online strong convexity estimation
- Diagnostics: `lambda_est`, `P_T_true`

### Deletion Gating (M9)
Automatic gating prevents deletions that would violate constraints:
- **Regret Gate**: Blocks deletions that would exceed regret budget
- **Privacy Gate**: Blocks deletions when privacy budget is depleted
- Logs blocked operations with `blocked_reason`

### Evaluation Suite (M10)
Generate comprehensive analysis reports:

```bash
cd experiments/deletion_capacity
python report.py --datasets linear covtype mnist --seeds 5
```

Features:
- Live capacity estimates (`N_star_live`, `m_theory_live`)
- Regret decomposition (static, adaptive, path components)
- S_T gradient energy tracking
- Sensitivity bin analysis
- Drift overlay visualization

## 📚 References

This implementation follows the theoretical framework established in our research on online machine unlearning with differential privacy guarantees and regret minimization.

## 📂 Additional Files

### Documentation
- `IMPLEMENTATION_SUMMARY.md`: Technical overview of implementation details
- `MILESTONE5_SUMMARY.md`: Summary of recent milestone achievements

### Demo and Examples
- `milestone5_demo.py`: Demonstration script showcasing current capabilities
- `oracle_integration_example.py`: Example of oracle integration patterns

### Test Suite
The repository includes comprehensive tests for various components:
- `test_minimal_experiment.py`: Basic experiment functionality tests
- `test_integration.py`: Integration tests across components
- `test_zcdp_odometer.py`: Zero-concentrated differential privacy tests
- `test_enhanced_oracle.py`: Enhanced oracle functionality tests
- `test_milestone5_*.py`: Milestone-specific functionality tests
- `test_dynamic_comparator.py`: Dynamic comparison algorithm tests
- `test_flags_default_noop.py`: Configuration flag tests
- `test_schema.py`: Data schema validation tests
- `test_issue_requirements.py`: Requirement compliance tests

To run the test suite:
```bash
python -m pytest test_*.py -v
```

## 📝 License

[Add appropriate license information]

## 👥 Maintainers

- Primary maintainer: @kennonstewart
- Issues & discussions: GitHub Issues tab
- For urgent build failures, tag maintainers in your PR