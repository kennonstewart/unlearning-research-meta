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
   pip install -e exp_engine
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


## ✍️ Development Guidelines

### Reproducibility
All scripts must be deterministic. Use the provided seeding utility:
```python
from code.data_loader.utils import set_global_seed
set_global_seed(args.seed)
```
