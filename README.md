

# Research on Online Machine Unlearning

## Introduction

This repository contains the code and experiments for our research into the performance and characteristics of online machine unlearning algorithms. It serves as a central hub to ensure the full reproducibility of our findings.

This project investigates several key questions:

Does the Memory-Pair learner achieve sub-linear cumulative regret on drifting data streams?

What is the deletion capacity of privacy-preserving unlearning methods?

How does model accuracy degrade as a function of sequential deletion operations?

## 📂 Repository Structure
This is a meta-repository that organizes multiple components using git submodule. Note that the directory structure within the submodule may have changed from the structure below.

```
.
├── README.md
├── code
│   ├── memory_pair
│   └── memory_pair_exp
├── data
│   ├── README.md
│   └── data_loader
│       ├── README.md
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-312.pyc
│       │   ├── cifar10.cpython-312.pyc
│       │   ├── covtype.cpython-312.pyc
│       │   ├── mnist.cpython-312.pyc
│       │   ├── streams.cpython-312.pyc
│       │   └── utils.cpython-312.pyc
│       ├── cifar10.py
│       ├── covtype.py
│       ├── data_loader.egg-info
│       │   ├── PKG-INFO
│       │   ├── SOURCES.txt
│       │   ├── dependency_links.txt
│       │   ├── requires.txt
│       │   └── top_level.txt
│       ├── mnist.py
│       ├── pyproject.toml
│       ├── requirements.txt
│       ├── sanity_check.py
│       ├── streams.py
│       └── utils.py
├── experiments
│   ├── deletion_capacity
│   ├── post_deletion_accuracy
│   └── sublinear_regret
└── structure.txt
```

## 🚀 Getting Started
Follow these steps to set up the project environment.

### Clone the Repository

Clone this meta-repository using the --recurse-submodules flag to automatically fetch all the component code.

```bash
git clone --recurse-submodules https://github.com/your-username/unlearning-research-meta.git
cd unlearning-research-meta
Set up the Python Environment
```

Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
Install Dependencies
```

Install all required packages from the top-level `requirements.txt`. Then, install our local fogo package in editable mode so it can be used by the experiment modules.

```bash
pip install -r requirements.txt
pip install -e code/fogo
```

🧪 Running Experiments
To run an experiment, navigate to its directory and execute the main script. For example, to run the sublinear regret experiment:

```bash
cd code/sublinear_regret_experiment
python run_regret.py --dataset rotmnist --stream drift --T 10000
```

## ✍️ Contribution Workflow
To make changes to a specific component (like the fogo algorithm):
Navigate to the submodule directory: `cd code/fogo`
Make, commit, and push your changes within that directory.
Navigate back to the meta-repo root: `cd ../..`

Commit the submodule pointer update: `git add code/fogo and git commit -m "docs: Update Fogo component"`. This final commit registers the new version of the submodule with the main project.

## 📜 Citation
If you use this work, please cite:

Code snippet
@inproceedings{yourname2025unlearning,
  title={Online Machine Unlearning: Regret, Capacity, and Accuracy},
  author={Your Name and Collaborator Name},
  booktitle={Conference on Neural Information Processing Systems},
  year={2025}
}