

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
├── README.md # this file
├── code
│   ├── memory_pair # contains the memory pair object definition
│   │   ├── README.md
│   │   ├── example_usage.py
│   │   ├── pyproject.toml
│   │   ├── src
│   │   │   ├── __init__.py
│   │   │   ├── event_logging
│   │   │   │   ├── __init__.py
│   │   │   │   └── config.yaml
│   │   │   ├── fogo.egg-info
│   │   │   │   ├── PKG-INFO
│   │   │   │   ├── SOURCES.txt
│   │   │   │   ├── dependency_links.txt
│   │   │   │   ├── requires.txt
│   │   │   │   └── top_level.txt
│   │   │   ├── l_bfgs.py
│   │   │   └── memory_pair.py
│   │   └── test_standalone.py
│   └── memory_pair_exp # contains legacy experiments for sublinear regret
│       ├── Memory Pair Paper.pdf
│       ├── README.md
│       ├── requirements.txt
│       ├── run_regret.py
│       └── src
│           ├── __init__.py
│           ├── algorithms.py
│           ├── data_streams.py
│           ├── l_bfgs.py
│           └── memory_pair.py
├── experiments
│   ├── deletion_capacity # experiment to test the deletion capacity
│   │   └── README.md
│   ├── post_deletion_accuracy # experiment to ensure accuracy following deletions
│   │   └── README.md
│   └── sublinear_regret # experiment testing the regret bounds for the algorithm
│       └── README.md
└── structure.txt
```
