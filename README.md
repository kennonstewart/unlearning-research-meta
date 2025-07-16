

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
