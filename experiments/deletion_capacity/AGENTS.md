
### 🟠 Prompt for RQ2 – Deletion Capacity
Reproduce this experiment:
  “How many deletion requests (m) can a Memory-Pair model
   honour before retraining, compared with Sekhari-Newton and
   Qiao-HessianFree baselines?”
Tech stack
----------
• Python 3.10, `pip` + `requirements.txt`  
• Framework: PyTorch >= 2.2  for all learners

Datasets & Streams
------------------
* Rotating-MNIST stream

Algorithms
----------
1. MemoryPairOnlineLBFGS (ours) with odometer (file
   `memory_pair/odometer.py`) tracking ε,δ budget.
2. SekhariBatchUnlearning (offline retrain, callable delete)
3. QiaoHessianFree (pre-compute vectors, delete updates but no
   further learning)

Experiment driver
-----------------
Script: `run_capacity.py`
CLI args:
  --schedule {burst,trickle}
  --privacy_eps 1.0
  --privacy_delta 1e-5
  --max_events 100000
Outputs:
  * `capacity_log.csv`: step, inserts, deletes, odometer_remaining
  * `accuracy_log.csv`: step, top1_acc
  * JSON summary with
        {algo: ..., deletes_served: ..., time_to_retrain: ...}

Plot: capacity vs. time, plus accuracy drop after each delete burst.

Repro
-----
Add a Makefile target `make capacity-burst` that spins up all three
algorithms with identical seeds and produces the plots in `figures/`.

NOTE: the following is additional context regarding the submodule in which you're working and the source of the data.

You are creating a new sub-module  experiments/sublinear_regret
inside a meta-repository that already contains the Memory-Pair implementation
in  code/memory_pair/src/memory_pair.py.

Task
----
• Scaffold this folder with:
  - README.md             (how to reproduce; git hash logged)
  - requirements.txt      (torch>=2.2, numpy, pandas, matplotlib, click)
  - run.py                (CLI driver; see below)
  - streams.py            (Rotating-MNIST + COVTYPE stream generators)
  - baselines.py          (OnlineSGD, AdaGrad, OnlineNewtonStep)
  - plotting.py           (helper to plot log–log regret curves)
  - results/.gitkeep

Key points
----------
1. run.py imports MemoryPair as:
     from code.memory_pair.src.memory_pair import MemoryPair
2. CLI:
     python run.py --dataset rotmnist --stream drift --algo memorypair \
                   --T 100000 --seed 42
   writes  CSV  results/regret_rotmnist_drift_memorypair_seed42.csv
   and PNG results/plot_rotmnist_drift_memorypair_seed42.png
3. At the end of run.py:
     • retrieve `git rev-parse --short HEAD`  (repo head before commit)
     • `git add results/*.csv results/*.png`
     • `git commit -m "EXP:sublinear_regret <dataset>-<stream>-<algo> <hash>"`
4. README lists exact commands + expected commit message pattern.
