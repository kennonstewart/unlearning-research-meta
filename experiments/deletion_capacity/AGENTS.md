# AGENTS.md — Experiment 2 ("Deletion-Capacity")

> **Role:** This file instructs a Code Agent (e.g. GitHub Copilot / Codex) how to generate the *second* empirical experiment that measures deletion-capacity for the online-L-BFGS **MemoryPair** learner that already exists in this repo.

---

## 1  Objective

*Measure how many deletions the learner can perform before it exceeds its certified capacity `m`, and what happens to utility & privacy when the limit is crossed.*  
The experiment must run end-to-end from a single CLI entry point and emit: logs, CSV traces, JSON summary, and publication-ready figures.

---

## 2  Resources you MUST use

| Purpose | Module / Function | Import path |
|---------|------------------|-------------|
| **Streaming data** | Rotating-MNIST & synthetic linear streams | `from data_loader import get_rotating_mnist_stream, get_synthetic_linear_stream` |
| **Learner under test** | Online L-BFGS memory pair | `from code.memory_pair.src.memory_pair import MemoryPair` |
| **Baselines** | Sekhari Newton-Step, Qiao Hessian-Free | `from code.baselines import SekhariBatchUnlearning, QiaoHessianFree` |
| **Privacy budget** | Odometer (already parameterised) | `from code.memory_pair.src.odometer import PrivacyOdometer` |
| **Plotting** | Matplotlib only (no seaborn) | `import matplotlib.pyplot as plt` |

Do **not** add external ML libraries; use `numpy` only.

---

## 3  Experiment script skeleton (`experiments/deletion_capacity/exp2_capacity.py`)

1. **CLI flags** (use `click`):

```

\--dataset         \[rot-mnist | synthetic]   (default: rot-mnist)
\--delete-ratio    FLOAT   # k : 1 insert\:delete     (default: 10)
\--eps-per-delete  FLOAT   # fixed ε per delete      (default: 0.02)
\--max-events      INT     # total stream length     (default: 100\_000)
\--seeds           INT     # number of random seeds  (default: 10)
\--out-dir         PATH    # artefact folder         (default: results/exp2)

````

2. **Data stream factory**  
*rot-mnist*: `get_rotating_mnist_stream(mode="iid", seed=seed)`  
*synthetic*: `get_synthetic_linear_stream(dim=20, seed=seed)` (implement if missing).

3. **Model factory** chooses one of:

```python
StreamNewtonMemoryPair(dim, odometer=PrivacyOdometer(eps_per_delete=ε))
SekhariBatchUnlearning(dim)
QiaoHessianFree(dim)
````

4. **Phases**

```python
# warm-up until sample-complexity N_star 
# workload:  k inserts  → 1 delete, repeat
# continue until odometer.consume() raises OR max_events reached
```

5. **Logging**

* Per-event dict: timestamp, op-type, regret, acc, ε\_spent, capacity\_remaining.
* Dump to `runs/{seed}_{algo}.csv`.
* Summary JSON (mean ± 95 % CI) in `results/`.

6. **Figures**

Generate at least **F-1** (capacity curve) and **F-2** (regret) with matplotlib and save to `figs/`.

---

## 4  Directory / file layout

```
experiments/deletion_capacity/
│
├─ run.py                  ← main driver
├─ plots.py                ← helper to build all figures from CSV logs
├─ metrics.py              ← regret & utility helpers
├─ configs/                ← optional YAMLs
└─ runs/  figs/  results/  ← auto-created
```

Keep all new helper code inside this folder.

---

## 5  Acceptance checks

1. Loop terminates because either `ε-budget exceeded` **or** `events_processed == max_events`.
2. `summary.json` contains: `deletes`, `inserts`, `eps_spent`, `final_regret`.
3. `figs/capacity_curve.pdf` exists and > 10 kB.

---

## 6  Tips for the Agent

* Re-use error-handling patterns from `run_capacity.py`.
* Use `numpy.random.default_rng(seed)` for reproducible streams.
* Collect logs in memory; write once at end to avoid I/O bottlenecks.
* Plot after the run; one process per seed to stay < 8 GB RAM.

---

## 7  Example invocation

```bash
python experiments/deletion_capacity/run.py \\
       --dataset rot-mnist --delete-ratio 10
```

This should create CSVs in `runs/`, a summary JSON in `results/`, and at least the capacity & regret figures in `figs/`.
