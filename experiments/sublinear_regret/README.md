# Sub-linear Regret Experiment

This experiment evaluates whether the Memory‑Pair online learner achieves sub‑linear cumulative regret on different streaming regimes.  Streams are generated via the shared `data_loader` package and algorithms are implemented with the `code.memory_pair` library.

The script logs cumulative regret to CSV and produces a log–log plot.  At the end of the run the results are committed with a message of the form:

```
EXP:sublinear_regret <dataset>-<stream>-<algo> <hash>
```
where `<hash>` is the short git commit hash prior to adding the results.
This experiment evaluates whether the Memory-Pair learner achieves sub-linear cumulative regret on drifting and adversarial streams. It depends on the shared `data_loader` module and the `code.memory_pair` implementation.

## Quick Start

```bash
git clone https://github.com/<USER>/memory-pair-exp.git
cd memory-pair-exp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Example run on Rotating MNIST with drift
PYTHONPATH=.:data python run.py --dataset rotmnist --stream drift \
  --algo memorypair --T 100000 --seed 42
```

The command above generates `results/rotmnist_drift_memorypair.csv` and
`results/rotmnist_drift_memorypair.png`.  After completion the script runs
`git commit` with the message `EXP:sublinear_regret rotmnist-drift-memorypair <hash>`.
python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000
```

Results (CSV and PNG) are written to the `results/` directory and committed automatically.
