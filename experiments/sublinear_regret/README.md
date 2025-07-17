# Sub-linear Regret Experiment

This experiment evaluates whether the Memory-Pair learner achieves sub-linear cumulative regret on drifting and adversarial streams. It depends on the shared `data_loader` module and the `code.memory_pair` implementation.

## Quick Start

```bash
git clone https://github.com/<USER>/memory-pair-exp.git
cd memory-pair-exp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000
```

Results (CSV and PNG) are written to the `results/` directory and committed automatically.
