# Deletion Capacity Experiment

This experiment measures how many deletion requests a learner can honour before retraining becomes necessary.  We compare the online Memory‑Pair implementation against two placeholder baselines.

The script `run_capacity.py` supports burst and trickle delete schedules over a Rotating‑MNIST stream provided by `data_loader`.

## Example

```bash
PYTHONPATH=.:data python run_capacity.py --algo memorypair --schedule burst --seed 42
```

After finishing the run a JSON file is written to `results/` and automatically committed with a message like:

```
EXP:del_capacity memorypair-burst <hash>
```
