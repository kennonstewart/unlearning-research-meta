# Post-Deletion Accuracy Experiment

This study tracks test accuracy after repeated delete operations.  CIFAR‑10 and COVTYPE streams are loaded via `data_loader`.  The Memory‑Pair algorithm is compared to two placeholder baselines.

Running `run_accuracy.py` logs metrics to a JSON file in `results/` and commits the file on completion.

## Example

```bash
PYTHONPATH=.:data python run_accuracy.py --dataset cifar10 --algo memorypair --gamma 0.05 --seed 7
```

This produces a commit with message

```
EXP:post_del_acc cifar10-memorypair <hash>
```
