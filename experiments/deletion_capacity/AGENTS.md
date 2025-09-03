**AGENTS.md**

---

## Purpose

Automate a **grid-search** over deletion-capacity experiments.
For every combination of tuning parameters, the agent must

1. **Patch** `config.py` (or instantiate `Config` via `from_cli_args`) with the new values.
2. **Run** `run.py` once per seed.
3. **Collect** the per-seed event data produced by the experiment.
4. **Write** all event data to HIVE-partitioned Parquet under `results_parquet/events/`.

The agent writes event-level Parquet files only. No CSV output or seed-level aggregation.

---

## Parameter grid (edit as needed)

| Group                           | Parameter                   | Values to sweep                           |
| ------------------------------- | --------------------------- | ----------------------------------------- |
| **Learning ↔ Privacy split**    | `gamma_learn`, `gamma_priv` | `(0.9,0.1) (0.7,0.3) (0.5,0.5) (0.3,0.7)` |
| **Calibrator conservativeness** | `quantile`                  | `0.90 0.95 0.99`                          |
| **Delete workload**             | `delete_ratio`              | `1 5 10`                                  |
| **Accountant type**             | `accountant`                | `"default" "rdp"`                         |
| *(optional)*                    | `eps_total`                 | `1.0 0.5`                                 |

> **Cartesian product** by default.
> Pass `--grid grids.yaml` at launch if you want to provide a custom grid.

---

## Agent workflow

1. **Read grid**

   ```python
   import itertools, yaml, copy
   grid = yaml.safe_load(open("grids.yaml"))  # fallback to hard-coded table above
   combos = itertools.product(*grid.values())
   param_names = list(grid.keys())
   ```
2. **Loop over grid**

   ```python
   for combo in combos:
       kwargs = dict(zip(param_names, combo))
       cfg    = Config.from_cli_args(**kwargs)
       for seed in range(cfg.seeds):
           run_experiment(cfg, seed)          # wrapper around `run.py`
   ```
3. **Dynamic config injection**
   *Preferred:* call `Config.from_cli_args`.

4. **Run execution**

   ```bash
   python run.py --silent   # produce event logs
   ```
5. **Write Parquet events**
   After each seed, write event data to Parquet format:
   `results_parquet/events/grid_id=<hash>/seed=<seed>/data.parquet`.

---

## Output directory layout

```
results_parquet/
├── events/                    # Event-level data only
│   └── grid_id=abc12345/
│       ├── seed=0/
│       │   └── data.parquet
│       ├── seed=1/
│       │   └── data.parquet
│       └── ...
└── grids/                     # Parameter metadata
    └── grid_id=abc12345/
        └── params.json
```

---

## Extending the grid

* Add new keys in `grids.yaml` – the agent automatically picks them up.
* If a parameter is **absent** from `Config`, the agent should warn and skip that combo.
* For large grids, set `--max-procs N` to spawn `N` worker processes.

---

## Minimal CLI example

```bash
# Basic grid run
python agents/grid_runner.py \
      --grid grids.yaml \
      --max-procs 8 \
      --parquet-out results_parquet

# The script writes event-level Parquet ready for DuckDB queries.
```
