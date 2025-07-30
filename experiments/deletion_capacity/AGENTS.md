**AGENTS.md**

---

## Purpose

Automate a **grid-search** over deletion-capacity experiments.
For every combination of tuning parameters, the agent must

1. **Patch** `config.py` (or instantiate `Config` via `from_cli_args`) with the new values.
2. **Run** `run.py` once per seed.
3. **Collect** the per-seed CSV produced by `RunLogger` (or the legacy writer).
4. **Concatenate** all runs into a single DataFrame whose schema is *identical* to the current default-accountant schema, **except** that

   * when `accountant="rdp"` extra columns `eps_converted`, `eps_remaining`, `delta_total`, `sens_*` are permitted;
   * when `accountant="default"` the columns `eps_spent`, `capacity_remaining`, `eps_step_theory`, `delta_step_theory` are permitted.

The agent then writes one aggregated CSV per grid cell and a master CSV covering the entire sweep.

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
> Pass `--grid-file grids.yaml` at launch if you want to provide a custom grid.

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
   *Fallback:* monkey-patch the dataclass before `run.py` is imported:

   ```python
   import importlib, types, pathlib, textwrap
   cfg_txt = textwrap.dedent(f"""
       dataset      = '{cfg.dataset}'
       gamma_learn  = {cfg.gamma_learn}
       ...
   """)
   pathlib.Path("config_runtime.py").write_text(cfg_txt)
   os.environ["CONFIG_MODULE"] = "config_runtime"   # read in run.py
   ```
4. **Run execution**

   ```bash
   python run.py --silent   # ensure it honours CONFIG_MODULE env-var
   ```
5. **Harvest CSVs**
   After each seed, move `${out_dir}/runs/{seed:03d}_${algo}.csv` to
   `sweep/${grid_id}/seed_${seed}.csv`.
6. **Aggregate**

   ```python
   import pandas as pd, glob
   df = pd.concat(pd.read_csv(f) for f in glob.glob("sweep/*/seed_*.csv"))
   df.to_csv("sweep/all_runs.csv", index=False)
   ```

   *Verify column match:*

   ```python
   base_cols = {...}  # list from the current default schema
   for acct in ("default", "rdp"):
       required = base_cols | EXTRA_COLS[acct]
       assert required <= set(df.columns)
   ```
7. **Report** – optionally call `plot_capacity_curve` / `plot_regret` for each cell.

---

## Output directory layout

```
sweep/
├── split_0.7-0.3_q0.95_k10_default/
│   ├── seed_000.csv
│   ├── seed_001.csv
│   └── ...
├── split_0.3-0.7_q0.90_k1_rdp/
│   └── ...
└── all_runs.csv               # master dataframe
```

---

## Extending the grid

* Add new keys in `grids.yaml` – the agent automatically picks them up.
* If a parameter is **absent** from `Config`, the agent should warn and skip that combo.
* For large grids, set `--parallel N` to spawn `N` worker processes, each assigned a slice of seeds.

---

## Minimal CLI example

```bash
# grid file + 8-process parallel run
python agents/grid_runner.py \
      --grid-file grids.yaml \
      --parallel 8 \
      --base-out results/grid_2025_07_30
```

The script leaves `all_runs.csv` ready for import into your LaTeX tables or plotting notebooks.
