#!/usr/bin/env python3
"""
Main experiment runner class and helpers
for experiment grid searches.git 
"""

from __future__ import annotations
import itertools
import yaml
import os
import sys
import argparse
from typing import Dict, List, Any, Optional
import multiprocessing as mp
from functools import partial

import math
import json

"""
Ensure local, self-contained imports when running as a script:
- Add the experiment directory to sys.path (not the parent) so that
  `from configs.config import Config` and `from runner import ExperimentRunner`
  resolve correctly after the repository reorg.
"""
_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from configs.config import Config
from runner import ExperimentRunner

# Import exp_engine integration
from exp_integration import build_params_from_config

# Optional: Parquet utilities (exp_engine), keep lazy/optional
try:  # pragma: no cover - optional dependency
    from exp_engine.engine.duck import create_connection_and_views as open_duckdb
except Exception:
    open_duckdb = None

# Module flag default path for Parquet
PARQUET_OUT = os.environ.get("PARQUET_OUT", "results_parquet")


def iter_df_in_chunks(df, chunk_size=50000):
    """Iterate over DataFrame in chunks to avoid large in-memory conversions."""
    n = len(df)
    for i in range(0, n, chunk_size):
        yield df.iloc[i : i + chunk_size]


def load_grid(grid_file: str) -> Dict[str, Any]:
    """Load parameter grid from YAML file.

    Supports legacy flat format (treated as matrix) and structured format
    with reserved keys: matrix, selectors, include, exclude, cases, limit, version.
    """
    with open(grid_file, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Legacy mode: treat entire file as matrix if no structured keys present
    if not isinstance(raw, dict):
        return {"matrix": {}}
    if (
        "matrix" not in raw
        and "selectors" not in raw
        and "include" not in raw
        and "exclude" not in raw
        and "cases" not in raw
    ):
        return {"matrix": raw}

    return raw


def _coerce_numeric_like(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"none", "null"}:
            return None
        if s in {"true", "false"}:
            return s == "true"
        try:
            if s in {"inf", "+inf", "infinity", "+infinity"}:
                return math.inf
            if s in {"-inf", "-infinity"}:
                return -math.inf
            return float(s)  # handles "1e-6", "0.1", "3"
        except ValueError:
            return v
    if isinstance(v, (list, tuple)):
        return [_coerce_numeric_like(x) for x in v]
    return v


def sanitize_params(params):
    return {k: _coerce_numeric_like(v) for k, v in params.items()}


# Selector/filter helpers for structured grids
RESERVED_KEYS = {
    "matrix",
    "selectors",
    "include",
    "exclude",
    "cases",
    "limit",
    "version",
}


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _match_where(combo: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """Return True if all constraints in 'where' match the combo.

    - Values can be scalars or lists (interpreted as membership).
    - Uses _coerce_numeric_like for normalization.
    """
    if not where:
        return True
    for k, want in where.items():
        want_list = _as_list(want)
        want_list = [_coerce_numeric_like(v) for v in want_list]
        have = _coerce_numeric_like(combo.get(k, None))
        if have not in want_list:
            return False
    return True


def _select_by_named_selectors(
    combos: List[Dict[str, Any]],
    selectors: List[Dict[str, Any]],
    include_items: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    name_to_where = {}
    for s in selectors or []:
        if isinstance(s, dict) and "name" in s:
            name_to_where[s["name"]] = s.get("where", {})

    selected: List[Dict[str, Any]] = []
    seen = set()

    for item in include_items or []:
        if isinstance(item, dict) and "where" in item:
            where = item["where"]
        elif isinstance(item, str):
            where = name_to_where.get(item, {})
        else:
            # ignore malformed include item
            continue

        for c in combos:
            if _match_where(c, where):
                key = json.dumps(
                    sanitize_params(c), sort_keys=True, separators=(",", ":")
                )
                if key not in seen:
                    seen.add(key)
                    selected.append(c)
    return selected


def _apply_excludes(
    combos: List[Dict[str, Any]], excludes: Optional[List[Any]]
) -> List[Dict[str, Any]]:
    if not excludes:
        return list(combos)
    out: List[Dict[str, Any]] = []
    for c in combos:
        drop = False
        for ex in excludes or []:
            where = ex.get("where", {}) if isinstance(ex, dict) else {}
            if _match_where(c, where):
                drop = True
                break
        if not drop:
            out.append(c)
    return out


def generate_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid.

    Rules:
    - gamma_bar and gamma_split are paired by index (with broadcasting), not crossed.
    - Scalar values are treated as singletons (no expansion).
    - List values are treated as options to cross, EXCEPT when the key indicates a
      literal list parameter (e.g., keys ending with '_bounds', '_clip', '_range',
      or known literal-list keys). Those are treated as a single literal value.
    - Other parameters are crossed via Cartesian product.
    """

    def is_literal_list_key(k: str) -> bool:
        k_lower = k.lower()
        if (
            k_lower.endswith("_bounds")
            or k_lower.endswith("_clip")
            or k_lower.endswith("_range")
        ):
            return True
        # Known literal list keys
        known = {
            "lbfgs_spectrum_clip",
            "lambda_est_bounds",
        }
        return k in known

    # Normalize grid into options dict: each key maps to a list of choices
    options: Dict[str, List[Any]] = {}
    for k, v in grid.items():
        if k in ("gamma_bar", "gamma_split"):
            # handled specially later; still ensure list form
            if isinstance(v, (list, tuple)):
                options[k] = list(v)
            else:
                options[k] = [v]
            continue

        if isinstance(v, (list, tuple)):
            if is_literal_list_key(k):
                # Literal list semantics:
                # - If user provided [[...]] (list of one list), use the inner list as the single literal value
                # - If user provided [[...], [...]] (multiple inner lists), treat each inner list as an option
                # - Otherwise (list of scalars), treat entire list as the single literal value
                if len(v) > 0 and all(isinstance(e, (list, tuple)) for e in v):
                    if len(v) == 1:
                        options[k] = [list(v[0])]
                    else:
                        options[k] = [list(e) for e in v]
                else:
                    options[k] = [list(v)]
            else:
                options[k] = list(v)
        else:
            options[k] = [v]

    # Special handling for gamma pairing
    has_gamma_pair = "gamma_bar" in options and "gamma_split" in options
    if has_gamma_pair:
        gamma_bar_list = options["gamma_bar"]
        gamma_split_list = options["gamma_split"]
        len_bar = len(gamma_bar_list)
        len_split = len(gamma_split_list)

        # Broadcasting
        if len_bar == 1 and len_split > 1:
            gamma_bar_list = gamma_bar_list * len_split
        elif len_split == 1 and len_bar > 1:
            gamma_split_list = gamma_split_list * len_bar
        elif len_bar != len_split:
            raise ValueError(
                f"gamma_bar and gamma_split must be the same length, or one must be length 1. "
                f"Got gamma_bar (len={len_bar}): {gamma_bar_list}, gamma_split (len={len_split}): {gamma_split_list}."
            )

        gamma_pairs = list(zip(gamma_bar_list, gamma_split_list))

        # Remove gamma keys from options for the Cartesian product
        other_keys = [
            k for k in options.keys() if k not in ("gamma_bar", "gamma_split")
        ]
        other_option_lists = [options[k] for k in other_keys]
        other_combos = (
            list(itertools.product(*other_option_lists)) if other_option_lists else [()]
        )

        # Build raw combinations honoring gamma pairing
        raw: List[Dict[str, Any]] = []
        for gb, gs in gamma_pairs:
            for other_combo in other_combos:
                combo = {"gamma_bar": gb, "gamma_split": gs}
                combo.update(dict(zip(other_keys, other_combo)))
                raw.append(combo)
    else:
        # No special gamma pairing; regular Cartesian product across all options
        keys = list(options.keys())
        option_lists = [options[k] for k in keys]
        combos = list(itertools.product(*option_lists)) if option_lists else [()]
        raw = [dict(zip(keys, c)) for c in combos]

    # Deduplicate combinations after normalizing values (e.g., numeric-like strings)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    import json as _json

    for c in raw:
        norm = sanitize_params(c)
        key = _json.dumps(norm, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def create_grid_id(params: Dict[str, Any]) -> str:
    """Create a unique identifier for this parameter combination (zCDP-only)."""
    import hashlib
    import json

    # Create a short, stable hash that reflects ALL parameters, so that grid ids are unique
    # across combinations even when not explicitly encoded in the human-readable prefix.
    def _normalize(x):
        # Make params JSON-serializable and order-stable
        if isinstance(x, dict):
            return {k: _normalize(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [_normalize(v) for v in x]
        # Convert numpy scalars to Python scalars if present
        try:
            import numpy as _np  # local import to avoid top-level dependency

            if isinstance(x, _np.generic):
                return x.item()
        except Exception:
            pass
        return x

    norm = _normalize(params)
    hash_src = json.dumps(
        norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    short_h = hashlib.md5(hash_src.encode("utf-8")).hexdigest()
    return short_h


def run_single_experiment(
    params: Dict[str, Any],
    seed: int,
    base_out_dir: str,
    output_granularity: str = "seed",
) -> Optional[str]:
    """Run a single experiment with given parameters and seed."""
    import re

    # Create config from parameters
    config_kwargs = params.copy()
    config_kwargs["seeds"] = 1  # Single seed per run
    config_kwargs["output_granularity"] = output_granularity  # Pass through granularity

    # Set output directory for this specific run
    grid_id = create_grid_id(params)
    run_out_dir = os.path.join(base_out_dir, "sweep", grid_id)
    os.makedirs(run_out_dir, exist_ok=True)
    config_kwargs["out_dir"] = run_out_dir

    # Set some defaults for faster testing
    config_kwargs.setdefault("bootstrap_iters", 500)  # Reduced for testing

    try:
        config_kwargs = sanitize_params(config_kwargs)
        numeric_pattern = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
        bad = {
            k: v
            for k, v in config_kwargs.items()
            if isinstance(v, str) and numeric_pattern.match(v.strip())
        }
        if bad:
            print(
                "Warning: numeric-like strings detected; consider fixing grids.yaml:",
                bad,
            )
        # Create config and runner
        cfg = Config.from_cli_args(**config_kwargs)
        runner = ExperimentRunner(cfg)

        # Run for this specific seed
        runner.run_one_seed(seed)

        return

    except Exception as e:
        print(f"Error running experiment for seed {seed} with params {params}: {e}")
        return None


def run_parameter_combination(
    params: Dict[str, Any],
    seeds: List[int],
    base_out_dir: str,
    output_granularity: str = "seed",
    parallel: int = 1,
    parquet_out: str = "results_parquet",
    parquet_write_events: bool = False,
    no_legacy_csv: bool = False,
) -> List[str]:
    """Run all seeds for a single parameter combination."""
    grid_id = create_grid_id(params)
    print(f"\n=== Running grid cell: {grid_id} ===")
    print(f"Parameters: {params}")
    print(f"Output granularity: {output_granularity}")

    # Create output directory for this grid cell
    grid_output_dir = os.path.join(base_out_dir, "sweep", grid_id)
    os.makedirs(grid_output_dir, exist_ok=True)

    if parallel == 1:
        # Sequential execution
        for seed in seeds:
            run_single_experiment(params, seed, base_out_dir, output_granularity)

    else:
        # Parallel execution
        with mp.Pool(parallel) as pool:
            run_func = partial(
                run_single_experiment,
                params,
                base_out_dir=base_out_dir,
                output_granularity=output_granularity,
            )
            pool.map(run_func, seeds)

    # Optionally process legacy CSV outputs if present (tests expect helpers)
    try:
        csv_files = []
        for fname in os.listdir(grid_output_dir):
            if fname.endswith(".csv") and "memorypair" in fname:
                csv_files.append(os.path.join(grid_output_dir, fname))

        if csv_files:
            mandatory_fields = {
                "G_hat": params.get("G_hat", None),
                "D_hat": params.get("D_hat", None),
                "sigma_step_theory": params.get("sigma_step_theory", None),
                # Common params propagated to outputs
                "gamma_bar": params.get("gamma_bar", None),
                "gamma_split": params.get("gamma_split", None),
                "eps_total": params.get("eps_total", None),
                "accountant": params.get("accountant", "zcdp"),
            }

            if output_granularity == "seed":
                process_seed_output(csv_files, grid_id, grid_output_dir, mandatory_fields)
            elif output_granularity == "event":
                process_event_output(csv_files, grid_id, grid_output_dir, mandatory_fields)
            elif output_granularity == "aggregate":
                process_aggregate_output(
                    csv_files, grid_id, grid_output_dir, mandatory_fields
                )
    except Exception as _e:
        print(f"[warn] Skipping CSV processing for {grid_id}: {_e}")

    return


# ===============================
# Output processing helpers (CSV)
# ===============================
def _ensure_mandatory_fields(df, mandatory_fields: Dict[str, Any]):
    for k, v in (mandatory_fields or {}).items():
        if k not in df.columns or df[k].isna().all():
            df[k] = v
    return df


def _extract_seed_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    try:
        # Expect filenames like: "0_memorypair.csv" or "seed_000_memorypair.csv"
        stem = base.split("_")[0]
        if stem == "seed":
            stem = base.split("_")[1]
        return int(stem)
    except Exception:
        return None


def process_seed_output(
    csv_files: List[str],
    grid_id: str,
    output_dir: str,
    mandatory_fields: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Aggregate per-seed metrics from event CSVs into one seed-level CSV.

    Produces columns required by tests: seed, grid_id, avg_regret_empirical,
    N_star_emp, m_emp, plus mandatory fields.
    """
    import pandas as pd

    rows = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        df = _ensure_mandatory_fields(df, mandatory_fields or {})

        # Basic empirical metrics
        avg_regret_emp = float(df["regret"].mean()) if "regret" in df.columns else float("nan")
        n_star_emp = int(df.shape[0])  # Simple proxy: events processed
        m_emp = int((df["op"] == "delete").sum()) if "op" in df.columns else 0

        seed = _extract_seed_from_filename(path)

        row = {
            "seed": seed if seed is not None else -1,
            "grid_id": grid_id,
            "avg_regret_empirical": avg_regret_emp,
            "N_star_emp": n_star_emp,
            "m_emp": m_emp,
        }

        # Carry mandatory fields into the seed row
        for k in (mandatory_fields or {}):
            row[k] = df[k].iloc[0] if k in df.columns else mandatory_fields[k]

        rows.append(row)

    if not rows:
        return None

    out_df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seed_summary_{grid_id}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


def process_event_output(
    csv_files: List[str],
    grid_id: str,
    output_dir: str,
    mandatory_fields: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Concatenate event CSVs and annotate with seed and grid_id.

    Ensures required columns exist: event, event_type, op, regret, acc, seed, grid_id.
    """
    import pandas as pd
    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = _ensure_mandatory_fields(df, mandatory_fields or {})
        seed = _extract_seed_from_filename(path)
        df["seed"] = seed if seed is not None else -1
        df["grid_id"] = grid_id
        # Normalize event_type
        if "event_type" not in df.columns and "op" in df.columns:
            df["event_type"] = df["op"]
        frames.append(df)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"events_{grid_id}.csv")
    out.to_csv(out_path, index=False)
    return out_path


def process_aggregate_output(
    csv_files: List[str],
    grid_id: str,
    output_dir: str,
    mandatory_fields: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Compute aggregate metrics across seeds using seed-level summaries.

    Produces: grid_id, num_seeds, avg_regret_mean, N_star_mean, m_mean, plus mandatory fields.
    """
    import pandas as pd

    seed_csv = process_seed_output(csv_files, grid_id, output_dir, mandatory_fields)
    if not seed_csv:
        return None

    seeds_df = pd.read_csv(seed_csv)
    agg = {
        "grid_id": grid_id,
        "num_seeds": int(seeds_df.shape[0]),
        "avg_regret_mean": float(seeds_df["avg_regret_empirical"].mean())
        if "avg_regret_empirical" in seeds_df.columns
        else float("nan"),
        "N_star_mean": float(seeds_df["N_star_emp"].mean()) if "N_star_emp" in seeds_df.columns else float("nan"),
        "m_mean": float(seeds_df["m_emp"].mean()) if "m_emp" in seeds_df.columns else float("nan"),
    }

    for k, v in (mandatory_fields or {}).items():
        if k not in agg:
            agg[k] = v

    out_df = pd.DataFrame([agg])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"aggregate_{grid_id}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Grid search runner for deletion capacity experiments"
    )
    parser.add_argument(
        "--grid-file", default="grids.yaml", help="YAML file with parameter grid"
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel processes"
    )
    parser.add_argument(
        "--base-out", default="results/grid_2025_01_01", help="Base output directory"
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds per grid cell"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show combinations without running"
    )
    parser.add_argument(
        "--output-granularity",
        choices=["seed", "event", "aggregate"],
        default="seed",
        help="Output granularity: seed (one row per seed), event (one row per event), aggregate (one row per grid-id)",
    )
    parser.add_argument(
        "--parquet-out",
        default=PARQUET_OUT,
        help="Base output directory for Parquet files",
    )
    parser.add_argument(
        "--parquet-write-events",
        default=True,
        action="store_true",
        help="Store per-event logs to Parquet (can be large)",
    )
    parser.add_argument(
        "--no-legacy-csv",
        default=True,
        action="store_true",
        help="Skip writing legacy CSV files",
    )

    args = parser.parse_args()

    # Validate output granularity (redundant but kept for clarity)
    if args.output_granularity not in ["seed", "event", "aggregate"]:
        print(
            f"Error: Invalid output-granularity '{args.output_granularity}'. Must be one of: seed, event, aggregate"
        )
        return 1

    # Load parameter grid
    print(f"Loading grid from {args.grid_file}")
    if not os.path.exists(args.grid_file):
        print(f"Error: Grid file {args.grid_file} not found")
        return 1

    grid_raw = load_grid(args.grid_file)

    # Backwards compat: matrix is either explicit or the entire file
    matrix = grid_raw.get("matrix", grid_raw)
    selectors = grid_raw.get("selectors", [])
    includes = grid_raw.get("include", None)
    cases = grid_raw.get("cases", [])
    excludes = grid_raw.get("exclude", [])
    limit = grid_raw.get("limit", None)

    print(f"Matrix parameters: {list(matrix.keys())}")

    # Generate full combinations from matrix only
    combinations_full = generate_combinations(matrix)
    print(f"Generated {len(combinations_full)} parameter combinations (full product)")

    # Determine selected subset
    if includes or cases:
        selected = _select_by_named_selectors(combinations_full, selectors, includes)
        if cases:
            case_includes = [{"where": c} for c in cases if isinstance(c, dict)]
            selected += _select_by_named_selectors(
                combinations_full, selectors, case_includes
            )
        # de-dup
        seen = set()
        combinations: List[Dict[str, Any]] = []
        for c in selected:
            key = json.dumps(sanitize_params(c), sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                combinations.append(c)
    else:
        combinations = list(combinations_full)

    # Apply excludes and limit
    combinations = _apply_excludes(combinations, excludes)
    if isinstance(limit, int) and limit > 0:
        combinations = combinations[:limit]

    print(f"Selected {len(combinations)} parameter combinations after filters")

    if args.dry_run:
        print("\nDry run - parameter combinations:")
        for i, combo in enumerate(combinations):
            grid_id = create_grid_id(combo)
            print(f"{i + 1:3d}. {grid_id}: {combo}")
        return 0

    # Create base output directory
    os.makedirs(args.base_out, exist_ok=True)
    sweep_dir = os.path.join(args.base_out, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    # Generate seed list
    seeds = list(range(args.seeds))

    # Run all combinations
    print(
        f"\nRunning {len(combinations)} combinations x {len(seeds)} seeds = {len(combinations) * len(seeds)} total experiments"
    )
    print(f"Output granularity: {args.output_granularity}")

    for i, params in enumerate(combinations):
        print(f"\nProgress: {i + 1}/{len(combinations)}")
        run_parameter_combination(
            params,
            seeds,
            args.base_out,
            args.output_granularity,
            args.parallel,
            args.parquet_out,
            args.parquet_write_events,
            args.no_legacy_csv,
        )

    # Create sweep manifest (requirement 2)
    try:
        print("\n=== Creating sweep manifest ===")

        # Create manifest mapping grid_id to parameters
        manifest = {}
        for params in combinations:
            grid_id = create_grid_id(params)
            manifest[grid_id] = params

        # Write manifest.json
        manifest_file = os.path.join(sweep_dir, "manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Sweep manifest saved to: {manifest_file}")
        print(f"   Manifest contains {len(manifest)} grid cells")

    except Exception as e:
        print(f"Warning: Failed to create sweep manifest: {e}")

    print("\nGrid search complete!")
    print(f"Results in: {sweep_dir}")
    print(f"Output granularity used: {args.output_granularity}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
