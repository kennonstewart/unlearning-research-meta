#!/usr/bin/env python3
"""
Grid search runner for deletion capacity experiments.
Implements the workflow described in AGENTS.md.
"""

import itertools
import yaml
import os
import sys
import argparse
import glob
from typing import Dict, List, Any, Optional
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import math
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import Config
from runner import ExperimentRunner

# Import exp_engine integration
from exp_integration import (
    build_params_from_config,
    write_seed_summary_parquet,
    write_event_rows_parquet,
)
# For Parquet-first aggregation
try:
    from exp_integration import open_duckdb  # thin wrapper around exp_engine.engine.duck
except Exception:
    open_duckdb = None

# Module flag default path for Parquet
PARQUET_OUT = os.environ.get("PARQUET_OUT", "results_parquet")


def iter_df_in_chunks(df, chunk_size=50000):
    """Iterate over DataFrame in chunks to avoid large in-memory conversions."""
    n = len(df)
    for i in range(0, n, chunk_size):
        yield df.iloc[i:i+chunk_size]


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

    gamma_bar = params.get("gamma_bar", 1.0)
    gamma_split = params.get("gamma_split", 0.5)
    quantile = params.get("quantile", 0.95)
    delete_ratio = params.get("delete_ratio", 10)
    rho_total = params.get("rho_total", 1.0)
    # Drift & scale knobs (optional)
    path_type = params.get("path_type", "rotating")
    rotate_angle = params.get("rotate_angle", 0.01)
    drift_rate = params.get("drift_rate", 0.001)
    feature_scale = params.get("feature_scale", 1.0)
    # Shorten path_type for ID
    p = {"static": "st", "rotating": "rot", "drift": "dr"}.get(
        str(path_type), str(path_type)[:3]
    )

    comparator = params.get("comparator", "dynamic")
    oracle_flag = "on" if params.get("enable_oracle", False) else "off"

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
    short_h = hashlib.md5(hash_src.encode("utf-8")).hexdigest()[:8]

    # Theory-first compact suffix (only include if provided)
    tf_parts = []
    tg = params.get("target_G", None)
    tD = params.get("target_D", None)
    tc = params.get("target_c", None)
    tC = params.get("target_C", None)
    tl = params.get("target_lambda", None)
    tpt = params.get("target_PT", None)
    tst = params.get("target_ST", None)
    ps = params.get("path_style", params.get("path_type", None))

    if tpt is not None:
        tf_parts.append(f"PT{tpt:g}")
    if tst is not None:
        tf_parts.append(f"ST{tst:g}")
    if tg is not None:
        tf_parts.append(f"G{tg:.2f}")
    if tl is not None:
        tf_parts.append(f"lam{tl:.3g}")
    if tc is not None and tC is not None:
        tf_parts.append(f"c{tc:.3g}-C{tC:.3g}")
    elif tc is not None:
        tf_parts.append(f"c{tc:.3g}")
    elif tC is not None:
        tf_parts.append(f"C{tC:.3g}")
    if ps:
        p = {
            "static": "st",
            "rotating": "rot",
            "drift": "dr",
            "brownian": "br",
            "piecewise-constant": "pc",
        }.get(str(ps), str(ps)[:3])

    return (
        f"gamma_{gamma_bar:.1f}-split_{gamma_split:.1f}_q{quantile:.2f}_k{delete_ratio:.0f}_"
        f"zcdp_rho{rho_total:.1f}_cmp{comparator}_orc{oracle_flag}_"
        f"p{p}_ang{rotate_angle:.3g}_dr{drift_rate:.3g}_fs{feature_scale:.3g}_"
        + ("_".join(tf_parts) + "_" if tf_parts else "")
        + f"h{short_h}"
    )


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
        runner.run_single_seed(seed)

        # The runner should create a CSV file directly in the out_dir
        # Look for files matching the expected pattern
        csv_pattern = os.path.join(run_out_dir, f"*{seed}*.csv")
        csv_files = glob.glob(csv_pattern)

        if csv_files:
            # Return the CSV file path
            return csv_files[0]
        else:
            print(
                f"Warning: No CSV file found for seed {seed} with pattern {csv_pattern}"
            )
            return None

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

    csv_paths = []

    if parallel == 1:
        # Sequential execution
        for seed in seeds:
            result = run_single_experiment(
                params, seed, base_out_dir, output_granularity
            )
            if result:
                csv_paths.append(result)
    else:
        # Parallel execution
        with mp.Pool(parallel) as pool:
            run_func = partial(
                run_single_experiment,
                params,
                base_out_dir=base_out_dir,
                output_granularity=output_granularity,
            )
            results = pool.map(run_func, seeds)
            csv_paths = [r for r in results if r is not None]

    # Ensure ALL mandatory fields that should be consistent across all seeds
    mandatory_fields = {
        # Static parameters from grid
        "gamma_bar": params.get("gamma_bar", np.nan),
        "gamma_split": params.get("gamma_split", np.nan),
        "accountant": params.get("accountant", "unknown"),
        "quantile": params.get("quantile", np.nan),
        "delete_ratio": params.get("delete_ratio", np.nan),
        "eps_total": params.get("eps_total", np.nan),
        "blocked_reason": "",  # Default to empty string
        # Calibration/diagnostics fields - will be filled from CSV if available
        "G_hat": np.nan,
        "D_hat": np.nan,
        "c_hat": np.nan,
        "C_hat": np.nan,
        "lambda_est": np.nan,
        "S_scalar": np.nan,
        # Odometer fields
        "sigma_step_theory": np.nan,
        "N_star_live": np.nan,
        "m_theory_live": np.nan,
        # Theory-first targets
        "target_G": params.get("target_G", np.nan),
        "target_D": params.get("target_D", np.nan),
        "target_c": params.get("target_c", np.nan),
        "target_C": params.get("target_C", np.nan),
        "target_lambda": params.get("target_lambda", np.nan),
        "target_PT": params.get("target_PT", np.nan),
        "target_ST": params.get("target_ST", np.nan),
        "rho_total": params.get("rho_total", np.nan),
        "path_style": params.get("path_style", params.get("path_type", None)),
    }

    if csv_paths:
        try:
            # Read first CSV to extract calibration fields that aren't in params
            first_df = pd.read_csv(csv_paths[0])
            if len(first_df) > 0:
                last_row = first_df.iloc[-1]
                # Extract calibration/diagnostic fields from the last row
                for field in [
                    "G_hat",
                    "D_hat",
                    "c_hat",
                    "C_hat",
                    "lambda_est",
                    "S_scalar",
                    "sigma_step_theory",
                    "N_star_live",
                    "m_theory_live",
                    "blocked_reason",
                ]:
                    if field in last_row and not pd.isna(last_row[field]):
                        mandatory_fields[field] = last_row[field]
        except Exception as e:
            print(f"Warning: Could not extract calibration fields from CSV: {e}")

    # Save params.json for this grid cell
    params_file = os.path.join(grid_output_dir, "params.json")
    with open(params_file, "w") as f:
        import json

        json.dump(params, f, indent=2)

    # Build params for Parquet writing (content-addressed hashing)
    params_with_grid = build_params_from_config(params)

    # Process outputs based on granularity
    processed_files = []
    if output_granularity == "seed":
        processed_files = process_seed_output(
            csv_paths, grid_id, grid_output_dir, mandatory_fields,
            parquet_out, params_with_grid, no_legacy_csv
        )
    elif output_granularity == "event":
        processed_files = process_event_output(
            csv_paths, grid_id, grid_output_dir, mandatory_fields,
            parquet_out, params_with_grid, parquet_write_events, no_legacy_csv
        )
    elif output_granularity == "aggregate":
        aggregate_file = process_aggregate_output(
            csv_paths, grid_id, grid_output_dir, mandatory_fields,
            parquet_out, params_with_grid, no_legacy_csv
        )
        if aggregate_file:
            processed_files = [aggregate_file]

    print(f"Completed {len(csv_paths)}/{len(seeds)} seeds for {grid_id}")
    print(
        f"Generated {len(processed_files)} output files with {output_granularity} granularity"
    )

    return processed_files


def aggregate_results(sweep_dir: str) -> Optional[str]:
    """Aggregate all CSV results into a master file."""
    print("\n=== Aggregating results ===")

    # Find all seed CSV files
    csv_files = glob.glob(os.path.join(sweep_dir, "*", "seed_*.csv"))

    if not csv_files:
        print("Warning: No CSV files found to aggregate")
        return None

    print(f"Found {len(csv_files)} CSV files to aggregate")

    # Read and concatenate all CSV files
    import json

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            grid_id = os.path.basename(os.path.dirname(csv_file))
            seed_file = os.path.basename(csv_file)
            seed = int(seed_file.split("_")[1].split(".")[0])

            # Add metadata columns
            df["grid_id"] = grid_id
            df["seed"] = seed

            # Read grid parameters from params.json
            params_path = os.path.join(os.path.dirname(csv_file), "params.json")
            params = {}
            if os.path.exists(params_path):
                with open(params_path, "r") as f:
                    params = json.load(f)
            # Attach relevant parameters as columns
            for k in [
                "gamma_bar",
                "gamma_split",
                "quantile",
                "delete_ratio",
                "accountant",
                "eps_total",
                "rho_total",
                "delta_total",
                # Drift/scale/feature controls
                "path_type",
                "path_style",
                "rotate_angle",
                "drift_rate",
                "feature_scale",
                "w_scale",
                "fix_w_norm",
                "noise_std",
                # Theory-first targets
                "target_G",
                "target_D",
                "target_c",
                "target_C",
                "target_lambda",
                "target_PT",
                "target_ST",
            ]:
                if k in params:
                    df[k] = params[k]
                else:
                    # Only add column if present in at least one params.json
                    if k in ["eps_total", "rho_total", "delta_total"]:
                        # skip if not present
                        continue
                    else:
                        df[k] = None
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")

    if not dfs:
        print("Error: No valid CSV files could be read")
        return None

    # Concatenate all dataframes
    master_df = pd.concat(dfs, ignore_index=True)

    # Write master CSV
    master_path = os.path.join(sweep_dir, "all_runs.csv")
    master_df.to_csv(master_path, index=False)

    print(f"Aggregated {len(master_df)} rows into {master_path}")
    print(f"Columns: {list(master_df.columns)}")

    return master_path


def aggregate_results_from_parquet(parquet_dir: str, sweep_dir: str) -> (Optional[str], Optional[str]):
    """Aggregate Parquet datasets into master outputs.

    Reads Parquet from results_parquet using DuckDB views and writes:
    - all_runs.csv (for backward compatibility with existing plots)
    - all_runs.parquet (Parquet-first artifact for downstream use)

    Returns (csv_path, parquet_path) where either may be None on failure.
    """
    print("\n=== Aggregating results (Parquet) ===")

    if open_duckdb is None:
        print("Warning: DuckDB not available; cannot aggregate from Parquet.")
        return None, None

    try:
        conn = open_duckdb(parquet_dir)
    except Exception as e:
        print(f"Warning: Failed to open DuckDB over Parquet at {parquet_dir}: {e}")
        return None, None

    # Prefer seed summaries if present; fall back to event summaries
    master_df = None
    try:
        # Try direct seeds view
        master_df = conn.execute("SELECT * FROM seeds").df()
    except Exception:
        master_df = None

    if master_df is None or master_df.empty:
        try:
            # Fallback: summarize events to approximate seed-level rows
            # Assumes event Parquet contains params columns via exp_integration enrichment
            master_df = conn.execute(
                """
                SELECT
                  grid_id,
                  COALESCE(seed, 0) AS seed,
                  AVG(CASE WHEN regret IS NOT NULL THEN regret END) AS avg_regret_empirical,
                  SUM(CASE WHEN op = 'insert' THEN 1 ELSE 0 END) AS N_star_emp,
                  SUM(CASE WHEN op = 'delete' THEN 1 ELSE 0 END) AS m_emp,
                  COUNT(*) AS total_events,
                  -- Pass through some common params when available
                  any_value(algo) AS algo,
                  any_value(accountant) AS accountant,
                  any_value(gamma_bar) AS gamma_bar,
                  any_value(gamma_split) AS gamma_split,
                  any_value(delete_ratio) AS delete_ratio,
                  any_value(rho_total) AS rho_total,
                  any_value(target_PT) AS target_PT,
                  any_value(target_ST) AS target_ST
                FROM events
                GROUP BY grid_id, seed
                """
            ).df()
        except Exception as e:
            print(f"Warning: Failed to summarize events from Parquet: {e}")
            master_df = None

    if master_df is None or master_df.empty:
        print("Warning: No Parquet data found to aggregate (seeds/events).")
        return None, None

    # Ensure output dir exists
    os.makedirs(sweep_dir, exist_ok=True)

    # Write CSV (compat with plotting) and Parquet artifacts
    csv_path = os.path.join(sweep_dir, "all_runs.csv")
    parquet_path = os.path.join(sweep_dir, "all_runs.parquet")
    try:
        master_df.to_csv(csv_path, index=False)
        print(f"Aggregated {len(master_df)} rows into {csv_path}")
    except Exception as e:
        print(f"Warning: Failed writing CSV aggregate: {e}")
        csv_path = None

    try:
        # If pyarrow is available, write as a single parquet file
        master_df.to_parquet(parquet_path, index=False)
        print(f"Aggregated {len(master_df)} rows into {parquet_path}")
    except Exception as e:
        print(f"Warning: Failed writing Parquet aggregate: {e}")
        parquet_path = None

    return csv_path, parquet_path


def validate_schema(csv_path: str, expected_accountants: List[str]) -> bool:
    """Validate that aggregated CSV has expected schema (simplified for zCDP-only)."""
    if not csv_path or not os.path.exists(csv_path):
        return False

    # Simple validation - just check that file exists and is readable
    try:
        df = pd.read_csv(csv_path)
        print("Schema validation passed (zCDP-only mode)")
        return True
    except Exception as e:
        print(f"Schema validation failed: {e}")
        return False


def process_seed_output(
    csv_files: List[str],
    grid_id: str,
    output_dir: str,
    mandatory_fields: Dict[str, Any],
    parquet_out: str = "results_parquet",
    params_with_grid: Dict[str, Any] = None,
    no_legacy_csv: bool = False,
) -> List[str]:
    """Process CSV files for seed granularity output."""
    processed_files = []
    seed_summaries = []  # Collect all seed summaries for Parquet

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue

            # Extract seed from filename
            seed_file = os.path.basename(csv_file)
            seed_match = [part for part in seed_file.split("_") if part.isdigit()]
            if not seed_match:
                continue
            seed = int(seed_match[0])

            # Aggregate to single row per seed
            summary_row = {
                "seed": seed,
                "grid_id": grid_id,
                "avg_regret_empirical": df["regret"].mean()
                if "regret" in df.columns
                else None,
                "N_star_emp": len(df[df["op"] == "insert"])
                if "op" in df.columns
                else None,
                "m_emp": len(df[df["op"] == "delete"]) if "op" in df.columns else None,
                "final_acc": df["acc"].iloc[-1]
                if "acc" in df.columns and len(df) > 0
                else None,
                "total_events": len(df),
            }

            # Add basic fields from mandatory_fields without strict validation
            for field, value in mandatory_fields.items():
                if field not in summary_row:
                    summary_row[field] = value

            mandatory_field_names = [
                "drift_rate",
                "feature_scale",
                "w_scale",
                "fix_w_norm",
                "noise_std",
                # Theory-first targets
                "target_G",
                "target_D",
                "target_c",
                "target_C",
                "target_lambda",
                "target_PT",
                "target_ST",
                "rho_total",
                "path_style",
            ]

            # Add mandatory fields from parameters first
            for field in mandatory_field_names:
                summary_row[field] = mandatory_fields.get(field, np.nan)

            # Add any additional fields from the last row (e.g., privacy metrics, calibration stats)
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in [
                    "eps_spent",
                    "capacity_remaining",
                    "eps_converted",
                    "eps_remaining",
                    "delta_total",
                    "G_hat",
                    "D_hat",
                    "c_hat",
                    "C_hat",
                    "N_star_theory",
                    "m_theory",
                    "N_star_live",
                    "m_theory_live",
                    "S_scalar",
                    "eta_t",
                    "lambda_est",
                    "sigma_step_theory",
                    "blocked_reason",
                ]:
                    if col in last_row and not pd.isna(last_row[col]):
                        summary_row[col] = last_row[col]

            # Ensure blocked_reason is a string
            if "blocked_reason" not in summary_row or pd.isna(
                summary_row["blocked_reason"]
            ):
                summary_row["blocked_reason"] = ""

            # === Theory-first acceptance metrics (computed from event CSV) ===
            try:
                # Targets
                tG = mandatory_fields.get("target_G", np.nan)
                tPT = mandatory_fields.get("target_PT", np.nan)
                tST = mandatory_fields.get("target_ST", np.nan)
                rho_tot = mandatory_fields.get("rho_total", np.nan)

                # Observables
                PT_final = (
                    float(df["P_T_true"].iloc[-1])
                    if "P_T_true" in df.columns
                    else np.nan
                )
                ST_final = (
                    float(df["ST_running"].iloc[-1])
                    if "ST_running" in df.columns
                    else (
                        float(df["S_scalar"].iloc[-1])
                        if "S_scalar" in df.columns
                        else np.nan
                    )
                )
                max_g = float(df["g_norm"].max()) if "g_norm" in df.columns else np.nan
                clip_rate = (
                    float(df["clip_applied"].mean())
                    if "clip_applied" in df.columns
                    else np.nan
                )
                rho_spent = np.nan
                if "privacy_spend_running" in df.columns:
                    rho_spent = float(df["privacy_spend_running"].iloc[-1])
                elif "privacy_spend" in df.columns:
                    rho_spent = float(df["privacy_spend"].iloc[-1])
                elif "rho_spent" in df.columns:
                    # Fallback to rho_spent for zCDP adapter compatibility
                    rho_spent = float(df["rho_spent"].iloc[-1])

                # Errors
                PT_err = (
                    abs(PT_final / tPT - 1.0)
                    if (not np.isnan(PT_final) and not np.isnan(tPT) and tPT != 0)
                    else np.nan
                )
                ST_err = (
                    abs(ST_final / tST - 1.0)
                    if (not np.isnan(ST_final) and not np.isnan(tST) and tST != 0)
                    else np.nan
                )

                # AT checks
                reasons = []
                if not np.isnan(PT_err) and PT_err > 0.05:
                    reasons.append(f"AT-1 PT err {PT_err * 100:.1f}%")
                if (not np.isnan(max_g) and not np.isnan(tG) and max_g > 1.05 * tG) or (
                    not np.isnan(clip_rate) and clip_rate > 0.05
                ):
                    reasons.append("AT-2 gradient/clipping")
                if not np.isnan(ST_err) and ST_err > 0.05:
                    reasons.append(f"AT-5 ST err {ST_err * 100:.1f}%")
                if (
                    not np.isnan(rho_tot)
                    and not np.isnan(rho_spent)
                    and rho_spent > rho_tot + 1e-9
                ):
                    reasons.append("AT-6 privacy overspend")

                # Store metrics
                summary_row["PT_final"] = PT_final
                summary_row["PT_error"] = None if np.isnan(PT_err) else PT_err
                summary_row["ST_final"] = ST_final
                summary_row["ST_error"] = None if np.isnan(ST_err) else ST_err
                summary_row["max_g_norm"] = max_g
                summary_row["avg_clip_rate"] = clip_rate
                summary_row["rho_spent_final"] = rho_spent
                
                # Compute rho_util for capacity utilization analysis
                if (not np.isnan(rho_spent) and not np.isnan(rho_tot) and rho_tot > 0):
                    summary_row["rho_util"] = rho_spent / rho_tot
                else:
                    summary_row["rho_util"] = np.nan

                # Blocked reason if any failures (append to any existing reason)
                if reasons:
                    prev_reason = summary_row.get("blocked_reason", "") or ""
                    reason_text = "; ".join(reasons)
                    summary_row["blocked_reason"] = (
                        prev_reason
                        + ("; " if prev_reason and reason_text else "")
                        + reason_text
                    )
            except Exception:
                pass

            # === γ-adherence checking (Experiment A pass/fail) ===
            try:
                # Get γ parameters from mandatory fields
                gamma_bar = mandatory_fields.get("gamma_bar", np.nan)
                gamma_split = mandatory_fields.get("gamma_split", np.nan)

                # Compute derived thresholds
                gamma_insert_threshold = (
                    gamma_bar * gamma_split
                    if not np.isnan(gamma_bar) and not np.isnan(gamma_split)
                    else np.nan
                )
                gamma_delete_threshold = (
                    gamma_bar * (1.0 - gamma_split)
                    if not np.isnan(gamma_bar) and not np.isnan(gamma_split)
                    else np.nan
                )

                # Robust computation of avg_regret_final
                avg_regret_final = np.nan

                # Method 1: Try explicit avg_regret column if present
                if "avg_regret" in df.columns and len(df) > 0:
                    avg_regret_final = float(df["avg_regret"].iloc[-1])
                # Method 2: Try cum_regret with event index
                elif "cum_regret" in df.columns and len(df) > 0:
                    cum_regret_final = float(df["cum_regret"].iloc[-1])
                    if "event" in df.columns:
                        event_last = int(df["event"].iloc[-1])
                        avg_regret_final = cum_regret_final / max(
                            event_last + 1, len(df)
                        )
                    else:
                        avg_regret_final = cum_regret_final / len(df)
                # Method 3: Fallback to mean of regret increments
                elif "regret" in df.columns and len(df) > 0:
                    avg_regret_final = float(df["regret"].mean())

                # γ-adherence check: overall pass/fail
                gamma_pass_overall = False
                gamma_error = np.nan
                if (
                    not np.isnan(avg_regret_final)
                    and not np.isnan(gamma_bar)
                    and gamma_bar > 0
                ):
                    gamma_pass_overall = avg_regret_final <= gamma_bar
                    gamma_error = max(0.0, avg_regret_final / gamma_bar - 1.0)

                # Decomposition checks for insert/delete (when regret_increment is present)
                avg_regret_insert_only = np.nan
                avg_regret_delete_only = np.nan
                gamma_pass_insert = np.nan
                gamma_pass_delete = np.nan

                if "regret_increment" in df.columns and "op" in df.columns:
                    insert_regrets = df[df["op"] == "insert"]["regret_increment"]
                    delete_regrets = df[df["op"] == "delete"]["regret_increment"]

                    if len(insert_regrets) > 0:
                        avg_regret_insert_only = float(insert_regrets.mean())
                        if (
                            not np.isnan(gamma_insert_threshold)
                            and gamma_insert_threshold > 0
                        ):
                            gamma_pass_insert = (
                                avg_regret_insert_only <= gamma_insert_threshold
                            )

                    if len(delete_regrets) > 0:
                        avg_regret_delete_only = float(delete_regrets.mean())
                        if (
                            not np.isnan(gamma_delete_threshold)
                            and gamma_delete_threshold > 0
                        ):
                            gamma_pass_delete = (
                                avg_regret_delete_only <= gamma_delete_threshold
                            )

                # Store γ-adherence metrics in summary row
                summary_row["avg_regret_final"] = avg_regret_final
                summary_row["gamma_bar_threshold"] = gamma_bar
                summary_row["gamma_split_threshold"] = gamma_split
                summary_row["gamma_insert_threshold"] = gamma_insert_threshold
                summary_row["gamma_delete_threshold"] = gamma_delete_threshold
                summary_row["gamma_pass_overall"] = gamma_pass_overall
                summary_row["gamma_pass_insert"] = gamma_pass_insert
                summary_row["gamma_pass_delete"] = gamma_pass_delete
                summary_row["gamma_error"] = gamma_error

                # Add AT-γ blocked reason if overall check fails
                if (
                    not gamma_pass_overall
                    and not np.isnan(avg_regret_final)
                    and not np.isnan(gamma_bar)
                ):
                    prev_reason = summary_row.get("blocked_reason", "") or ""
                    at_gamma_reason = (
                        f"AT-γ avg regret {avg_regret_final:.3g} > γ̄ {gamma_bar:.3g}"
                    )
                    summary_row["blocked_reason"] = (
                        prev_reason + ("; " if prev_reason else "") + at_gamma_reason
                    )

            except Exception:
                # Ensure columns exist even if computation fails
                summary_row["avg_regret_final"] = np.nan
                summary_row["gamma_bar_threshold"] = mandatory_fields.get(
                    "gamma_bar", np.nan
                )
                summary_row["gamma_split_threshold"] = mandatory_fields.get(
                    "gamma_split", np.nan
                )
                summary_row["gamma_insert_threshold"] = np.nan
                summary_row["gamma_delete_threshold"] = np.nan
                summary_row["gamma_pass_overall"] = False
                summary_row["gamma_pass_insert"] = np.nan
                summary_row["gamma_pass_delete"] = np.nan
                summary_row["gamma_error"] = np.nan

            # Write seed summary file (CSV)
            if not no_legacy_csv:
                seed_output_file = os.path.join(output_dir, f"seed_{seed:03d}.csv")
                pd.DataFrame([summary_row]).to_csv(seed_output_file, index=False)
                processed_files.append(seed_output_file)
            
            # Collect for Parquet writing
            seed_summaries.append(summary_row)

        except Exception as e:
            print(f"Warning: Failed to process {csv_file} for seed output: {e}")

    # Write Parquet for all seed summaries
    if seed_summaries and params_with_grid:
        try:
            write_seed_summary_parquet(seed_summaries, parquet_out, params_with_grid)
            print(f"✓ Written {len(seed_summaries)} seed summaries to Parquet: {parquet_out}/seeds/")
        except Exception as e:
            print(f"Warning: Failed to write seed Parquet: {e}")

    return processed_files


def process_event_output(
    csv_files: List[str],
    grid_id: str,
    output_dir: str,
    mandatory_fields: Dict[str, Any],
    parquet_out: str = "results_parquet",
    params_with_grid: Dict[str, Any] = None,
    parquet_write_events: bool = False,
    no_legacy_csv: bool = False,
) -> List[str]:
    """Process CSV files for event granularity output."""
    processed_files = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue

            # Extract seed from filename
            seed_file = os.path.basename(csv_file)
            seed_match = [part for part in seed_file.split("_") if part.isdigit()]
            if not seed_match:
                continue
            seed = int(seed_match[0])

            # Add metadata columns
            df["seed"] = seed
            df["grid_id"] = grid_id

            # Ensure ALL mandatory fields are present in every row, even if NaN
            mandatory_field_names = [
                "gamma_bar",
                "gamma_split",
                "accountant",
                "G_hat",
                "D_hat",
                "c_hat",
                "C_hat",
                "lambda_est",
                "S_scalar",
                "sigma_step_theory",
                "N_star_live",
                "m_theory_live",
                "blocked_reason",
                # Drift/scale/feature controls
                "path_type",
                "rotate_angle",
                "drift_rate",
                "feature_scale",
                "w_scale",
                "fix_w_norm",
                "noise_std",
                # Theory-first targets
                "target_G",
                "target_D",
                "target_c",
                "target_C",
                "target_lambda",
                "target_PT",
                "target_ST",
                "rho_total",
                "path_style",
            ]

            for field in mandatory_field_names:
                if field not in df.columns:
                    df[field] = mandatory_fields.get(field, np.nan)
                else:
                    # Fill NaN values with mandatory field values if available
                    if field in mandatory_fields:
                        df[field] = df[field].fillna(mandatory_fields[field])

            # Ensure blocked_reason is a string
            if "blocked_reason" in df.columns:
                df["blocked_reason"] = df["blocked_reason"].fillna("")

            # Add event_type column if not present (derive from 'op' or 'event')
            if "event_type" not in df.columns:
                if "op" in df.columns:
                    df["event_type"] = df["op"]
                else:
                    df["event_type"] = "unknown"

            # Write event-level file (CSV)
            if not no_legacy_csv:
                event_output_file = os.path.join(output_dir, f"seed_{seed:03d}_events.csv")
                df.to_csv(event_output_file, index=False)
                processed_files.append(event_output_file)
            
            # Write events to Parquet if enabled
            if parquet_write_events and params_with_grid and not df.empty:
                try:
                    # Process in chunks to avoid memory issues
                    for chunk in iter_df_in_chunks(df, 50000):
                        event_rows = (row._asdict() if hasattr(row, "_asdict") else row.to_dict() 
                                     for _, row in chunk.iterrows())
                        write_event_rows_parquet(event_rows, parquet_out, params_with_grid)
                    print(f"✓ Written {len(df)} events for seed {seed} to Parquet")
                except Exception as e:
                    print(f"Warning: Failed to write events for seed {seed} to Parquet: {e}")

        except Exception as e:
            print(f"Warning: Failed to process {csv_file} for event output: {e}")

    return processed_files


def process_aggregate_output(
    csv_files: List[str],
    grid_id: str,
    output_dir: str,
    mandatory_fields: Dict[str, Any],
    parquet_out: str = "results_parquet",
    params_with_grid: Dict[str, Any] = None,
    no_legacy_csv: bool = False,
) -> Optional[str]:
    """Process CSV files for aggregate granularity output."""
    if not csv_files:
        return None

    all_summaries = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue

            # Extract seed from filename
            seed_file = os.path.basename(csv_file)
            seed_match = [part for part in seed_file.split("_") if part.isdigit()]
            if not seed_match:
                continue
            seed = int(seed_match[0])

            # Create per-seed summary
            summary = {
                "seed": seed,
                "avg_regret_empirical": df["regret"].mean()
                if "regret" in df.columns
                else np.nan,
                "N_star_emp": len(df[df["op"] == "insert"])
                if "op" in df.columns
                else np.nan,
                "m_emp": len(df[df["op"] == "delete"])
                if "op" in df.columns
                else np.nan,
                "final_acc": df["acc"].iloc[-1]
                if "acc" in df.columns and len(df) > 0
                else np.nan,
                "total_events": len(df),
            }

            # Add privacy metrics from last row
            if len(df) > 0:
                last_row = df.iloc[-1]
                for col in [
                    "eps_spent",
                    "capacity_remaining",
                    "eps_converted",
                    "eps_remaining",
                    "delta_total",
                ]:
                    if col in last_row:
                        summary[col] = last_row[col]

            all_summaries.append(summary)

        except Exception as e:
            print(f"Warning: Failed to process {csv_file} for aggregate: {e}")

    if not all_summaries:
        return None

    # Create aggregate statistics
    summary_df = pd.DataFrame(all_summaries)

    aggregate_row = {
        "grid_id": grid_id,
        "num_seeds": len(all_summaries),
        "avg_regret_mean": summary_df["avg_regret_empirical"].mean(),
        "avg_regret_median": summary_df["avg_regret_empirical"].median(),
        "avg_regret_std": summary_df["avg_regret_empirical"].std(),
        "N_star_mean": summary_df["N_star_emp"].mean(),
        "N_star_median": summary_df["N_star_emp"].median(),
        "m_mean": summary_df["m_emp"].mean(),
        "m_median": summary_df["m_emp"].median(),
        "final_acc_mean": summary_df["final_acc"].mean(),
        "final_acc_median": summary_df["final_acc"].median(),
    }

    # Ensure ALL mandatory fields are present, even if NaN
    mandatory_field_names = [
        "gamma_bar",
        "gamma_split",
        "accountant",
        "G_hat",
        "D_hat",
        "c_hat",
        "C_hat",
        "lambda_est",
        "S_scalar",
        "sigma_step_theory",
        "N_star_live",
        "m_theory_live",
        "blocked_reason",
        # Drift/scale/feature controls
        "path_type",
        "rotate_angle",
        "drift_rate",
        "feature_scale",
        "w_scale",
        "fix_w_norm",
        "noise_std",
        # Theory-first targets
        "target_G",
        "target_D",
        "target_c",
        "target_C",
        "target_lambda",
        "target_PT",
        "target_ST",
        "rho_total",
        "path_style",
    ]

    for field in mandatory_field_names:
        if field in mandatory_fields:
            aggregate_row[field] = mandatory_fields[field]
        else:
            aggregate_row[field] = np.nan if field != "blocked_reason" else ""

    # Add privacy metrics aggregates
    for col in [
        "eps_spent",
        "capacity_remaining",
        "eps_converted",
        "eps_remaining",
        "delta_total",
    ]:
        if col in summary_df.columns:
            aggregate_row[f"{col}_mean"] = summary_df[col].mean()
            aggregate_row[f"{col}_median"] = summary_df[col].median()

    # Write aggregate file (CSV)
    if not no_legacy_csv:
        aggregate_output_file = os.path.join(output_dir, "aggregate.csv")
        pd.DataFrame([aggregate_row]).to_csv(aggregate_output_file, index=False)
    else:
        aggregate_output_file = None
    
    # Write aggregate to Parquet (as a seed summary with aggregate=True flag)
    if params_with_grid:
        try:
            aggregate_row_parquet = aggregate_row.copy()
            aggregate_row_parquet["is_aggregate"] = True
            write_seed_summary_parquet([aggregate_row_parquet], parquet_out, params_with_grid)
            print(f"✓ Written aggregate summary to Parquet: {parquet_out}/seeds/")
        except Exception as e:
            print(f"Warning: Failed to write aggregate Parquet: {e}")

    return aggregate_output_file


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
        default = True,
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

    all_csv_paths = []
    for i, params in enumerate(combinations):
        print(f"\nProgress: {i + 1}/{len(combinations)}")
        csv_paths = run_parameter_combination(
            params, seeds, args.base_out, args.output_granularity, args.parallel,
            args.parquet_out, args.parquet_write_events, args.no_legacy_csv
        )
        all_csv_paths.extend(csv_paths)

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
    print(f"Total output files completed: {len(all_csv_paths)}")
    print(f"Output granularity used: {args.output_granularity}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
