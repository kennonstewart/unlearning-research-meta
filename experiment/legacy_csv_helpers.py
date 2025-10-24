"""
Legacy CSV processing helpers for backward compatibility with existing tests.

This module extracts the CSV processing functions from the old grid_runner.py
to maintain compatibility with existing tests while the main execution logic
has been replaced by experiment/run.py.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional


def _ensure_mandatory_fields(df, mandatory_fields: Dict[str, Any]):
    """Ensure mandatory fields are present in DataFrame."""
    for k, v in (mandatory_fields or {}).items():
        if k not in df.columns or df[k].isna().all():
            df[k] = v
    return df


def _extract_seed_from_filename(path: str) -> Optional[int]:
    """Extract seed number from filename."""
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
    rows = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        df = _ensure_mandatory_fields(df, mandatory_fields or {})

        # Basic empirical metrics
        avg_regret_emp = (
            float(df["regret"].mean()) if "regret" in df.columns else float("nan")
        )
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
        for k in mandatory_fields or {}:
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
        "N_star_mean": float(seeds_df["N_star_emp"].mean())
        if "N_star_emp" in seeds_df.columns
        else float("nan"),
        "m_mean": float(seeds_df["m_emp"].mean())
        if "m_emp" in seeds_df.columns
        else float("nan"),
    }

    for k, v in (mandatory_fields or {}).items():
        if k not in agg:
            agg[k] = v

    out_df = pd.DataFrame([agg])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"aggregate_{grid_id}.csv")
    out_df.to_csv(out_path, index=False)
    return out_path
