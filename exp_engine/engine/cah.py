from __future__ import annotations
import hashlib
import json
from typing import Any, Dict

_VOLATILE_KEYS = {
    "base_out",
    "out_dir", 
    "output_dir",
    "results_dir",
    "timestamp",
    "hostname",
    "host",
    "user",
    "seed",
    "seeds",
    "commit_results",
    "figs_dir",
    "figs",
    "log_dir",
    "run_id",
}

_FLOAT_PRECISION = 10  # round floats to 10 decimal places deterministically


def _is_path_like_key(k: str) -> bool:
    kl = k.lower()
    return any(s in kl for s in ("path", "dir", "folder", "file"))


def _round_float(v: float) -> float:
    try:
        rounded = float(round(float(v), _FLOAT_PRECISION))
        # If rounding resulted in insignificant change, return clean integer if applicable
        if abs(rounded - round(rounded)) < 1e-10:
            return float(round(rounded))
        return rounded
    except Exception:
        return v  # leave as-is if not coercible


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted(obj.keys()):
            if k in _VOLATILE_KEYS or _is_path_like_key(k):
                continue
            v = obj[k]
            if isinstance(v, float):
                v = _round_float(v)
            elif isinstance(v, (list, tuple)):
                v = [_canonicalize(item) for item in v]
            elif isinstance(v, dict):
                v = _canonicalize(v)
            out[k] = v
        return out
    elif isinstance(obj, (list, tuple)):
        return [_canonicalize(item) for item in obj]
    elif isinstance(obj, float):
        return _round_float(obj)
    else:
        return obj


def canonicalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize parameters for content-addressed hashing.
    
    Removes volatile keys (paths, timestamps, etc.) and normalizes floats
    to ensure deterministic hashing across runs.
    
    Args:
        params: Raw parameter dictionary
        
    Returns:
        Canonicalized parameter dictionary
    """
    return _canonicalize(params)


def grid_hash(params: Dict[str, Any]) -> str:
    """Generate a content-addressed hash for grid parameters.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        8-character hex hash of canonicalized parameters
    """
    canonical = canonicalize_params(params)
    json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    hash_obj = hashlib.md5(json_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def attach_grid_id(params: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a grid_id to parameters based on content hash.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Parameters with grid_id added
    """
    result = params.copy()
    result['grid_id'] = grid_hash(params)
    return result