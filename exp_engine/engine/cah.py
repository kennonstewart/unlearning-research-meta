from __future__ import annotations
import hashlib
import json
import re
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

# Regex pattern for path-like keys using word boundaries to avoid false positives
_PATH_LIKE_PATTERN = re.compile(
    r'(^|_)(path|paths|dir|directory|directories|folder|folders|file|filename|files)($|_)', 
    re.IGNORECASE
)


def _is_path_like_key(k: str) -> bool:
    """Check if a key is path-like using word boundaries to avoid false positives.
    
    This avoids incorrectly flagging keys like 'profile' (contains 'file') or 
    'coordinated' (contains 'dir') as path-like.
    """
    return bool(_PATH_LIKE_PATTERN.search(k))


def _round_float(v: float) -> float:
    """Round float to fixed precision and convert near-integers to clean integers.
    
    This ensures deterministic hashing by:
    1. Rounding to _FLOAT_PRECISION decimal places
    2. Converting values very close to integers (e.g., 1.0000000001) to clean integers
    
    Note: This means 1.0000000001 becomes 1.0, which may surprise some users.
    """
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
    
    Uses SHA-256 and returns a 12-character hex prefix to reduce collision 
    probability in long-running research projects.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        12-character hex hash of canonicalized parameters
    """
    canonical = canonicalize_params(params)
    json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))
    return hash_obj.hexdigest()[:12]


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