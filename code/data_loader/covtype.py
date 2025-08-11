import os
import numpy as np
from typing import Optional, Iterator, Dict, Any

try:
    from .utils import set_global_seed
    from .streams import make_stream
    from .event_schema import create_event_record_with_diagnostics
except ImportError:
    from utils import set_global_seed
    from streams import make_stream
    from event_schema import create_event_record_with_diagnostics

try:
    from sklearn.datasets import fetch_covtype
except Exception:  # noqa: BLE001
    fetch_covtype = None


class OnlineStandardizer:
    """Online feature standardization using Welford's algorithm."""
    
    def __init__(self, d: int, clip_k: float = 3.0, eps: float = 1e-8):
        self.d = d
        self.clip_k = clip_k
        self.eps = eps
        
        # Welford statistics
        self.count = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros(d, dtype=np.float64)  # sum of squared differences
        
        # Diagnostics
        self.clip_count = 0
        
    def update_and_standardize(self, x: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
        """Update statistics and return standardized features with diagnostics."""
        self.count += 1
        
        # Welford's algorithm
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        # Compute current variance and std
        if self.count > 1:
            variance = self.M2 / (self.count - 1)
            std = np.sqrt(variance + self.eps)
        else:
            std = np.ones_like(self.mean)
            
        # Standardize
        x_std = (x - self.mean) / std
        
        # Clip to k-sigma
        clipped = np.clip(x_std, -self.clip_k, self.clip_k)
        
        # Track clipping rate
        if np.any(np.abs(x_std) > self.clip_k):
            self.clip_count += 1
            
        # Compute diagnostics
        diagnostics = {
            "mean_l2": float(np.linalg.norm(self.mean)),
            "std_l2": float(np.linalg.norm(std)),
            "clip_rate": self.clip_count / self.count if self.count > 0 else 0.0
        }
        
        return clipped, diagnostics


class LabelShiftTracker:
    """Track label distribution for label shift detection and reweighting."""
    
    def __init__(self, n_classes: int, window_size: int = 1000):
        self.n_classes = n_classes
        self.window_size = window_size
        
        # Sliding window of labels
        self.window = []
        self.segment_id = 0
        self.segment_length = 0
        
        # Prior schedule parameters
        self.prior_schedule = None
        
    def update(self, y: int) -> tuple[int, Optional[float]]:
        """Update label statistics and return segment_id and importance weight."""
        self.window.append(y)
        self.segment_length += 1
        
        # Keep sliding window
        if len(self.window) > self.window_size:
            self.window.pop(0)
            
        # Simple segment change detection: every 1000 samples
        if self.segment_length >= 1000:
            self.segment_id += 1
            self.segment_length = 0
            
        # For now, return uniform weights (importance weighting can be added later)
        importance_weight = 1.0
        
        return self.segment_id, importance_weight


def _simulate_covtype(n=581012, d=54, seed=42):
    set_global_seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(1, 8, size=n, dtype=np.int64)
    return X, y


def download_covtype(data_dir: str):
    if fetch_covtype is None:
        return _simulate_covtype()
    try:
        ds = fetch_covtype(data_home=data_dir)
        X = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        return X, y
    except Exception:
        return _simulate_covtype()


def get_covtype_stream(
    mode="iid", 
    batch_size=1, 
    seed=42, 
    use_event_schema=True,
    online_standardize=False,
    clip_k=3.0,
    label_shift_window=1000
):
    """
    Get CovType stream with optional online standardization and label shift tracking.
    
    Args:
        mode: Stream mode (iid, drift, etc.)
        batch_size: Batch size (currently only 1 supported)
        seed: Random seed
        use_event_schema: Whether to emit event records
        online_standardize: Whether to apply online standardization
        clip_k: Clipping parameter for standardization (k-sigma)
        label_shift_window: Window size for label shift detection
    """
    X, y = download_covtype(os.path.expanduser("~/.cache/memory_pair_data"))
    
    if use_event_schema:
        # Wrap the stream to emit event records with optional preprocessing
        base_stream = make_stream(X, y, mode=mode, seed=seed)
        return _wrap_stream_with_preprocessing(
            base_stream, 
            X.shape[1], 
            online_standardize, 
            clip_k, 
            label_shift_window
        )
    else:
        # Legacy behavior
        return make_stream(X, y, mode=mode, seed=seed)


def _wrap_stream_with_preprocessing(
    base_stream: Iterator,
    d: int,
    online_standardize: bool,
    clip_k: float,
    label_shift_window: int
):
    """Wrap stream with preprocessing and enhanced event records."""
    event_id = 0
    
    # Initialize processors
    standardizer = OnlineStandardizer(d, clip_k) if online_standardize else None
    label_tracker = LabelShiftTracker(7, label_shift_window)  # CovType has 7 classes
    
    for x, y in base_stream:
        # Apply online standardization if enabled
        if standardizer is not None:
            x_processed, std_diagnostics = standardizer.update_and_standardize(x)
        else:
            x_processed = x
            std_diagnostics = {}
            
        # Track label shift
        segment_id, importance_weight = label_tracker.update(int(y))
        
        # Create sample ID
        sample_id = f"covtype_{hash((x.tobytes(), float(y))) % 1000000:06d}"
        
        # Create event record with diagnostics
        yield create_event_record_with_diagnostics(
            x=x_processed,
            y=y,
            sample_id=sample_id,
            event_id=event_id,
            segment_id=segment_id,
            metrics={"importance_weight": importance_weight},
            **std_diagnostics
        )
        event_id += 1
