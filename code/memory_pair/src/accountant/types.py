"""
Accountant interface types and protocols.

Defines the unified Accountant protocol that abstracts privacy accounting
across different privacy models (eps-delta, zCDP, etc.).
"""
from typing import Protocol, Tuple, Dict, Any, Optional

class Accountant(Protocol):
    def finalize(self, stats: Dict[str, float], T_estimate: int) -> None: ...
    def ready(self) -> bool: ...
    def pre_delete(self, sensitivity: float) -> Tuple[bool, Optional[float], Optional[str]]:
        """Return (ok, sigma, reason). If not ok, reason is e.g. 'privacy_gate'."""
    def spend(self, sensitivity: float, sigma: float) -> None: ...
    def metrics(self) -> Dict[str, Any]: ...