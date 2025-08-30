"""
Unified accountant factory and adapter interface.

Now zCDP-only. "default" maps to "zcdp".
"""
from typing import Any
from .types import Accountant

__all__ = ["get_adapter"]

def get_adapter(name: str, **kwargs) -> Accountant:
    name = (name or "zcdp").lower()
    if name in ("default", "zcdp"):
        from .zcdp import Adapter
        return Adapter(**kwargs)
    raise ValueError(f"Unknown accountant type: {name} (only 'zcdp' is supported)")