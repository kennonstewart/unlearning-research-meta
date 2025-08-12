"""
Unified accountant factory and adapter interface.

Provides factory function for creating privacy accountant adapters that conform
to the unified Accountant protocol, supporting eps-delta DP, zCDP, and future extensions.
"""
from typing import Any
from .types import Accountant

__all__ = ['get_adapter']

def get_adapter(name: str, **kwargs) -> Accountant:
    """Get accountant adapter by name."""
    # Handle backward compatibility
    if name == "default":
        name = "eps_delta"
    
    if name == "eps_delta":
        from .eps_delta import Adapter
        return Adapter(**kwargs)
    elif name == "zcdp":
        from .zcdp import Adapter
        return Adapter(**kwargs)
    elif name == "relaxed":
        # TODO: For now, map relaxed to eps_delta as a temporary shim.
        # Future implementation will differ with altered regret gate or different bound constants.
        from .eps_delta import Adapter
        return Adapter(**kwargs)
    else:
        raise ValueError(f"Unknown accountant type: {name}")