from typing import Any
from .types import Accountant

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
        # For now, map relaxed to eps_delta with modified parameters
        # This could be a separate implementation in the future
        from .eps_delta import Adapter
        return Adapter(**kwargs)
    else:
        raise ValueError(f"Unknown accountant type: {name}")