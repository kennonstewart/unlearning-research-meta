from typing import Any
from .types import Accountant

def get_adapter(name: str, **kwargs) -> Accountant:
    """Get accountant adapter by name."""
    if name == "eps_delta":
        from .eps_delta import Adapter
        return Adapter(**kwargs)
    elif name == "zcdp":
        from .zcdp import Adapter
        return Adapter(**kwargs)
    else:
        raise ValueError(f"Unknown accountant type: {name}")