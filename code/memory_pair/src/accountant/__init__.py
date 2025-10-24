"""
zCDP accountant interface.

Simplified to only support zCDP accounting.
"""
from .types import Accountant
from .zcdp import Adapter as ZCDPAccountant

__all__ = ["Accountant", "ZCDPAccountant"]