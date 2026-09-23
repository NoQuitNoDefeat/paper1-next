"""环境与执行后端: authoritative packet state and plan execution."""

from .base import BACKEND, Backend
from . import lightweight  # noqa: F401  (register implementations)

__all__ = ["BACKEND", "Backend"]
