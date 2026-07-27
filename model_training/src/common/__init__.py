"""
Shared filesystem path, locking, and publication contracts.

Provides:
- locking: process-safe advisory file-lock contexts
- paths: logical storage and saved-run path resolution
- serialization: atomic file publication and stable SHA-256 helpers
"""

from . import common_locking as locking
from . import common_paths as paths
from . import common_serialization as serialization

__all__ = [
    "locking",
    "paths",
    "serialization",
]
