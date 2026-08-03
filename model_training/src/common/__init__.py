"""
Shared filesystem and atomic-publication infrastructure.

Provides:
- locking: process-safe file and directory coordination
- paths: canonical repository and storage-path resolution
- serialization: deterministic serialization and atomic persistence
"""

from . import common_locking as locking
from . import common_paths as paths
from . import common_serialization as serialization

__all__ = ["locking", "paths", "serialization"]
