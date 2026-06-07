"""Executable CLI entry points for experiments."""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load CLI modules to avoid import-time side effects."""
    if name == "cli_build_artifacts":
        return importlib.import_module(f"{__name__}.cli_build_artifacts")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["cli_build_artifacts"]


__all__ = []
