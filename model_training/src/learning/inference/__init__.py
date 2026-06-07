"""Model reconstruction and inference context."""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load inference modules to avoid training.train_uno import."""
    if name == "learning_inference":
        return importlib.import_module(".learning_inference", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["learning_inference"]


__all__ = []
