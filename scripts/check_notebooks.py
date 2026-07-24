"""Validate repository notebooks without executing or rewriting them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_ROOT = _REPOSITORY_ROOT / "model_training" / "notebooks"
_CELL_TYPES = {"code", "markdown", "raw"}
_NOTEBOOK_FORMAT = 4


def _cell_source(cell: dict[str, Any], *, path: Path, index: int) -> str:
    """Return one cell source after validating its JSON representation."""
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    message = f"{path}: cell {index} source must be a string or a list of strings."
    raise TypeError(message)


def validate_notebook(path: Path) -> int:
    """Validate structure, cleared outputs, execution counts, and Python syntax."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{path}: notebook root must be a JSON object."
        raise TypeError(message)
    if payload.get("nbformat") != _NOTEBOOK_FORMAT:
        message = f"{path}: only notebook format 4 is maintained."
        raise ValueError(message)
    if not isinstance(payload.get("metadata"), dict):
        message = f"{path}: notebook metadata must be a JSON object."
        raise TypeError(message)

    cells = payload.get("cells")
    if not isinstance(cells, list):
        message = f"{path}: cells must be a JSON array."
        raise TypeError(message)

    code_cell_count = 0
    for index, raw_cell in enumerate(cells):
        if not isinstance(raw_cell, dict):
            message = f"{path}: cell {index} must be a JSON object."
            raise TypeError(message)
        cell_type = raw_cell.get("cell_type")
        if cell_type not in _CELL_TYPES:
            message = f"{path}: cell {index} has unsupported type {cell_type!r}."
            raise ValueError(message)
        if not isinstance(raw_cell.get("metadata"), dict):
            message = f"{path}: cell {index} metadata must be a JSON object."
            raise TypeError(message)
        source = _cell_source(raw_cell, path=path, index=index)
        if cell_type != "code":
            continue

        code_cell_count += 1
        if raw_cell.get("execution_count") is not None:
            message = f"{path}: cell {index} must have a null execution count."
            raise ValueError(message)
        if raw_cell.get("outputs") != []:
            message = f"{path}: cell {index} must not contain saved outputs."
            raise ValueError(message)
        compile(source, f"{path}:cell {index}", "exec")

    return code_cell_count


def main() -> int:
    """Validate every tracked notebook under the maintained notebook root."""
    notebook_paths = sorted(_NOTEBOOK_ROOT.glob("*.ipynb"))
    if not notebook_paths:
        message = f"No notebooks found under {_NOTEBOOK_ROOT}."
        raise FileNotFoundError(message)

    code_cell_count = sum(validate_notebook(path) for path in notebook_paths)
    print(f"Validated {len(notebook_paths)} notebooks ({code_cell_count} code cells).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
