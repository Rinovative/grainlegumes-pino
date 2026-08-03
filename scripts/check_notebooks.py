"""Validate every discovered maintained notebook without executing or rewriting it."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_ROOT = _REPOSITORY_ROOT / "model_training" / "notebooks"
_NOTEBOOK_FORMAT = 4


def discover_notebooks(root: Path = _NOTEBOOK_ROOT) -> tuple[Path, ...]:
    """Return every notebook in a maintained directory and reject an empty inventory."""
    paths = tuple(sorted(path for path in root.glob("*.ipynb") if path.is_file()))
    if not paths:
        message = f"Maintained notebook directory contains no notebooks: {root}"
        raise FileNotFoundError(message)
    return paths


def _notebook_cells(path: Path) -> Sequence[Mapping[str, Any]]:
    """
    Return structurally usable nbformat-4 cells from one notebook JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Notebook JSON to parse without mutation.

    Returns
    -------
    collections.abc.Sequence[collections.abc.Mapping[str, Any]]
        Cell mappings in persisted order.

    Raises
    ------
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If the root is not a notebook-format-4 object.
    TypeError
        If ``cells`` is not a list containing only mappings.

    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("nbformat") != _NOTEBOOK_FORMAT:
        message = f"{path}: expected a notebook-format-4 JSON object."
        raise ValueError(message)
    cells = payload.get("cells")
    if not isinstance(cells, list) or not all(isinstance(cell, Mapping) for cell in cells):
        message = f"{path}: cells must be a JSON array of cell objects."
        raise TypeError(message)
    return cells


def validate_notebook(path: Path) -> None:
    """Require runnable code cells with cleared execution state and valid cell kinds."""
    cells = _notebook_cells(path)
    code_cells: list[Mapping[str, Any]] = []
    for index, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            message = f"{path}: cell {index} has unsupported type {cell_type!r}."
            raise ValueError(message)
        if cell_type != "code":
            continue
        code_cells.append(cell)
        if cell.get("execution_count") is not None:
            message = f"{path}: code cell {index} must have a null execution count."
            raise ValueError(message)
        if cell.get("outputs") != []:
            message = f"{path}: code cell {index} must not contain saved outputs."
            raise ValueError(message)

    if not code_cells:
        message = f"{path}: a maintained notebook must contain runnable code."
        raise ValueError(message)


def main() -> int:
    """Validate every notebook currently maintained below the notebook root."""
    paths = discover_notebooks()
    for path in paths:
        validate_notebook(path)
    print(f"Validated {len(paths)} maintained notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
