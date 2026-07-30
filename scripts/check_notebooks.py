"""
===============================================================================
check_notebooks.py
===============================================================================
Enforce repository-specific notebook lifecycle and cleared-state contracts.

Responsibilities:
  - Require the exact maintained and archived notebook inventory
  - Require maintained notebooks to contain cleared runnable code cells
  - Require notice-only notebooks to remain Markdown-only
  - Parse notebook-format-4 JSON without executing or rewriting notebooks

Design principles:
  - Validation is static, deterministic, read-only, and independent of cell counts
  - Notebook lifecycle roles are explicit rather than inferred from cell content

This module does NOT:
  - Execute notebooks or validate the scientific behavior of their workflows
  - Replace Ruff's syntax, import, and lint checks for notebook code cells
===============================================================================
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK_ROOT = _REPOSITORY_ROOT / "model_training" / "notebooks"
_NOTEBOOK_FORMAT = 4
_MAINTAINED_NOTEBOOKS = frozenset(
    {
        "eda.ipynb",
        "eval_single_model.ipynb",
        "eval_comparison_models.ipynb",
        "training_pipeline.ipynb",
    }
)
_ARCHIVED_NOTEBOOKS = frozenset({"sensitivity.ipynb"})
NotebookRole = Literal["maintained", "archived"]


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


def validate_notebook(path: Path, *, role: NotebookRole) -> None:
    """
    Validate one notebook's repository-specific release state.

    Parameters
    ----------
    path : pathlib.Path
        Notebook-format-4 JSON file to inspect without mutation.
    role : {"maintained", "archived"}
        Maintained notebooks must contain at least one cleared code cell.
        Archived notices must contain Markdown cells only.

    Raises
    ------
    json.JSONDecodeError
        If the file is not valid JSON. Ruff separately validates notebook code.
    TypeError
        If the minimal notebook/cell structure needed by this gate is malformed.
    ValueError
        If outputs/execution counts are saved, a maintained notebook has no code,
        or the archived notice contains a non-Markdown cell.

    Notes
    -----
    Python syntax and style deliberately remain Ruff's responsibility.

    """
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

    if role == "maintained" and not code_cells:
        message = f"{path}: a maintained notebook must contain runnable code."
        raise ValueError(message)
    if role == "archived" and any(cell.get("cell_type") != "markdown" for cell in cells):
        message = f"{path}: an archived notebook must be a Markdown-only notice."
        raise ValueError(message)


def main() -> int:
    """
    Validate the declared maintained and archived notebook contracts.

    Returns
    -------
    int
        Zero after all declared notebooks pass clearing and lifecycle-role checks.

    Raises
    ------
    FileNotFoundError
        If a declared notebook is absent or an undeclared notebook appears in the
        maintained notebook directory.

    """
    expected = _MAINTAINED_NOTEBOOKS | _ARCHIVED_NOTEBOOKS
    observed = {path.name for path in _NOTEBOOK_ROOT.glob("*.ipynb")}
    missing = sorted(expected - observed)
    undeclared = sorted(observed - expected)
    if missing or undeclared:
        message = f"Notebook lifecycle inventory mismatch: missing={missing}; undeclared={undeclared}."
        raise FileNotFoundError(message)

    for name in sorted(_MAINTAINED_NOTEBOOKS):
        validate_notebook(_NOTEBOOK_ROOT / name, role="maintained")
    for name in sorted(_ARCHIVED_NOTEBOOKS):
        validate_notebook(_NOTEBOOK_ROOT / name, role="archived")
    archive_count = len(_ARCHIVED_NOTEBOOKS)
    archive_label = "notice" if archive_count == 1 else "notices"
    print(f"Validated {len(_MAINTAINED_NOTEBOOKS)} maintained notebooks and {archive_count} archived {archive_label}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
