# ruff: noqa: S101
"""
Protect the maintained training notebook lifecycle and safety boundary.

Static JSON/code checks require cleared cells, current public pipeline services,
official launcher coverage, and false-by-default mutation controls. Native Ruff
and ``scripts/check_notebooks.py`` own repository-wide notebook lint/inventory;
notebook execution and real data, training, resume, or inference belong to manual
acceptance.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

_NOTEBOOK_ROOT = Path(__file__).resolve().parents[2] / "notebooks"
_TRAINING_NOTEBOOK = _NOTEBOOK_ROOT / "training_pipeline.ipynb"


def _payload(path: Path) -> dict[str, Any]:
    """
    Load one notebook as a JSON object for static contract checks.

    Non-object roots raise immediately so later cell assertions never operate on
    a structurally invalid notebook payload.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"Notebook must contain a JSON object: {path}"
        raise TypeError(message)
    return value


def _joined_source(payload: dict[str, Any]) -> str:
    """
    Join all Markdown and code source without assuming a cell count.

    The combined text supports repository-policy scans while leaving individual
    code cells available for syntax parsing in the dedicated test.
    """
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_training_notebook_is_cleared_and_mutation_safe_by_default() -> None:
    """
    Require cleared diagnostics and false-by-default mutation controls.

    The maintained training notebook must contain code yet persist no outputs,
    enable no expensive action, and embed no credential or host-specific path.
    """
    payload = _payload(_TRAINING_NOTEBOOK)
    code_cells = [cell for cell in payload["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None and cell.get("outputs") == [] for cell in code_cells)
    source = _joined_source(payload)
    for guard in (
        "RUN_BACKWARD_SMOKE = False",
        "RUN_TINY_TRAINING = False",
        "RUN_EXPLICIT_RESUME = False",
        "RUN_INFERENCE_SMOKE = False",
    ):
        assert guard in source
    assert "WANDB_API_KEY=" not in source
    assert "/home/" not in source
    assert "/Users/" not in source
    assert "C:\\Users\\" not in source


def test_training_notebook_uses_current_services_and_official_commands() -> None:
    """
    Require current services, recipes, and official launcher commands.

    Static source membership protects the notebook as an orientation surface for
    all maintained pipeline stages without executing training or inference.
    """
    source = _joined_source(_payload(_TRAINING_NOTEBOOK))
    required_apis = (
        "experiments.config.loader.create_dataloaders_from_config",
        "learning.models.factory.build_model",
        "learning.losses.factory.build_training_loss",
        "learning.training.loop.eval_one_epoch",
        "experiments.run.run_experiment",
        "experiments.run.validate_completed_run",
        "learning.inference.context.load_inference_context",
    )
    assert all(api in source for api in required_apis)
    for recipe in (
        "steady_flow_fno.yaml",
        "steady_flow_pifno.yaml",
        "steady_flow_uno.yaml",
        "steady_flow_piuno.yaml",
    ):
        assert recipe in source
    assert "scripts/docker_job.sh" in source
    assert "src.experiments.cli.cli_optuna" in source
    assert "src.experiments.cli.cli_build_artifacts" in source


def test_maintained_code_cells_parse_and_sensitivity_stays_archived() -> None:
    """
    Parse maintained code cells and enforce the archived-notebook boundary.

    Every maintained cell must be valid Python, while sensitivity remains a
    Markdown-only archival notice rather than a second executable workflow.
    """
    for name in (
        "eda.ipynb",
        "training_pipeline.ipynb",
        "eval_single_model.ipynb",
        "eval_comparison_models.ipynb",
    ):
        payload = _payload(_NOTEBOOK_ROOT / name)
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"{name}:cell-{index}")
    sensitivity = _payload(_NOTEBOOK_ROOT / "sensitivity.ipynb")
    assert sensitivity["cells"]
    assert all(cell.get("cell_type") == "markdown" for cell in sensitivity["cells"])
