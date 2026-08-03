# ruff: noqa: S101, S102
"""Protect dynamic notebook discovery, role boundaries, and safe training control flow."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import re
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from IPython import display as ipython_display
from src import experiments
from src.analysis.eda import eda_dataframe

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_NOTEBOOK_ROOT = _REPOSITORY_ROOT / "model_training" / "notebooks"
_TRAINING_NOTEBOOK = _NOTEBOOK_ROOT / "training_pipeline.ipynb"
_CHECKER = _REPOSITORY_ROOT / "scripts" / "check_notebooks.py"
_MIN_CATEGORIZED_EXPERIMENT_PARTS = 4


def _payload(path: Path) -> dict[str, Any]:
    """Load one notebook as a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        message = f"Notebook root must be an object: {path}"
        raise TypeError(message)
    return value


def _code_sources(path: Path) -> tuple[str, ...]:
    """Return code-cell sources in persisted order."""
    return tuple("".join(cell.get("source", [])) for cell in _payload(path)["cells"] if cell.get("cell_type") == "code")


def _qualified_name(node: ast.expr) -> str:
    """Return a dotted spelling for one call target when statically available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _call_names(path: Path) -> set[str]:
    """Collect statically qualified call targets across one notebook."""
    names: set[str] = set()
    for index, source in enumerate(_code_sources(path)):
        tree = ast.parse(source, filename=f"{path.name}:cell-{index}")
        names.update(name for node in ast.walk(tree) if isinstance(node, ast.Call) and (name := _qualified_name(node.func)))
    return names


def test_every_discovered_notebook_is_cleared_parseable_and_checker_owned(
    tmp_path: Path,
) -> None:
    """Validate every discovered notebook and the checker's principal failure paths."""
    checker = runpy.run_path(str(_CHECKER), run_name="notebook_checker")
    discover = checker["discover_notebooks"]
    validate = checker["validate_notebook"]
    paths = tuple(sorted(_NOTEBOOK_ROOT.glob("*.ipynb")))

    assert paths
    assert discover() == paths
    for path in paths:
        validate(path)
        for index, source in enumerate(_code_sources(path)):
            ast.parse(source, filename=f"{path.name}:cell-{index}")

    invalid = copy.deepcopy(_payload(_TRAINING_NOTEBOOK))
    code_cell = next(cell for cell in invalid["cells"] if cell.get("cell_type") == "code")
    code_cell["execution_count"] = 1
    invalid_path = tmp_path / "saved-output.ipynb"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="null execution count"):
        validate(invalid_path)
    with pytest.raises(FileNotFoundError, match="contains no notebooks"):
        discover(tmp_path / "empty")


def test_notebook_roles_respect_data_and_workload_boundaries() -> None:
    """Keep EDA on generated data, evaluation on training data, and training read-only."""
    paths = tuple(sorted(_NOTEBOOK_ROOT.glob("*.ipynb")))
    calls = {path.name: _call_names(path) for path in paths}
    sources = {path.name: "\n".join(_code_sources(path)) for path in paths}

    eda_calls = calls["eda.ipynb"]
    assert {
        "common.paths.get_generated_data_root",
        "common.paths.get_generation_meta_root",
        "common.paths.get_generation_raw_root",
        "common.paths.get_generation_processed_root",
    } <= eda_calls
    assert not any("get_model_training_data_root" in name for name in eda_calls)
    progress = inspect.signature(eda_dataframe.generate_eda_dataframe).parameters["show_progress"]
    assert progress.default is True
    assert "show_progress" not in sources["eda.ipynb"]

    for name in ("eval_single_model.ipynb", "eval_comparison_models.ipynb"):
        role_calls = calls[name]
        assert {
            "common.paths.get_model_training_data_root",
            "common.paths.get_training_meta_root",
            "common.paths.get_training_raw_root",
            "common.paths.get_training_processed_root",
        } <= role_calls
        assert not any("get_generated_data_root" in call or "get_generation_" in call for call in role_calls)

    training_calls = calls["training_pipeline.ipynb"]
    assert "common.paths.get_model_training_data_root" in training_calls
    assert not any("get_generated_data_root" in call or "get_generation_" in call for call in training_calls)
    forbidden_training_calls = {
        "experiments.run.run_experiment",
        "experiments.config.loader.create_dataloaders_from_config",
        "learning.models.factory.build_model",
        "learning.losses.factory.build_training_loss",
        "analysis.artifacts.service.build_artifacts",
        "torch.load",
        "torch.save",
        "wandb.init",
    }
    assert forbidden_training_calls.isdisjoint(training_calls)
    assert not any(call.endswith((".mkdir", ".write_text", ".write_bytes")) for call in training_calls)

    for source in sources.values():
        assert "sys.path" not in source
        assert "os.chdir" not in source
        assert "WANDB_API_KEY=" not in source


def test_training_notebook_runs_top_to_bottom_without_workloads_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the default training control plane while heavy validation stays unreachable."""
    training_root = tmp_path / "training-data"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    displayed: list[object] = []
    monkeypatch.setattr(ipython_display, "display", displayed.append)

    validator_calls: list[object] = []

    def unexpected_validator(config: object) -> object:
        validator_calls.append(config)
        message = "default notebook execution reached the heavy validator"
        raise AssertionError(message)

    monkeypatch.setattr(
        experiments.validation.data_pipeline,
        "validate_full_data_pipeline",
        unexpected_validator,
    )
    config_paths = tuple(sorted((_REPOSITORY_ROOT / "model_training/configs").rglob("*.yaml")))
    before = {path: path.read_bytes() for path in config_paths}
    namespace: dict[str, Any] = {}
    for index, source in enumerate(_code_sources(_TRAINING_NOTEBOOK)):
        exec(compile(source, filename=f"training_pipeline.ipynb:cell-{index}", mode="exec"), namespace)

    assert namespace["CONTEXT"] is not None
    assert namespace["RUN_FULL_DATA_VALIDATION"] is False
    assert namespace["EXPERIMENT_CONFIGS"]
    discovered = set(namespace["EXPERIMENT_CONFIGS"].values())
    expected = {
        path
        for experiments_root in namespace["TASK_CONFIG_ROOT"].glob("*/experiments")
        for path in experiments_root.rglob("*.yaml")
        if path.is_file()
    }
    assert discovered == expected
    assert all(len(path.relative_to(namespace["TASK_CONFIG_ROOT"]).parts) >= _MIN_CATEGORIZED_EXPERIMENT_PARTS for path in discovered)
    steady_flow_categories = {
        path.relative_to(namespace["TASK_CONFIG_ROOT"]).parts[2]
        for path in discovered
        if path.relative_to(namespace["TASK_CONFIG_ROOT"]).parts[0] == "steady_flow"
    }
    assert steady_flow_categories == {"capacity_and_physics", "best_of_class", "model_selection"}
    assert displayed
    assert validator_calls == []
    assert not training_root.exists()
    assert {path: path.read_bytes() for path in config_paths} == before

    namespace["CONTEXT"] = None
    with pytest.raises(RuntimeError, match="Run Step 5"):
        namespace["require_context"]()

    context_source = next(source for source in _code_sources(_TRAINING_NOTEBOOK) if "prepare_notebook_context(CONFIG_PATH)" in source)

    class PreparationError(RuntimeError):
        """Signal the synthetic context preparation failure."""

    failure_message = "synthetic config failure"

    def fail_preparation(_path: Path) -> None:
        raise PreparationError(failure_message)

    failure_namespace = {
        "CONTEXT": "stale",
        "CONFIG_PATH": Path("selected.yaml"),
        "experiments": SimpleNamespace(
            notebook_support=SimpleNamespace(prepare_notebook_context=fail_preparation),
        ),
    }
    with pytest.raises(PreparationError, match="synthetic config failure"):
        exec(
            compile(context_source, filename="training_pipeline.ipynb:context", mode="exec"),
            failure_namespace,
        )
    assert failure_namespace["CONTEXT"] is None


def test_training_markdown_references_importable_clis_and_existing_configs() -> None:
    """Keep terminal examples current without freezing notebook cell shape or inventory."""
    payload = _payload(_TRAINING_NOTEBOOK)
    markdown = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "markdown")
    references = set(
        re.findall(
            r"(?:model_training/)?configs/[A-Za-z0-9_.\-/]+\.yaml",
            markdown,
        ),
    )
    assert references
    for reference in references:
        candidate = _REPOSITORY_ROOT / reference
        if reference.startswith("configs/"):
            candidate = _REPOSITORY_ROOT / "model_training" / reference
        assert candidate.is_file(), reference

    modules = set(re.findall(r"python -m\s+([A-Za-z0-9_.]+)", markdown))
    assert modules
    assert all(importlib.util.find_spec(module) is not None for module in modules)
    assert "WANDB_API_KEY=" not in markdown
    assert "eval_single_model.ipynb" in markdown
    assert "eval_comparison_models.ipynb" in markdown
    assert "```bash" not in "\n".join(_code_sources(_TRAINING_NOTEBOOK))


def test_training_full_data_validation_is_one_default_off_opt_in_dispatch() -> None:
    """Skip the heavy validator by default and delegate exactly once when enabled."""
    matches = [source for source in _code_sources(_TRAINING_NOTEBOOK) if "validate_full_data_pipeline" in source]
    assert len(matches) == 1
    source = matches[0]
    assignment = "RUN_FULL_DATA_VALIDATION = False"
    assert source.startswith(assignment)

    shown: list[object] = []
    default_namespace = {
        "Markdown": lambda value: value,
        "show": shown.append,
    }
    exec(
        compile(source, filename="training_pipeline.ipynb:validation-default", mode="exec"),
        default_namespace,
    )
    assert len(shown) == 1

    context = SimpleNamespace(official_config={"task": "steady_flow"})
    result = object()
    presentation = SimpleNamespace(tables=("overall", "membership"), conclusion="PASS")
    validation_calls: list[object] = []
    presentation_calls: list[object] = []
    displayed_tables: list[object] = []

    def validate(config: object) -> object:
        validation_calls.append(config)
        return result

    def prepare(value: object) -> object:
        presentation_calls.append(value)
        return presentation

    enabled = source.replace(assignment, "RUN_FULL_DATA_VALIDATION = True", 1)
    namespace = {
        "Markdown": lambda value: value,
        "show": shown.append,
        "show_tables": lambda *tables: displayed_tables.extend(tables),
        "require_context": lambda: context,
        "experiments": SimpleNamespace(
            validation=SimpleNamespace(
                data_pipeline=SimpleNamespace(validate_full_data_pipeline=validate),
            ),
            notebook_support=SimpleNamespace(prepare_validation_presentation=prepare),
        ),
    }
    shown.clear()
    exec(
        compile(enabled, filename="training_pipeline.ipynb:validation-enabled", mode="exec"),
        namespace,
    )

    assert validation_calls == [context.official_config]
    assert presentation_calls == [result]
    assert displayed_tables == list(presentation.tables)
    assert shown == [presentation.conclusion]
