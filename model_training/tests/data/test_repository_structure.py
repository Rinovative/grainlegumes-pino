# ruff: noqa: S101
"""Protect final generation-source ownership and the single builder command."""

from __future__ import annotations

from pathlib import Path

from src import common


def test_generation_builder_has_one_python_owner_and_one_module_command() -> None:
    project_root = common.paths.get_project_root()
    generation_root = project_root / "data_generation"
    builder = generation_root / "build_training_dataset.py"
    old_builder = generation_root / "matlab" / "build_batch_dataset.py"
    workflow = (project_root / ".github/workflows/quality.yml").read_text(encoding="utf-8")

    assert builder.is_file()
    assert not old_builder.exists()
    assert (generation_root / "__init__.py").is_file()
    assert not (generation_root / "matlab" / "__init__.py").exists()
    assert not list((generation_root / "matlab").glob("*.py"))
    assert workflow.count("PYTHONPATH=model_training python -m data_generation.build_training_dataset --help") == 1
    assert "data_generation.matlab." + "build_batch_dataset" not in workflow


def test_source_tree_contains_no_duplicate_dataset_builder_module() -> None:
    project_root = common.paths.get_project_root()
    candidates = [
        path
        for path in (project_root / "data_generation").rglob("*.py")
        if path.name in {"build_training_dataset.py", "build_batch_dataset.py"}
        and "__pycache__" not in path.parts
    ]
    assert candidates == [project_root / "data_generation/build_training_dataset.py"]


def test_generation_batch_lock_is_hidden_outside_scientific_stages() -> None:
    project_root = common.paths.get_project_root()
    batch_source = (project_root / "data_generation/matlab/batch_run.m").read_text(encoding="utf-8")
    assert "fullfile(generated_data_root, '.state', 'locks')" in batch_source
    legacy_stage = "processed"
    legacy_lock_directory = ".locks"
    legacy_expression = f"fullfile(generated_data_root, '{legacy_stage}', '{legacy_lock_directory}')"
    assert legacy_expression not in batch_source
