# ruff: noqa: S101, SLF001
"""Protect lightweight, read-only notebook support services."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from src import common, datasets, experiments
from support import configs

if TYPE_CHECKING:
    from pathlib import Path

support = experiments.notebook_support


def test_prepare_context_uses_resolved_synthetic_dataset_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a test-owned config and delegate each dataset preview once."""
    config_path = configs.write_yaml(
        tmp_path / "direct.yaml",
        configs.direct_config(),
    )
    requests: list[dict[str, Any]] = []

    def preview(**request: Any) -> support.DatasetPreview:
        requests.append(request)
        return support.DatasetPreview(
            role=request["role"],
            dataset_id=request["dataset_id"],
            path=request["dataset_root"] / f"{request['dataset_id']}.pt",
            exists=False,
            sample_count=None,
            fingerprint=None,
            metadata_validated=False,
        )

    monkeypatch.setattr(support, "_dataset_preview", preview)

    context = support.prepare_notebook_context(config_path)

    assert isinstance(context, support.NotebookContext)
    assert context.config_path == config_path
    assert context.task.id == "steady_flow"
    assert [request["dataset_id"] for request in requests] == [
        "synthetic_train",
        "synthetic_ood",
    ]
    assert tuple(preview.dataset_id for preview in context.dataset_previews) == (
        "synthetic_train",
        "synthetic_ood",
    )


def test_dataset_preview_distinguishes_absent_and_invalid_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat absence as inspectable while propagating mounted-package corruption."""
    dataset_root = tmp_path / "datasets"
    metadata_root = tmp_path / "metadata"
    dataset_id = "synthetic_preview"
    dataset_path = common.paths.resolve_dataset_path(
        dataset_id,
        dataset_root=dataset_root,
    )
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"not-loaded")
    task: Any = SimpleNamespace(id="steady_flow")

    monkeypatch.setattr(
        datasets.metadata,
        "load_dataset_metadata_summary",
        lambda *_args, **_kwargs: pytest.fail("absent metadata must not be loaded"),
    )
    absent = support._dataset_preview(
        role="ID",
        dataset_id=dataset_id,
        task=task,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
    )
    assert absent.exists is True
    assert absent.metadata_validated is False

    metadata_directory = common.paths.resolve_dataset_metadata_dir(
        dataset_id,
        metadata_root=metadata_root,
    )
    metadata_directory.mkdir(parents=True)

    invalid_message = "invalid synthetic metadata package"

    def invalid_summary(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(invalid_message)

    monkeypatch.setattr(
        datasets.metadata,
        "load_dataset_metadata_summary",
        invalid_summary,
    )
    with pytest.raises(ValueError, match="invalid synthetic metadata package"):
        support._dataset_preview(
            role="ID",
            dataset_id=dataset_id,
            task=task,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
        )


def test_run_inspection_delegates_summary_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the run lifecycle reader without loading checkpoints or artifacts."""
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    (run_dir / common.paths.RUN_CONFIG_FILENAME).write_text(
        "task: steady_flow\n",
        encoding="utf-8",
    )
    run_module = importlib.import_module("src.experiments.experiments_run")
    calls: list[Path] = []

    def read_summary(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {
            "status": "completed",
            "objective": {"id": "synthetic_objective"},
            "completed_epoch": 2,
            "best_epoch": 1,
            "best_metric": 0.25,
        }

    monkeypatch.setattr(run_module, "read_run_summary", read_summary)

    inspection = support.prepare_run_inspection(
        run_dir,
        ood_dataset_id="synthetic_ood",
    )

    assert isinstance(inspection, support.RunInspection)
    assert inspection.run_dir == run_dir.resolve()
    assert calls == [run_dir.resolve()]
    existence = dict(inspection.existence_rows)
    assert existence[common.paths.RUN_CONFIG_FILENAME] is True
    assert existence[common.paths.RUN_SUMMARY_FILENAME] is False
