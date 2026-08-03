# ruff: noqa: S101, SLF001
"""
Protect the lightweight, read-only training-notebook support boundary.

Official recipes exercise lightweight official context resolution. Focused composition
tests cover deterministic presentation tables, metadata summaries, and completed-run
inspection while preventing workload execution from entering context preparation.
"""

from __future__ import annotations

import importlib
from dataclasses import fields
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from src import common, datasets, experiments
from support import configs

if TYPE_CHECKING:
    from pathlib import Path

_RECIPE_KINDS = (
    ("fno", False),
    ("uno", False),
    ("fno", True),
    ("uno", True),
)
support = experiments.notebook_support


def test_prepare_context_delegates_metadata_previews_and_keeps_state_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve one official recipe atomically through the shared preview boundary."""
    requests: list[dict[str, Any]] = []

    def preview(**request: Any) -> support.DatasetPreview:
        requests.append(request)
        return support.DatasetPreview(
            role=request["role"],
            dataset_id=request["dataset_id"],
            path=request["dataset_root"] / request["dataset_id"] / f"{request['dataset_id']}.pt",
            exists=False,
            sample_count=None,
            fingerprint=None,
            metadata_validated=False,
        )

    monkeypatch.setattr(support, "_dataset_preview", preview)
    config_path = configs.experiment_config_path(model_kind="fno", physics_enabled=False)
    context = support.prepare_notebook_context(config_path)

    assert isinstance(context, support.NotebookContext)
    assert context.config_path == config_path
    assert context.task.id == context.official_config["task"]
    expected_ids = [
        context.official_config["data"]["train_dataset"],
        *context.official_config["data"]["ood_datasets"],
    ]
    assert [request["dataset_id"] for request in requests] == expected_ids
    assert all(request["task"] == context.task for request in requests)
    assert len(context.dataset_previews) == len(expected_ids)
    assert all(isinstance(item, support.DatasetPreview) for item in context.dataset_previews)


def test_dataset_preview_marks_absent_metadata_but_propagates_invalid_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat an unmounted package as absent without hiding malformed mounted metadata."""
    dataset_root = tmp_path / "datasets"
    metadata_root = tmp_path / "metadata"
    dataset_id = "tiny_preview"
    dataset_path = common.paths.resolve_dataset_path(dataset_id, dataset_root=dataset_root)
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
    assert absent.sample_count is None

    metadata_directory = common.paths.resolve_dataset_metadata_dir(
        dataset_id,
        metadata_root=metadata_root,
    )
    metadata_directory.mkdir(parents=True)

    invalid_message = "invalid compact metadata package"

    def invalid_summary(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(invalid_message)

    monkeypatch.setattr(datasets.metadata, "load_dataset_metadata_summary", invalid_summary)
    with pytest.raises(ValueError, match="invalid compact metadata package"):
        support._dataset_preview(
            role="ID",
            dataset_id=dataset_id,
            task=task,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
        )


def test_run_inspection_uses_production_summary_reader_and_expected_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed-run interpretation delegated to the production summary owner."""
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    (run_dir / common.paths.RUN_CONFIG_FILENAME).write_text("task: steady_flow\n", encoding="utf-8")
    run_module = importlib.import_module("src.experiments.experiments_run")
    calls: list[Path] = []

    def read_summary(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {
            "status": "completed",
            "objective": {"id": "normalized_macro_rmse"},
            "completed_epoch": 2,
            "best_epoch": 1,
            "best_metric": 0.25,
        }

    monkeypatch.setattr(run_module, "read_run_summary", read_summary)
    inspection = support.prepare_run_inspection(run_dir, ood_dataset_id="ood_case")

    assert isinstance(inspection, support.RunInspection)
    assert calls == [run_dir.resolve()]
    assert dict(inspection.summary_rows)["Status"] == "completed"
    existence = dict(inspection.existence_rows)
    assert existence["config.yaml"] is True
    assert existence["summary.json"] is False
    assert "analysis/ood/ood_case/" in existence


def _resolved_context(
    model_kind: str = "fno",
    physics_enabled: bool = False,
) -> support.NotebookContext:
    """Prepare one semantically selected recipe without reading tensor payloads."""
    return support.prepare_notebook_context(
        configs.experiment_config_path(
            model_kind=model_kind,
            physics_enabled=physics_enabled,
        ),
    )


def _value_map(table: support.NotebookTable) -> dict[str, object]:
    """Map first-column labels to second-column values for focused assertions."""
    return {str(row[0]): row[1] for row in table.rows}


def test_configuration_tables_are_complete_architecture_aware_and_section_free() -> None:
    """Present resolved FNO, UNO, PI, and non-PI settings without dormant weights."""
    contexts = {recipe_kind: _resolved_context(*recipe_kind) for recipe_kind in _RECIPE_KINDS}
    expected_titles = (
        "Task and data",
        "Model",
        "Runtime and training",
        "Optimization, loss, and physics",
        "Evaluation and W&B",
    )
    for context in contexts.values():
        tables = support.prepare_configuration_tables(context)
        assert tuple(table.title for table in tables) == expected_titles
        assert all(table.columns == ("Setting", "Resolved value", "Meaning") for table in tables)
        assert all(all("Section" not in row for row in table.rows) for table in tables)

        by_title = {table.title: table for table in tables}
        task_values = _value_map(by_title["Task and data"])
        assert task_values["ID dataset"] == context.official_config["data"]["train_dataset"]
        assert task_values["OOD dataset"] == ", ".join(context.official_config["data"]["ood_datasets"])
        assert task_values["Input fields"] == ", ".join(context.task.input_names)
        assert task_values["Output fields"] == ", ".join(context.task.output_names)
        assert task_values["Normalizer fit role"] == context.task.preprocessing.fit_split

        evaluation_values = _value_map(by_title["Evaluation and W&B"])
        assert evaluation_values["Primary objective"] == context.objective["id"]
        assert evaluation_values["W&B mode"] == context.official_config["tracking"]["wandb"]["mode"]
        assert evaluation_values["Automatic categories"] == "Overview, Accuracy, Physics, Diagnostics"

    fno_model = _value_map(
        {table.title: table for table in support.prepare_configuration_tables(contexts[("fno", False)])}["Model"],
    )
    uno_model = _value_map(
        {table.title: table for table in support.prepare_configuration_tables(contexts[("uno", False)])}["Model"],
    )
    assert {"Implementation", "FNO skip", "Lifting ratio", "Projection ratio"} <= fno_model.keys()
    assert "Mode ratio" not in fno_model
    assert "Mode ratio" in uno_model
    assert {"Implementation", "FNO skip", "Lifting ratio", "Projection ratio"}.isdisjoint(uno_model)

    for model_kind in ("fno", "uno"):
        physics_values = _value_map(
            {table.title: table for table in support.prepare_configuration_tables(contexts[(model_kind, False)])}["Optimization, loss, and physics"],
        )
        assert physics_values["Physics enabled"] == "no"
        assert physics_values["Residual training contribution"] == "inactive"
        assert physics_values["Boundary training contribution"] == "inactive"
        assert "Residual weight" not in physics_values
        assert "Boundary weight" not in physics_values

        physics_values = _value_map(
            {table.title: table for table in support.prepare_configuration_tables(contexts[(model_kind, True)])}["Optimization, loss, and physics"],
        )
        assert physics_values["Physics enabled"] == "yes"
        assert physics_values["Residual weight"] != "inactive"
        assert physics_values["Boundary weight"] != "inactive"
        assert "Residual training contribution" not in physics_values
        assert "Boundary training contribution" not in physics_values


def test_dataset_and_run_preview_tables_are_deterministic_and_redacted(tmp_path: Path) -> None:
    """Keep identities and production path ownership while hiding mount prefixes."""
    context = _resolved_context()
    project_root = common.paths.get_project_root()
    model_training_data_root = common.paths.get_model_training_data_root()
    dataset = support.prepare_dataset_table(
        context,
        project_root=project_root,
        model_training_data_root=model_training_data_root,
    )
    run = support.prepare_run_preview_table(
        context,
        project_root=project_root,
        model_training_data_root=model_training_data_root,
    )
    assert dataset == support.prepare_dataset_table(
        context,
        project_root=project_root,
        model_training_data_root=model_training_data_root,
    )
    assert dataset.columns == ("Role", "Dataset ID", "Resolved path", "Exists", "Sample count", "Fingerprint")
    assert [row[1] for row in dataset.rows] == [
        context.official_config["data"]["train_dataset"],
        *context.official_config["data"]["ood_datasets"],
    ]
    assert all(str(row[2]).startswith("$MODEL_TRAINING_DATA_ROOT/") for row in dataset.rows)

    run_values = _value_map(run)
    assert run_values["Deterministic run name"] == context.official_config["run"]["name"]
    assert run_values["ID dataset"] == context.official_config["data"]["train_dataset"]
    assert run_values["OOD dataset"] == ", ".join(context.official_config["data"]["ood_datasets"])
    assert str(run_values["Expected run directory"]).startswith("$MODEL_TRAINING_DATA_ROOT/")
    assert (
        support.display_path(
            tmp_path / "outside",
            project_root=project_root,
            model_training_data_root=model_training_data_root,
        )
        == "<explicit external path>/outside"
    )


def test_run_output_inventory_matches_the_maintained_contract() -> None:
    """Keep one deterministic inventory for required and optional run outputs."""
    table = support.prepare_run_output_inventory_table()
    assert table.columns == ("File or directory", "Purpose", "Required", "Consumer")
    assert [row[0] for row in table.rows] == [
        "config.yaml",
        "summary.json",
        "split_indices.pt",
        "normalizer.pt",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "wandb/",
        "analysis/id/",
        "analysis/ood/<dataset>/",
        "artifact_provenance.json and case files",
    ]


def test_run_inspection_tables_preserve_summary_and_path_evidence(tmp_path: Path) -> None:
    """Convert validated inspection records without duplicating run admission."""
    project_root = common.paths.get_project_root()
    model_training_data_root = common.paths.get_model_training_data_root()
    run_dir = model_training_data_root / "processed" / "steady_flow" / "runs" / "example"
    inspection = support.RunInspection(
        run_dir=run_dir,
        summary_rows=(("Run directory", run_dir), ("Status", "completed"), ("Best metric", 0.25)),
        existence_rows=(("config.yaml", True), ("best_checkpoint.pt", False)),
    )
    summary, existence = support.prepare_run_inspection_tables(
        inspection,
        project_root=project_root,
        model_training_data_root=model_training_data_root,
    )
    assert summary.rows == (
        ("Run directory", "$MODEL_TRAINING_DATA_ROOT/processed/steady_flow/runs/example"),
        ("Status", "completed"),
        ("Best metric", "0.25"),
    )
    assert existence.rows == (("config.yaml", "yes"), ("best_checkpoint.pt", "no"))
    assert tmp_path != run_dir


def test_validation_presentation_preserves_every_result_field() -> None:
    """Map every typed validation field to deterministic tables and a conclusion."""
    validation = experiments.validation.data_pipeline
    overall = validation.ValidationCheck(check="Production APIs", evidence="admitted", result="PASS")
    membership = validation.DatasetMembershipRecord(
        role="ID full",
        dataset_id="id_data",
        full_samples=4,
        expected=4,
        observed=4,
        duplicates=0,
        missing=0,
        shape="BCHW",
        dtype="float32",
        fingerprint="a" * 16,
        task_digest="d" * 16,
        finite=True,
        policy="complete",
        result="PASS",
    )
    channel = validation.ChannelStatisticsRecord(
        tensor_role="Input",
        channel="input",
        fitted_mean=0.0,
        fitted_scale=1.0,
        normalized_mean=0.0,
        normalized_scale=1.0,
        finite=True,
        result="PASS",
    )
    coverage = validation.LoaderCoverageRecord(
        loader="ID train",
        sampler="RandomSampler",
        batches=1,
        batch_size=2,
        final_batch=2,
        drop_last=False,
        inverse_checked=True,
        finite=True,
        result="PASS",
    )
    result = validation.FullDataValidationResult(
        overall=(overall,),
        dataset_membership=(membership,),
        channels=(channel,),
        coverage=(coverage,),
        elapsed_seconds=1.25,
        peak_gib=0.5,
    )
    presentation = support.prepare_validation_presentation(result)
    tables = {table.title: table for table in presentation.tables}
    assert tables["Overall status"].rows[0] == tuple(getattr(overall, field.name) for field in fields(overall))
    assert tables["Dataset membership"].rows[0] == tuple(getattr(membership, field.name) for field in fields(membership))
    assert tables["Channel normalization"].rows[0] == tuple(getattr(channel, field.name) for field in fields(channel))
    assert tables["Loader coverage"].rows[0] == tuple(getattr(coverage, field.name) for field in fields(coverage))
    assert tables["Execution footprint"].rows == (("Elapsed seconds", 1.25), ("Peak memory (GiB)", 0.5))
    assert presentation.conclusion.startswith("**PASS:**")
