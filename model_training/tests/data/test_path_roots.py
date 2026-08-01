# ruff: noqa: S101
"""Verify the two public data domains and their derived lifecycle paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from src import common, experiments


def test_two_public_domains_derive_owned_lifecycle_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every generated and training path remains below exactly one public root."""
    generated_root = tmp_path / "generated domain"
    training_root = tmp_path / "training domain"
    monkeypatch.setenv("GENERATED_DATA_ROOT", str(generated_root))
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))

    assert common.paths.get_generated_data_root() == generated_root
    assert common.paths.get_generation_meta_root() == generated_root / "meta"
    assert common.paths.get_generation_raw_root() == generated_root / "raw"
    assert common.paths.get_generation_processed_root() == generated_root / "processed"
    assert common.paths.get_model_training_data_root() == training_root
    assert common.paths.get_training_meta_root() == training_root / "meta"
    assert common.paths.get_training_raw_root() == training_root / "raw"
    assert common.paths.get_training_processed_root() == training_root / "processed"
    assert common.paths.get_training_state_root() == training_root / ".state"
    assert common.paths.get_dataset_build_locks_root() == training_root / ".state" / "dataset-builds" / "locks"
    assert common.paths.get_dataset_build_transactions_root() == training_root / ".state" / "dataset-builds" / "transactions"
    assert common.paths.get_run_locks_root() == training_root / ".state" / "runs" / "locks"
    assert common.paths.get_dataset_root() == training_root / "raw"
    assert common.paths.get_output_root() == training_root / "processed"

    assert common.paths.resolve_generated_batch_dir("tiny", stage="raw") == generated_root / "raw" / "tiny"
    assert common.paths.resolve_generated_batch_dir("tiny", stage="processed") == generated_root / "processed" / "tiny"
    assert common.paths.resolve_dataset_path("tiny") == training_root / "raw" / "tiny" / "tiny.pt"
    assert common.paths.resolve_dataset_metadata_dir("tiny") == training_root / "meta" / "tiny"
    assert common.paths.resolve_dataset_build_lock_path("tiny") == training_root / ".state/dataset-builds/locks/dataset-tiny.lock"
    assert common.paths.resolve_dataset_build_transaction_path("tiny") == training_root / ".state/dataset-builds/transactions/dataset-tiny.json"
    run_lock = common.paths.resolve_run_lock_path(training_root / "processed/steady_flow/runs/run")
    artifact_lock = common.paths.resolve_artifact_lock_path(training_root / "processed/steady_flow/runs/run/analysis/id")
    assert run_lock.parent == training_root / ".state/runs/locks"
    assert run_lock.name.startswith("run-")
    assert run_lock.suffix == ".lock"
    assert artifact_lock.parent == training_root / ".state/runs/locks"
    assert artifact_lock.name.startswith("artifact-")
    assert artifact_lock.suffix == ".lock"
    assert common.paths.resolve_run_output_dir("steady_flow", "run") == (training_root / "processed" / "steady_flow" / "runs" / "run")
    assert common.paths.resolve_study_dir("steady_flow", "study") == (training_root / "processed" / "steady_flow" / "studies" / "study")


def test_repository_local_defaults_do_not_depend_on_host_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset domain overrides resolve only from the repository-local contract."""
    project_root = tmp_path / "repository"
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("GENERATED_DATA_ROOT", raising=False)
    monkeypatch.delenv("MODEL_TRAINING_DATA_ROOT", raising=False)

    assert common.paths.get_generated_data_root() == project_root / "data_generation" / "data"
    assert common.paths.get_model_training_data_root() == project_root / "model_training" / "data"


def test_resolved_training_config_records_only_training_domain_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training provenance resolves only its self-contained model-training domain."""
    generated_root = tmp_path / "generated"
    training_root = tmp_path / "training"
    monkeypatch.setenv("GENERATED_DATA_ROOT", str(generated_root))
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    config = experiments.config.loader.load_and_resolve_config(
        Path("model_training/configs/experiments/steady_flow_fno.yaml"),
    )

    assert config["paths"] == {
        "project_root": str(common.paths.get_project_root()),
        "model_training_data_root": str(training_root),
        "training_meta_root": str(training_root / "meta"),
        "dataset_root": str(training_root / "raw"),
        "output_root": str(training_root / "processed"),
    }


def test_output_override_cannot_relocate_dataset_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded output override moves runs but not immutable dataset inputs."""
    training_root = tmp_path / "training"
    second_output_root = tmp_path / "bounded outputs"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    config = experiments.config.loader.load_and_resolve_config(
        Path("model_training/configs/experiments/steady_flow_fno.yaml"),
    )
    dataset_before = common.paths.resolve_dataset_path(
        config["data"]["train_dataset"],
        dataset_root=config["paths"]["dataset_root"],
    )
    run_before = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=config["paths"]["output_root"],
    )

    config["paths"]["output_root"] = str(second_output_root)
    dataset_after = common.paths.resolve_dataset_path(
        config["data"]["train_dataset"],
        dataset_root=config["paths"]["dataset_root"],
    )
    run_after = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=config["paths"]["output_root"],
    )

    assert dataset_before == dataset_after
    assert run_before != run_after
    assert dataset_after.is_relative_to(training_root / "raw")
    assert run_after.is_relative_to(second_output_root)


_INVALID_LOGICAL_NAMES = (
    "",
    ".",
    "..",
    "../escape",
    "nested/name",
    "nested\\name",
    "/outside/escape",
    " trailing",
)


def test_logical_name_validator_rejects_unsafe_components() -> None:
    """Empty, traversal, separator, absolute, and untrimmed names are rejected."""
    for invalid_name in _INVALID_LOGICAL_NAMES:
        with pytest.raises(ValueError, match="single non-empty path component"):
            common.paths.validate_logical_name(invalid_name, label="logical name")


def test_owned_path_resolvers_apply_logical_name_validation(tmp_path: Path) -> None:
    """Every public resolver rejects traversal at its ownership boundary."""
    invalid_name = "../escape"
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_path(invalid_name, dataset_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_metadata_dir(invalid_name, metadata_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_build_lock_path(invalid_name, model_training_data_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_build_transaction_path(invalid_name, model_training_data_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_generated_batch_dir(invalid_name, stage="raw", generated_data_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_run_output_dir("steady_flow", invalid_name, output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_run_output_dir(invalid_name, "run", output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_optuna_trial_dir("steady_flow", invalid_name, 0, output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_runs_root(invalid_name, output_root=tmp_path)
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_ood_analysis_dir(tmp_path / "run", invalid_name)
