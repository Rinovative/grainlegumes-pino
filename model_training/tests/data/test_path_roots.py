# ruff: noqa: S101
"""Verify independent dataset, generated-data, and output storage roots."""

from __future__ import annotations

from pathlib import Path

import pytest
from src import common, experiments


def test_central_roots_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each root has one explicit environment concept."""
    dataset_root = tmp_path / "datasets"
    generated_root = tmp_path / "generated"
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("GENERATED_DATA_ROOT", str(generated_root))
    monkeypatch.setenv("OUTPUT_ROOT", str(output_root))

    assert common.paths.get_dataset_root() == dataset_root
    assert common.paths.get_generated_data_root() == generated_root
    assert common.paths.get_output_root() == output_root
    assert common.paths.resolve_dataset_path("tiny") == dataset_root / "tiny" / "tiny.pt"
    assert common.paths.resolve_run_output_dir("steady_flow", "run") == (output_root / "steady_flow" / "runs" / "run")


def test_output_override_cannot_relocate_dataset_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing only output_root leaves logical dataset paths byte-for-byte equal."""
    dataset_root = tmp_path / "datasets"
    first_output_root = tmp_path / "outputs-a"
    second_output_root = tmp_path / "outputs-b"
    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("OUTPUT_ROOT", str(first_output_root))
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
    assert dataset_after.is_relative_to(dataset_root)
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
    """The central validator rejects every supported traversal/name hazard."""
    for invalid_name in _INVALID_LOGICAL_NAMES:
        with pytest.raises(ValueError, match="single non-empty path component"):
            common.paths.validate_logical_name(invalid_name, label="logical name")


def test_owned_path_resolvers_apply_logical_name_validation(tmp_path: Path) -> None:
    """Each owned-path resolver wires the central component validation."""
    invalid_name = "../escape"
    with pytest.raises(ValueError, match="single non-empty path component"):
        common.paths.resolve_dataset_path(invalid_name, dataset_root=tmp_path)
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
