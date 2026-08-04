"""
===============================================================================
analysis_artifact_service.py
===============================================================================
Discover, validate, rebuild, generate, and reuse split-aware run artifacts.

Responsibilities:
  - Admit completed or evidence-valid terminal runs by explicit current path
  - Bind caches to exact checkpoint/config/normalizer/dataset/split identity
  - Reject partial or semantically mismatched caches without substitution
  - Rebuild only explicit exact targets and propagate every material failure
  - Load valid artifacts or generate only missing roles through one public service

Design principles:
  - Saved split membership is always required
  - Provenance is published last as the cache completion marker
  - Rebuild publishes a validated replacement only for the requested target
  - Local notebook orchestration never invokes tracking uploads

This module does NOT:
  - Parse CLI arguments or choose which runs a user intends to process
  - Define scientific metrics or the Parquet/NPZ payload schema
  - Render notebook figures or broaden the curated W&B upload inventory
===============================================================================
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src import common, domain, experiments

from . import contracts, generation, timing

if TYPE_CHECKING:
    from src.analysis.evaluation.evaluation_artifact_loader import LoadedRunArtifacts
    from src.datasets.dataset_identity import DatasetIdentity
    from src.datasets.dataset_metadata import DatasetMetadata
    from src.learning.learning_device import DeviceResolution

ArtifactSplit = Literal["eval", "ood"]
_SPLIT_INDEX_KEYS: dict[ArtifactSplit, str] = {
    "eval": "eval_indices",
    "ood": "ood_indices",
}
_SPLIT_IDENTITY_KEYS: dict[ArtifactSplit, str] = {
    "eval": "train",
    "ood": "ood",
}
_SPLIT_COUNT_KEYS: dict[ArtifactSplit, str] = {
    "eval": "n_eval",
    "ood": "n_ood",
}
_SPLIT_FULL_COUNT_KEYS: dict[ArtifactSplit, str] = {
    "eval": "n_train_full",
    "ood": "n_ood_full",
}


class ArtifactCacheError(RuntimeError):
    """
    Signal rejection of an existing or upload-bound artifact cache.

    Artifact admission raises this exception when schema, scientific identity,
    aggregate, manifest, or payload evidence cannot prove that a cache is both
    current and complete. Missing source runs/datasets and invalid caller paths
    retain their ordinary project exception types.
    """


@dataclass(frozen=True)
class RunArtifactPlan:
    """
    Bind one evaluable run to the exact ID and OOD datasets it persisted.

    Parameters
    ----------
    run_dir : pathlib.Path
        Current evaluable run whose config and saved split metadata were validated.
    id_dataset_name : str
        Logical training dataset supplying the saved evaluation membership.
    ood_dataset_name : str
        Logical OOD dataset supplying the saved OOD membership.

    Notes
    -----
    The dataclass is frozen so a validated plan cannot be retargeted in place.

    """

    run_dir: Path
    id_dataset_name: str
    ood_dataset_name: str
    lifecycle_status: str
    is_completed: bool
    scientific_run_name: str


@dataclass(frozen=True)
class ArtifactRequest:
    """
    Describe the complete request identity expected from one artifact target.

    Parameters
    ----------
    provenance : dict[str, Any]
        Canonical run, model, dataset, evaluator, physics, selection,
        generation, and runtime request evidence. Generated aggregate/results
        are deliberately absent until artifact publication.
    source_indices : tuple[int, ...]
        Exact complete ordered final-dataset membership saved by the run.
    batch_size : int
        Saved resolved evaluation batch size used by online and artifact inference.

    Notes
    -----
    Field rebinding is frozen. The provenance mapping is internal transport and
    is treated as immutable after request construction.

    """

    provenance: dict[str, Any]
    source_indices: tuple[int, ...]
    batch_size: int
    case_ids: tuple[str, ...] = ()
    dataset_metadata: DatasetMetadata | None = None


@dataclass(frozen=True)
class _EvaluatorArtifactContract:
    """
    Carry the task-declared payload schema resolved from admitted provenance.

    Parameters
    ----------
    task_id : str
        Exact task identity copied from ``provenance.run.task``.
    input_fields, output_fields, output_units : tuple[str, ...]
        Ordered names and output units required in every Parquet row and NPZ.
    output_groups : tuple[tuple[str, tuple[str, ...]], ...]
        Ordered task-owned physical output groups.
    train_standard_deviations : dict[str, float]
        Exact raw output scales from the saved training normalizer.
    normalization_denominator_floor : float
        Saved transform denominator addition used to validate normalized evidence.
    group_metric_columns : tuple[str, ...]
        Task-declared per-case physical and normalized group diagnostics.
    physics_kind : str
        Physics contract selecting generic versus steady-Brinkman payload rules.

    """

    task_id: str
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    output_units: tuple[str, ...]
    output_groups: tuple[tuple[str, tuple[str, ...]], ...]
    train_standard_deviations: dict[str, float]
    normalization_denominator_floor: float
    group_metric_columns: tuple[str, ...]
    physics_kind: str

    @property
    def is_steady_brinkman(self) -> bool:
        """Return whether the concrete steady-flow diagnostic schema applies."""
        return self.physics_kind == domain.physics.contracts.STEADY_BRINKMAN_KIND

    @property
    def velocity_unit(self) -> str:
        """Return the shared unit of the task-owned velocity output group."""
        matching_groups = tuple(fields for group_id, fields in self.output_groups if group_id == "velocity")
        if len(matching_groups) != 1:
            msg = "Steady artifact validation requires one task-owned velocity output group."
            raise ArtifactCacheError(msg)
        unit_by_field = dict(zip(self.output_fields, self.output_units, strict=True))
        try:
            units = {unit_by_field[field] for field in matching_groups[0]}
        except KeyError as error:
            msg = "Steady artifact velocity-group fields must be declared outputs."
            raise ArtifactCacheError(msg) from error
        if len(units) != 1:
            msg = f"Steady artifact velocity-group fields must share one unit, got {sorted(units)}."
            raise ArtifactCacheError(msg)
        return next(iter(units))


# ======================================================================
# Utilities
# ======================================================================


def cleanup_runtime(device: torch.device) -> None:
    """
    Collect Python state and release CUDA caches only for a CUDA execution path.

    Parameters
    ----------
    device : torch.device
        Concrete CPU or CUDA device resolved by the artifact-service boundary.

    Raises
    ------
    TypeError
        If ``device`` is not a concrete supported ``torch.device``.

    Notes
    -----
    CPU cleanup intentionally performs no CUDA availability query or CUDA API call.

    """
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Artifact cleanup requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _contains_evaluable_run_marker(path: Path) -> bool:
    """Return whether one directory contains any evaluation-contract marker."""
    return path.is_dir() and any((path / filename).exists() for filename in common.paths.CURRENT_RUN_REQUIRED_FILES)


def iter_run_dirs(root: Path, *, run_names: Iterable[str] | None = None) -> Iterable[Path]:
    """
    Iterate over terminal evidence-valid run directories in deterministic order.

    Explicit names below ``root`` are storage aliases. Internal scientific run
    names are validated independently and never inferred from directory leaves.
    """
    root = Path(root).expanduser()
    selected_run_names = list(run_names or [])
    if selected_run_names:
        for selected_storage_alias in selected_run_names:
            storage_alias = common.paths.validate_logical_name(selected_storage_alias, label="run_name")
            run_dir = root / storage_alias
            experiments.run.validate_evaluable_run(run_dir)
            yield run_dir.resolve()
        return

    if common.paths.is_evaluable_run_dir(root):
        experiments.run.validate_evaluable_run(root)
        yield root.resolve()
        return
    if not root.is_dir():
        msg = f"Run discovery root not found: {root}"
        raise FileNotFoundError(msg)
    if _contains_evaluable_run_marker(root):
        experiments.run.validate_evaluable_run(root)
        msg = f"Run validation returned without identifying an evaluable run: {root}"
        raise RuntimeError(msg)
    for candidate in sorted(root.iterdir()):
        if not _contains_evaluable_run_marker(candidate):
            continue
        experiments.run.validate_evaluable_run(candidate)
        yield candidate.resolve()


def _dataset_name_from_identity(metadata: Mapping[str, Any], *, identity_key: str) -> str:
    """
    Read one validated logical dataset ID from nested saved split metadata.

    Missing or malformed ``metadata.datasets[identity_key].dataset_id`` values
    fail at this boundary so config/path fallbacks cannot change cache identity.
    """
    datasets_meta = metadata.get("datasets")
    if not isinstance(datasets_meta, Mapping):
        msg = "split_indices.pt metadata.datasets must be a mapping."
        raise TypeError(msg)
    saved_identity = datasets_meta.get(identity_key)
    if not isinstance(saved_identity, Mapping):
        msg = f"split_indices.pt metadata.datasets.{identity_key} must be a mapping."
        raise TypeError(msg)
    dataset_id = saved_identity.get("dataset_id")
    return common.paths.validate_logical_name(
        dataset_id,
        label=f"split_indices.pt metadata.datasets.{identity_key}.dataset_id",
    )


def _load_split_contract(run_dir: Path) -> Mapping[str, Any]:
    """
    Load the authoritative saved split mapping on CPU without tensor-only mode.

    The complete mapping, including metadata and dataset identities, is required.
    Non-mapping payloads fail before artifact path or membership derivation.
    """
    split_indices_path = common.paths.resolve_split_indices_path(run_dir)
    split_indices = torch.load(
        split_indices_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(split_indices, Mapping):
        msg = f"split_indices.pt must contain a mapping: {split_indices_path}"
        raise TypeError(msg)
    return split_indices


def _split_metadata(split_contract: Mapping[str, Any], *, run_dir: Path) -> Mapping[str, Any]:
    """Return the required saved dataset/split metadata mapping."""
    metadata = split_contract.get("metadata")
    if not isinstance(metadata, Mapping):
        split_indices_path = common.paths.resolve_split_indices_path(run_dir)
        msg = f"split_indices.pt must contain metadata mapping: {split_indices_path}"
        raise TypeError(msg)
    return metadata


def _load_split_metadata(run_dir: Path) -> Mapping[str, Any]:
    """Load split_indices.pt metadata for artifact output naming."""
    return _split_metadata(_load_split_contract(run_dir), run_dir=run_dir)


def _load_run_config(run_dir: Path) -> Mapping[str, Any]:
    """Load and validate the top-level mapping in current config.yaml."""
    config_path = common.paths.resolve_run_config_path(run_dir)
    config = experiments.config.loader.load_yaml(config_path)
    if not isinstance(config, Mapping):
        msg = f"config.yaml must contain a top-level mapping: {config_path}"
        raise TypeError(msg)
    return config


def _load_data_config(run_dir: Path) -> Mapping[str, Any]:
    """Load the data section from current config.yaml."""
    config_path = common.paths.resolve_run_config_path(run_dir)
    data_cfg = _load_run_config(run_dir).get("data")
    if not isinstance(data_cfg, Mapping):
        msg = f"config.yaml must contain a data mapping: {config_path}"
        raise TypeError(msg)
    return data_cfg


def _required_config_dataset_name(data_cfg: Mapping[str, Any], key: str) -> str:
    """Return a required single dataset name from config data."""
    value = data_cfg.get(key)
    return common.paths.validate_logical_name(value, label=f"config.yaml data.{key}")


def _required_config_ood_dataset_name(data_cfg: Mapping[str, Any]) -> str:
    """Return the sole current-contract OOD dataset name from config data."""
    value = data_cfg.get("ood_datasets")
    if not isinstance(value, list) or len(value) != 1:
        msg = "config.yaml data.ood_datasets must contain exactly one logical dataset id."
        raise TypeError(msg)
    return common.paths.validate_logical_name(value[0], label="config.yaml data.ood_datasets[0]")


def load_run_artifact_plan(run_dir: Path) -> RunArtifactPlan:
    """
    Resolve the exact ID and OOD artifact datasets owned by an evaluable run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Current evaluable run containing config, split, normalizer, best checkpoint,
        and summary evidence.

    Returns
    -------
    RunArtifactPlan
        Frozen run path and logical dataset names taken from saved split identity.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError, RuntimeError
        If the run is incomplete or config dataset names disagree with the
        authoritative ``split_indices.pt`` metadata.

    Notes
    -----
    Output naming therefore uses the same saved identity consumed by inference.
    Config text alone never retargets artifacts.

    """
    run_dir = Path(run_dir).expanduser().resolve()
    admitted = experiments.run.validate_evaluable_run(run_dir)
    data_cfg = _load_data_config(run_dir)
    split_metadata = _load_split_metadata(run_dir)

    id_dataset_name = _dataset_name_from_identity(split_metadata, identity_key="train")
    ood_dataset_name = _dataset_name_from_identity(split_metadata, identity_key="ood")

    configured_id_dataset = _required_config_dataset_name(data_cfg, "train_dataset")
    configured_ood_dataset = _required_config_ood_dataset_name(data_cfg)

    if configured_id_dataset != id_dataset_name:
        msg = (
            "config.yaml data.train_dataset does not match split_indices.pt metadata.\n"
            f"config:   {configured_id_dataset}\n"
            f"metadata: {id_dataset_name}"
        )
        raise RuntimeError(msg)

    if ood_dataset_name != configured_ood_dataset:
        msg = (
            "config.yaml data.ood_datasets[0] does not match split_indices.pt metadata.\n"
            f"config:   {configured_ood_dataset}\n"
            f"metadata: {ood_dataset_name}"
        )
        raise RuntimeError(msg)

    return RunArtifactPlan(
        run_dir=run_dir,
        id_dataset_name=id_dataset_name,
        ood_dataset_name=ood_dataset_name,
        lifecycle_status=str(admitted["lifecycle_status"]),
        is_completed=bool(admitted["is_completed"]),
        scientific_run_name=str(admitted["scientific_run_name"]),
    )


def _artifact_save_root(*, run_dir: Path, dataset_name: str, split: ArtifactSplit) -> Path:
    """Resolve the artifact save root for a split."""
    if split == "eval":
        return common.paths.resolve_id_analysis_dir(run_dir)
    return common.paths.resolve_ood_analysis_dir(run_dir, dataset_name)


def _normalise_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving symbolic links."""
    return Path(os.path.abspath(path.expanduser()))  # noqa: PTH100 -- lexical normalization must not follow symlinks


def _validated_artifact_target(*, run_dir: Path, save_root: Path) -> tuple[Path, Path]:
    """
    Admit one exact run-owned artifact leaf without following symlink aliases.

    Only ``analysis/id`` and one named ``analysis/ood/<dataset>`` leaf are
    accepted. Lexical containment and every existing path component are checked
    before a destructive rebuild or publication lock path is derived.
    """
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    analysis_root = _normalise_path(common.paths.resolve_analysis_root(run_root))
    target = _normalise_path(Path(save_root))
    if not analysis_root.is_relative_to(run_root) or analysis_root == run_root or not target.is_relative_to(analysis_root):
        msg = f"Refusing rebuild outside one exact artifact target: {target}"
        raise ValueError(msg)
    id_target = analysis_root / "id"
    ood_root = analysis_root / "ood"
    is_named_ood_target = target.parent == ood_root and target.name not in {"", ".", ".."}
    if target != id_target and not is_named_ood_target:
        msg = f"Refusing rebuild outside one exact artifact target: {target}"
        raise ValueError(msg)

    current = analysis_root
    for part in (Path(), *target.relative_to(analysis_root).parents[::-1], target.relative_to(analysis_root)):
        candidate = current if part == Path() else analysis_root / part
        if candidate.is_symlink():
            msg = f"Refusing rebuild outside one exact artifact target through symbolic link: {candidate}"
            raise ValueError(msg)
    return analysis_root, target


def _artifact_lock_path(*, run_dir: Path, save_root: Path) -> Path:
    """Return one centralized lock path outside an exact deletable artifact target."""
    _analysis_root, target = _validated_artifact_target(run_dir=run_dir, save_root=save_root)
    return common.paths.resolve_artifact_lock_path(target)


def _completion_marker_identity(path: Path) -> tuple[int, int, int, int] | None:
    """Return replacement-sensitive identity for one cache completion marker."""
    try:
        result = path.stat()
    except FileNotFoundError:
        return None
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def _required_metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    """Return one non-negative integer from saved split metadata."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"split_indices.pt metadata {key!r} must be a non-negative integer."
        raise TypeError(msg)
    return value


def _indices_sha256(indices: Iterable[int]) -> str:
    """Hash ordered split membership without hashing dataset tensor contents."""
    return contracts.ordered_indices_sha256(indices)


def _load_bound_dataset_metadata(
    source_dataset: Any,
    *,
    dataset_name: str,
    dataset_identity: DatasetIdentity,
    metadata_root: Path,
) -> DatasetMetadata:
    """Load metadata and bind its exact manifest digest to the final payload."""
    from src import datasets  # noqa: PLC0415

    package = datasets.metadata.load_dataset_metadata(
        dataset_name,
        dataset_identity=dataset_identity,
        metadata_root=metadata_root,
        dataset_path=source_dataset.path,
    )
    source_payload = getattr(source_dataset, "data", None)
    source_provenance = source_payload.get("source_provenance") if isinstance(source_payload, Mapping) else None
    if not isinstance(source_provenance, Mapping):
        msg = "Final dataset must expose source-batch provenance."
        raise TypeError(msg)
    if source_provenance.get("batch_manifest_sha256") != package.source_manifest_sha256:
        msg = "Dataset metadata source manifest does not match the final dataset's operational provenance."
        raise ValueError(msg)
    return package


def _build_artifact_request(  # noqa: C901, PLR0915
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    metadata_root: Path,
) -> ArtifactRequest:
    """
    Build and validate the complete semantic request for one artifact cache.

    This boundary admits the evaluable run, its exact saved split membership,
    the source dataset identity, the resolved task/objective, model capacity,
    normalizer, physics semantics, and runtime device decision. Filesystem
    locations, mtimes, and byte sizes are intentionally excluded from the
    scientific identity so equivalent relocations remain comparable.

    Parameters
    ----------
    run_dir : pathlib.Path
        Evaluable run whose immutable evidence owns the artifacts.
    dataset_name : str
        Dataset identity expected for the selected split role.
    split : {"eval", "ood"}
        Saved split membership to materialize.
    device_resolution : DeviceResolution
        Already resolved execution-device decision.
    dataset_root : pathlib.Path
        Root used only to resolve and validate the source dataset.
    metadata_root : pathlib.Path
        Root containing the dataset's validated provenance and timing snapshots.

    Returns
    -------
    ArtifactRequest
        Exact provenance document and complete ordered source membership.

    Raises
    ------
    RuntimeError, TypeError, ValueError
        If run evidence, dataset identity, split membership, or resolved
        semantics disagree or are incomplete.

    """
    run_dir = Path(run_dir).expanduser().resolve()
    admitted = experiments.run.validate_evaluable_run(run_dir)
    split_contract = admitted["split_indices"]
    run_config = admitted["config"]
    summary = admitted["summary"]
    metadata = _split_metadata(split_contract, run_dir=run_dir)
    data_cfg = run_config.get("data")
    run_cfg = run_config.get("run")
    model_cfg = run_config.get("model")
    loss_cfg = run_config.get("loss")
    evaluation_cfg = run_config.get("evaluation")
    if (
        not isinstance(data_cfg, Mapping)
        or not isinstance(run_cfg, Mapping)
        or not isinstance(model_cfg, Mapping)
        or not isinstance(loss_cfg, Mapping)
        or not isinstance(evaluation_cfg, Mapping)
    ):
        msg = f"config.yaml must contain data, run, model, loss, and evaluation mappings: {common.paths.resolve_run_config_path(run_dir)}"
        raise TypeError(msg)
    from src import learning  # noqa: PLC0415

    if not isinstance(device_resolution, learning.device.DeviceResolution):
        msg = "Artifact provenance requires one resolved runtime device decision."
        raise TypeError(msg)
    raw_batch_size = data_cfg.get("batch_size")
    if isinstance(raw_batch_size, bool) or not isinstance(raw_batch_size, Integral) or int(raw_batch_size) <= 0:
        msg = "Evaluable run config data.batch_size must be a positive integer."
        raise TypeError(msg)
    batch_size = int(raw_batch_size)
    task = experiments.config.loader.validate_resolved_task_contract(run_config)
    output_groups = contracts.output_group_payload(task.output_groups)
    normalizer_state = admitted.get("normalizer_state")
    if not isinstance(normalizer_state, Mapping):
        msg = "Evaluable run must contain the validated saved normalizer state."
        raise TypeError(msg)
    train_standard_deviations = contracts.output_standard_deviations_from_state(
        normalizer_state,
        output_fields=task.output_names,
    )
    from src import datasets  # noqa: PLC0415

    dataset_path = common.paths.resolve_dataset_path(dataset_name, dataset_root=dataset_root)
    source_dataset = datasets.simulation.create_task_dataset(dataset_path, task=task)
    expected_identity = source_dataset.identity
    validated_indices = datasets.base.validate_split_info(
        split_contract,
        train_identity=expected_identity if split == "eval" else None,
        ood_identity=expected_identity if split == "ood" else None,
        expected_train_ratio=data_cfg["train_ratio"],
        expected_ood_fraction=data_cfg["ood_fraction"],
        expected_split_seed=experiments.run.build_seed_plan(int(run_cfg["seed"]))["split"],
    )
    identity_key = _SPLIT_IDENTITY_KEYS[split]
    saved_dataset_name = _dataset_name_from_identity(metadata, identity_key=identity_key)
    if saved_dataset_name != dataset_name:
        msg = f"Requested dataset {dataset_name!r} does not match saved {split!r} dataset {saved_dataset_name!r}."
        raise RuntimeError(msg)

    saved_indices = validated_indices[_SPLIT_INDEX_KEYS[split]]
    full_source_indices = tuple(int(value) for value in saved_indices.tolist())
    count_key = _SPLIT_COUNT_KEYS[split]
    saved_selected_count = _required_metadata_int(metadata, count_key)
    if saved_selected_count != len(full_source_indices):
        msg = f"split_indices.pt metadata {count_key!r} does not match saved index count {len(full_source_indices)}."
        raise RuntimeError(msg)
    full_count_key = _SPLIT_FULL_COUNT_KEYS[split]
    dataset_full_count = _required_metadata_int(metadata, full_count_key)
    if dataset_full_count != expected_identity.sample_count:
        msg = f"Saved {full_count_key!r} does not match the verified dataset sample count."
        raise RuntimeError(msg)

    effective_count = len(full_source_indices)
    effective_source_indices = full_source_indices
    effective_case_ids = tuple(expected_identity.sample_ids[index] for index in full_source_indices)
    dataset_metadata = _load_bound_dataset_metadata(
        source_dataset,
        dataset_name=dataset_name,
        dataset_identity=expected_identity,
        metadata_root=metadata_root,
    )
    membership_digests = metadata.get("membership_digests")
    if not isinstance(membership_digests, Mapping):
        msg = "split_indices.pt metadata.membership_digests must be a mapping."
        raise TypeError(msg)
    metrics = evaluation_cfg.get("metrics")
    if not isinstance(metrics, list):
        msg = "config.yaml evaluation.metrics must be a list."
        raise TypeError(msg)
    objective = experiments.config.loader.get_resolved_objective(run_config)
    physics_cfg = loss_cfg.get("physics")
    if not isinstance(physics_cfg, Mapping):
        msg = "config.yaml loss.physics must be a resolved mapping."
        raise TypeError(msg)
    selected_training_continuity = physics_cfg.get("continuity")
    if not isinstance(selected_training_continuity, str):
        msg = "config.yaml loss.physics.continuity must be a resolved semantic identifier."
        raise TypeError(msg)
    if selected_training_continuity not in task.physics.allowed_continuities:
        msg = f"Resolved training continuity {selected_training_continuity!r} is not allowed by task {task.id!r}."
        raise ValueError(msg)

    physics_provenance: dict[str, Any] | None = None
    if task.physics.kind == domain.physics.contracts.STEADY_BRINKMAN_KIND:
        physics_provenance = {
            "residual_schema_version": contracts.RESIDUAL_SCHEMA_VERSION,
            "task_id": task.id,
            "task_contract_digest": task.contract_digest,
            "equation_kind": task.physics.kind,
            "equation_set": task.physics.equation_set,
            "boundary_condition_kind": task.physics.boundary,
            "selected_training_continuity": selected_training_continuity,
            "evaluated_continuity_formulations": list(domain.physics.contracts.available_continuity_kinds()),
            "constants": {
                "dynamic_viscosity_pa_s": domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY,
                "porosity_floor": domain.physics.brinkman.POROSITY_FLOOR,
                "permeability_scale_floor_m2": domain.physics.brinkman.PERMEABILITY_SCALE_FLOOR,
                "permeability_determinant_floor": domain.physics.brinkman.PERMEABILITY_DETERMINANT_FLOOR,
                "permeability_cross_ratio_clip": domain.physics.brinkman.PERMEABILITY_CROSS_RATIO_CLIP,
            },
            "permeability_representation": {
                "kxx": "10**stored_log10_ratio_to_1_m2",
                "kxy": "stored_dimensionless_ratio_times_sqrt(kxx*kyy)",
                "kyy": "10**stored_log10_ratio_to_1_m2",
                "inverse": "normalized_symmetric_2x2_inverse_with_declared_floors",
            },
            "derivatives": {
                "kind": contracts.ARTIFACT_DERIVATIVE_KIND,
                "extension": contracts.ARTIFACT_DERIVATIVE_EXTENSION,
                "operator_axes": list(task.operator_axes),
                "grid_axes": ["y", "x"],
            },
            "interior_crop": contracts.EVAL_PAD,
            "residual_evaluation_region": {
                "momentum_residual_mse": "interior grid after symmetric cell crop",
                "div_velocity_mse": "interior grid after symmetric cell crop",
                "div_eps_velocity_mse": "interior grid after symmetric cell crop",
                "pressure_boundary_mse": "full-grid inlet and outlet masks",
                "pressure_inlet_mse": "full-grid y-min inlet mask",
                "pressure_outlet_mean_square": "square of the sample mean on the full-grid y-max outlet mask",
                "residual_arrays": "full grid",
            },
            "scalar_definitions": {
                "momentum_residual_mse": {"formula": "mean(Rx**2 + Ry**2)", "unit": "(Pa/m)^2"},
                "div_velocity_mse": {"formula": "mean(div(u)**2)", "unit": "1/s^2"},
                "div_eps_velocity_mse": {"formula": "mean(div(eps*u)**2)", "unit": "1/s^2"},
                "pressure_boundary_mse": {
                    "formula": "pressure_inlet_mse + pressure_outlet_mean_square",
                    "unit": "Pa^2",
                },
                "pressure_inlet_mse": {"formula": "mean_inlet((p-p_bc)**2)", "unit": "Pa^2"},
                "pressure_outlet_mean_square": {"formula": "mean_outlet(p)**2", "unit": "Pa^2"},
            },
            "array_definitions": {
                "Rx": {"formula": "-dp/dx + div(tau)_x - mu*(K^-1*u)_x", "unit": "Pa/m"},
                "Ry": {"formula": "-dp/dy + div(tau)_y - mu*(K^-1*u)_y", "unit": "Pa/m"},
                "div_u": {"formula": "du/dx + dv/dy", "unit": "1/s"},
                "div_eps_u": {"formula": "d(eps*u)/dx + d(eps*v)/dy", "unit": "1/s"},
            },
        }

    raw_parameter_counts = summary.get("model_parameter_counts")
    parameter_counts: dict[str, int] | None = None
    if raw_parameter_counts is not None:
        if not isinstance(raw_parameter_counts, Mapping):
            msg = "Run summary model_parameter_counts must be a mapping when recorded."
            raise TypeError(msg)
        parameter_counts = {}
        for name in ("total", "trainable"):
            value = raw_parameter_counts.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                msg = "Run model_parameter_counts must contain positive integer total and trainable values."
                raise TypeError(msg)
            parameter_counts[name] = int(value)
    architecture = model_cfg.get("params")
    if not isinstance(architecture, Mapping):
        msg = "Resolved config model.params must be a mapping."
        raise TypeError(msg)

    model_provenance: dict[str, Any] = {
        "kind": model_cfg.get("kind"),
        "architecture": dict(architecture),
        "physics_enabled": physics_cfg.get("enabled") is True,
    }
    if parameter_counts is not None:
        model_provenance["parameter_counts"] = parameter_counts
    provenance: dict[str, Any] = {
        "provenance_schema_version": contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": contracts.ARTIFACT_SCHEMA_VERSION,
        "run": {
            "name": admitted["scientific_run_name"],
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "effective_config_digest": admitted["effective_config_digest"],
            "best_checkpoint_sha256": admitted["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": admitted["selected_checkpoint_epoch"],
            "normalizer_sha256": admitted["normalizer_sha256"],
            "lifecycle_status": admitted["lifecycle_status"],
            "is_completed": admitted["is_completed"],
            "is_provisional": admitted["is_provisional"],
            "selected_checkpoint_role": admitted["selected_checkpoint_role"],
        },
        "model": model_provenance,
        "split_role": split,
        "dataset": {
            "name": dataset_name,
            "full_case_count": dataset_full_count,
            "fingerprint": expected_identity.fingerprint,
            "data_contract_digest": expected_identity.data_contract_digest,
            "saved_membership_digest": membership_digests[split],
        },
        "selection": {
            "index_key": _SPLIT_INDEX_KEYS[split],
            "full_selected_case_count": len(full_source_indices),
            "effective_case_count": effective_count,
            "generation_limit": None,
            "full_ordered_source_indices_sha256": _indices_sha256(full_source_indices),
            "effective_ordered_source_indices_sha256": _indices_sha256(effective_source_indices),
        },
        "normalizer": {
            "sha256": admitted["normalizer_sha256"],
            "identity": "saved_run_normalizer.pt",
            "fit_split": task.preprocessing.fit_split,
            "output_normalization": task.preprocessing.output_normalization,
            "denominator_floor": 1e-7,
            "output_standard_deviations": train_standard_deviations,
        },
        "evaluator": {
            "metrics": metrics,
            "objective": objective,
            "input_fields": list(task.input_names),
            "input_units": {field.name: field.unit for field in task.inputs},
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
            "output_groups": output_groups,
            "physics_kind": task.physics.kind,
            "group_objective_evidence": {
                "squared_error_accumulation_dtype": "float64",
                "per_case_physical_columns": {field: list(contracts.physical_statistic_columns(field)) for field in task.output_names},
                "per_case_normalized_columns": {field: list(contracts.normalized_statistic_columns(field)) for field in task.output_names},
                "train_scale_source": "saved_run_normalizer.output_standard_deviations",
                "dataset_reduction": ("sum physical field SSE/count, finalize shared-scale group RMSE, equal macro mean over output groups"),
            },
            "predictive_metrics": {
                "rel_l2": "per-case arithmetic mean of physical per-field relative L2 ratios",
                "rel_h1": "per-case arithmetic mean of physical per-field relative H1 ratios on the declared artifact region where available",
                "physical_rmse_columns": {field: contracts.physical_statistic_columns(field)[2] for field in task.output_names},
            },
        },
        "generation": {
            "effective_case_limit": None,
            "inference_batch_size": batch_size,
            "compression": "numpy savez_compressed",
        },
        "runtime": {
            **device_resolution.as_dict(),
            "batch_size": batch_size,
        },
    }
    if physics_provenance is not None:
        provenance["physics"] = physics_provenance
    return ArtifactRequest(
        provenance=provenance,
        source_indices=effective_source_indices,
        batch_size=batch_size,
        case_ids=effective_case_ids,
        dataset_metadata=dataset_metadata,
    )


def _read_artifact_provenance(path: Path) -> Mapping[str, Any]:
    """
    Read one provenance JSON object through the cache-admission exception boundary.

    Filesystem, decoding, and non-object payload failures are normalized to
    :class:`ArtifactCacheError` because an observed completion marker must be
    wholly trustworthy before reuse.
    """
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Artifact provenance is unreadable: {path}: {error}"
        raise ArtifactCacheError(msg) from error
    if not isinstance(payload, Mapping):
        msg = f"Artifact provenance must contain a JSON object: {path}"
        raise ArtifactCacheError(msg)
    return payload


def _scientific_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """
    Isolate the scientific request identity used to compare cache provenance.

    Operational device facts, generated aggregate values, and output digests
    are excluded. The saved online-evaluation batch size remains under generation
    provenance because batch-sensitive operators must reproduce that exact path.
    """
    identity = dict(provenance)
    identity.pop("runtime", None)
    identity.pop("aggregate", None)
    identity.pop("outputs", None)
    run = identity.get("run")
    if isinstance(run, Mapping) and run.get("is_provisional") is not True:
        normalized_run = dict(run)
        for field in (
            "best_checkpoint_epoch",
            "lifecycle_status",
            "is_completed",
            "is_provisional",
            "selected_checkpoint_role",
        ):
            normalized_run.pop(field, None)
        identity["run"] = normalized_run
    return identity


def _require_current_provenance_schema(provenance: Mapping[str, Any]) -> None:
    """Require the exact current artifact and provenance schema versions."""
    versions = (
        (
            "provenance_schema_version",
            contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        ),
        ("artifact_schema_version", contracts.ARTIFACT_SCHEMA_VERSION),
    )
    for field, expected in versions:
        value = provenance.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            msg = f"Artifact provenance has an unsupported {field}."
            raise ArtifactCacheError(msg)


def _runtime_identities(request: ArtifactRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the minimal dataset and model identities bound to timing."""
    run = request.provenance.get("run")
    dataset = request.provenance.get("dataset")
    selection = request.provenance.get("selection")
    if not isinstance(run, Mapping) or not isinstance(dataset, Mapping) or not isinstance(selection, Mapping):
        msg = "Artifact request lacks timing identity provenance."
        raise TypeError(msg)
    return (
        {
            "name": dataset.get("name"),
            "fingerprint": dataset.get("fingerprint"),
            "data_contract_digest": dataset.get("data_contract_digest"),
            "saved_membership_digest": dataset.get("saved_membership_digest"),
            "effective_ordered_source_indices_sha256": selection.get("effective_ordered_source_indices_sha256"),
        },
        {
            "run_name": run.get("name"),
            "effective_config_digest": run.get("effective_config_digest"),
            "best_checkpoint_sha256": run.get("best_checkpoint_sha256"),
        },
    )


def _validate_runtime_comparison_request(
    payload: Mapping[str, Any],
    *,
    request: ArtifactRequest,
) -> dict[str, Any]:
    """Bind operational timing to the current model and saved membership."""
    validated = timing.validate_runtime_comparison(payload)
    dataset_identity, model_identity = _runtime_identities(request)
    if validated["dataset_identity"] != dataset_identity or validated["model_identity"] != model_identity:
        msg = "Runtime comparison disagrees with artifact dataset or model identity."
        raise ArtifactCacheError(msg)
    if validated["split_role"] != request.provenance.get("split_role"):
        msg = "Runtime comparison disagrees with the saved split role."
        raise ArtifactCacheError(msg)
    if validated["measurement"]["batch_size"] != request.batch_size:
        msg = "Runtime comparison disagrees with the saved evaluation batch size."
        raise ArtifactCacheError(msg)
    if [case["case_id"] for case in validated["cases"]] != list(request.case_ids):
        msg = "Runtime comparison case IDs disagree with saved membership."
        raise ArtifactCacheError(msg)
    if [case["source_index"] for case in validated["cases"]] != list(request.source_indices):
        msg = "Runtime comparison source indices disagree with saved membership."
        raise ArtifactCacheError(msg)
    return validated


def _resolve_comsol_timing(
    request: ArtifactRequest,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Return timing only from the validated model-training metadata snapshot."""
    package = request.dataset_metadata
    if package is None:
        return None, None, "validated model-training dataset metadata is unavailable"
    manifest_sha256 = package.source_manifest_sha256
    timing_payload = package.timing
    if timing_payload is None:
        status = package.timing_summary["status"]
        return None, None, f"validated model-training COMSOL timing snapshot is {status}"
    try:
        validated = timing.validate_comsol_solve_timing(timing_payload)
    except (TypeError, ValueError) as error:
        return None, None, f"COMSOL timing snapshot is incompatible: {error}"
    if validated["batch_manifest_sha256"] != manifest_sha256:
        return None, None, "COMSOL timing snapshot does not bind the dataset metadata manifest"
    if not validated["cases"]:
        return None, None, "validated model-training COMSOL timing snapshot is missing"
    return validated, manifest_sha256, None


def _report_runtime_comparison(
    *,
    save_root: Path,
    request: ArtifactRequest,
) -> None:
    """Report timing availability without mutating the scientific DataFrame."""
    try:
        payload = _validate_runtime_comparison_request(
            timing.load_runtime_comparison(save_root),
            request=request,
        )
    except (ArtifactCacheError, RuntimeError, TypeError, ValueError) as error:
        print(
            "[TIMING] runtime timing is unavailable or incompatible. "
            "Scientific artifacts remain valid. Use an explicit --rebuild "
            f"to measure again: {error}"
        )
    else:
        measured = payload["aggregates"]["neural_operator_forward_s"]["count"]
        matched = payload["aggregates"]["speedup"]["count"]
        print(f"[TIMING] validated runtime comparison: measured={measured}, matched={matched}")


def _cache_has_outputs(*, save_root: Path, parquet_path: Path, npz_dir: Path) -> bool:
    """
    Detect any content that requires fail-closed cache admission.

    A non-empty target counts even when its files are unrecognized, preventing
    normal generation from overwriting interrupted or foreign content.
    """
    return any(
        (
            save_root.is_dir() and any(save_root.iterdir()),
            parquet_path.exists(),
            contracts.artifact_provenance_path(save_root).exists(),
            any(npz_dir.glob("*.npz")),
            any(save_root.glob(".*.tmp")),
            any(npz_dir.glob(".*.tmp")),
        )
    )


def _require_identity_values(df: pd.DataFrame, column: str) -> tuple[int, ...]:
    """Return an integer artifact identity column without coercion."""
    if column not in df.columns:
        msg = f"Cached Parquet is missing required identity column {column!r}."
        raise ArtifactCacheError(msg)

    values = df.loc[:, column].tolist()
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        msg = f"Cached Parquet column {column!r} must contain only integers."
        raise ArtifactCacheError(msg)
    return tuple(int(value) for value in values)


def _require_metadata_without_identity(raw_meta: Any, *, label: str) -> dict[str, Any]:
    """
    Parse one case metadata object while protecting authoritative identity fields.

    JSON text and mappings are accepted, but case/source/split identity keys are
    rejected because those values must come only from validated Parquet and NPZ
    scalar fields.
    """
    if not isinstance(raw_meta, str):
        msg = f"{label} metadata must be a JSON string."
        raise ArtifactCacheError(msg)
    try:
        metadata = json.loads(raw_meta)
    except json.JSONDecodeError as error:
        msg = f"{label} metadata is invalid JSON: {error}"
        raise ArtifactCacheError(msg) from error
    if not isinstance(metadata, dict):
        msg = f"{label} metadata must decode to an object."
        raise ArtifactCacheError(msg)
    reserved = {"case_index", "source_index", "split_local_index"}.intersection(metadata)
    if reserved:
        msg = f"{label} metadata duplicates reserved top-level identity fields: {sorted(reserved)}."
        raise ArtifactCacheError(msg)
    return metadata


def _require_provenance_string_list(value: Any, *, label: str) -> tuple[str, ...]:
    """Return one non-empty, duplicate-free JSON string list."""
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        msg = f"Artifact provenance {label} must be a non-empty list of strings."
        raise ArtifactCacheError(msg)
    values = tuple(value)
    if len(set(values)) != len(values):
        msg = f"Artifact provenance {label} contains duplicate field names."
        raise ArtifactCacheError(msg)
    return values


def _evaluator_artifact_contract(provenance: Mapping[str, Any]) -> _EvaluatorArtifactContract:
    """
    Resolve the exact evaluator payload contract from admitted provenance.

    Task identity, unique input/output order, units, task-owned output groups,
    raw saved training scales, declared group diagnostics, and physics kind are
    validated once and frozen for every subsequent Parquet/NPZ check.
    """
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        msg = "Artifact provenance must contain an evaluator mapping."
        raise ArtifactCacheError(msg)
    run = provenance.get("run")
    if not isinstance(run, Mapping):
        msg = "Artifact provenance must contain a run mapping."
        raise ArtifactCacheError(msg)
    task_id = run.get("task")
    if not isinstance(task_id, str) or not task_id:
        msg = "Artifact provenance run.task must be a non-empty string."
        raise ArtifactCacheError(msg)
    input_fields = _require_provenance_string_list(evaluator.get("input_fields"), label="evaluator.input_fields")
    output_fields = _require_provenance_string_list(evaluator.get("output_fields"), label="evaluator.output_fields")
    raw_units = evaluator.get("output_units")
    if not isinstance(raw_units, Mapping) or set(raw_units) != set(output_fields):
        msg = "Artifact provenance evaluator.output_units must map exactly the declared output fields."
        raise ArtifactCacheError(msg)
    if any(not isinstance(raw_units[name], str) or not raw_units[name] for name in output_fields):
        msg = "Artifact provenance evaluator.output_units values must be non-empty strings."
        raise ArtifactCacheError(msg)
    try:
        group_payload = contracts.output_group_payload(evaluator.get("output_groups", ()))
    except (TypeError, ValueError) as error:
        msg = f"Artifact evaluator output groups are invalid: {error}"
        raise ArtifactCacheError(msg) from error
    grouped_fields = tuple(field for group in group_payload for field in group["fields"])
    if grouped_fields != output_fields:
        msg = "Artifact evaluator output groups do not partition declared outputs in order."
        raise ArtifactCacheError(msg)

    normalizer = provenance.get("normalizer")
    raw_scales = normalizer.get("output_standard_deviations") if isinstance(normalizer, Mapping) else None
    denominator_floor = normalizer.get("denominator_floor") if isinstance(normalizer, Mapping) else None
    if (
        isinstance(denominator_floor, bool)
        or not isinstance(denominator_floor, Real)
        or not np.isfinite(float(denominator_floor))
        or float(denominator_floor) < 0.0
    ):
        msg = "Artifact normalizer denominator_floor must be finite and non-negative."
        raise ArtifactCacheError(msg)
    if not isinstance(raw_scales, Mapping) or set(raw_scales) != set(output_fields):
        msg = "Artifact normalizer output_standard_deviations must map exactly the output fields."
        raise ArtifactCacheError(msg)
    train_standard_deviations: dict[str, float] = {}
    for field in output_fields:
        value = raw_scales[field]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) <= 0.0:
            msg = f"Artifact train standard deviation for {field!r} must be finite and strictly positive."
            raise ArtifactCacheError(msg)
        train_standard_deviations[field] = float(value)

    raw_metrics = evaluator.get("metrics")
    if not isinstance(raw_metrics, list):
        msg = "Artifact evaluator metrics must be a list."
        raise ArtifactCacheError(msg)
    group_fields = {tuple(group["fields"]) for group in group_payload}
    group_metric_columns: list[str] = []
    for metric in raw_metrics:
        if not isinstance(metric, Mapping) or metric.get("kind") not in {"group_rmse", "vector_rmse"}:
            continue
        metric_id = metric.get("id")
        raw_metric_fields = metric.get("fields")
        metric_fields = output_fields if raw_metric_fields == "all" else tuple(raw_metric_fields or ())
        if not isinstance(metric_id, str) or not metric_id or metric_fields not in group_fields:
            msg = "Artifact evaluator contains an invalid output-group diagnostic metric."
            raise ArtifactCacheError(msg)
        group_metric_columns.append(metric_id)
    if len(group_metric_columns) != len(set(group_metric_columns)):
        msg = "Artifact evaluator contains duplicate output-group diagnostic ids."
        raise ArtifactCacheError(msg)

    physics_kind = evaluator.get("physics_kind")
    if not isinstance(physics_kind, str) or not physics_kind:
        msg = "Artifact provenance evaluator.physics_kind must be a non-empty string."
        raise ArtifactCacheError(msg)
    return _EvaluatorArtifactContract(
        task_id=task_id,
        input_fields=input_fields,
        output_fields=output_fields,
        output_units=tuple(raw_units[name] for name in output_fields),
        output_groups=tuple((str(group["id"]), tuple(group["fields"])) for group in group_payload),
        train_standard_deviations=train_standard_deviations,
        normalization_denominator_floor=float(denominator_floor),
        group_metric_columns=tuple(group_metric_columns),
        physics_kind=physics_kind,
    )


def _require_npz_string_vector(value: np.ndarray, *, label: str) -> tuple[str, ...]:
    """Return an exact one-dimensional string vector from an NPZ payload."""
    if value.ndim != 1:
        msg = f"{label} must be a one-dimensional string array."
        raise ArtifactCacheError(msg)
    items = value.tolist()
    if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
        msg = f"{label} must contain only non-empty strings without object pickles."
        raise ArtifactCacheError(msg)
    return tuple(items)


def _require_finite_npz_array(value: np.ndarray, *, label: str, rank: int) -> np.ndarray:
    """Return a numeric finite NPZ array with the required rank."""
    if value.ndim != rank or not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        msg = f"{label} must be a rank-{rank} numeric array."
        raise ArtifactCacheError(msg)
    if not np.isfinite(value).all():
        msg = f"{label} contains non-finite values."
        raise ArtifactCacheError(msg)
    return value


def _validate_npz_payload(
    path: Path,
    *,
    case_index: int,
    source_index: int,
    split_local_index: int,
    contract: _EvaluatorArtifactContract,
) -> tuple[dict[str, Any], tuple[str, ...] | None]:
    """
    Validate one per-case NPZ against its row identity and evaluator contract.

    Generic tasks require exact input/output field declarations and finite raw,
    prediction, target, and error tensors. Steady Brinkman tasks additionally
    require permeability, boundary pressure, coordinates, and residual arrays
    with the declared grid shape. Object arrays and undeclared keys are rejected.

    Returns
    -------
    tuple[dict[str, Any], tuple[str, ...] | None]
        Identity-free case metadata and, for Brinkman artifacts, permeability
        channel names used for cross-checking the corresponding Parquet row.

    Raises
    ------
    ArtifactCacheError
        If the file is unreadable or any schema, identity, field, unit, shape,
        finiteness, or numerical-consistency check fails.

    """
    common_fields = {
        "case_index",
        "source_index",
        "split_local_index",
        "meta",
        "pred",
        "gt",
        "err",
        "artifact_fields",
        "artifact_units",
        "x_raw",
        "y_raw",
        "input_fields",
        "output_fields",
        "output_units",
    }
    steady_fields = {
        "kappa_encoded",
        "kappa",
        "kappa_names",
        "p_bc",
        "coordinates",
        "Rx",
        "Ry",
        "div_u",
        "div_eps_u",
    }
    required_fields = common_fields | steady_fields if contract.is_steady_brinkman else common_fields
    try:
        with np.load(path, allow_pickle=False) as artifact:
            payload = {name: np.asarray(artifact[name]) for name in artifact.files}
    except (OSError, TypeError, ValueError, KeyError) as error:
        msg = f"Cached NPZ is unreadable or incompatible: {path}: {error}"
        raise ArtifactCacheError(msg) from error

    fields = set(payload)
    missing = required_fields.difference(fields)
    unexpected = fields.difference(required_fields)
    if missing or unexpected:
        msg = f"Cached NPZ schema mismatch for {path}: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        raise ArtifactCacheError(msg)

    actual_identity: dict[str, int] = {}
    for name in ("case_index", "source_index", "split_local_index"):
        value = payload[name]
        if value.shape != () or not np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_):
            msg = f"Cached NPZ {path} field {name!r} must be one integer scalar."
            raise ArtifactCacheError(msg)
        actual_identity[name] = int(value.item())
    expected_identity = {
        "case_index": case_index,
        "source_index": source_index,
        "split_local_index": split_local_index,
    }
    if actual_identity != expected_identity:
        msg = f"Cached NPZ identity mismatch for {path}: expected {expected_identity}, got {actual_identity}."
        raise ArtifactCacheError(msg)

    metadata = _require_metadata_without_identity(
        payload["meta"].item(),
        label=f"Cached NPZ {path}",
    )
    input_fields = _require_npz_string_vector(
        payload["input_fields"],
        label=f"Cached NPZ {path} input_fields",
    )
    output_fields = _require_npz_string_vector(
        payload["output_fields"],
        label=f"Cached NPZ {path} output_fields",
    )
    output_units = _require_npz_string_vector(
        payload["output_units"],
        label=f"Cached NPZ {path} output_units",
    )
    artifact_fields = _require_npz_string_vector(
        payload["artifact_fields"],
        label=f"Cached NPZ {path} artifact_fields",
    )
    artifact_units = _require_npz_string_vector(
        payload["artifact_units"],
        label=f"Cached NPZ {path} artifact_units",
    )
    expected_artifact_fields = (*contract.output_fields, "U") if contract.is_steady_brinkman else contract.output_fields
    expected_artifact_units = (*contract.output_units, contract.velocity_unit) if contract.is_steady_brinkman else contract.output_units
    if (
        input_fields != contract.input_fields
        or output_fields != contract.output_fields
        or output_units != contract.output_units
        or artifact_fields != expected_artifact_fields
        or artifact_units != expected_artifact_units
    ):
        msg = f"Cached NPZ declared fields or units do not match evaluator provenance: {path}"
        raise ArtifactCacheError(msg)

    prediction = _require_finite_npz_array(payload["pred"], label=f"Cached NPZ {path} pred", rank=3)
    target = _require_finite_npz_array(payload["gt"], label=f"Cached NPZ {path} gt", rank=3)
    error_array = _require_finite_npz_array(payload["err"], label=f"Cached NPZ {path} err", rank=3)
    inputs = _require_finite_npz_array(payload["x_raw"], label=f"Cached NPZ {path} x_raw", rank=3)
    raw_targets = _require_finite_npz_array(payload["y_raw"], label=f"Cached NPZ {path} y_raw", rank=3)
    expected_prediction_channels = len(expected_artifact_fields)
    if prediction.shape != target.shape or prediction.shape != error_array.shape:
        msg = f"Cached NPZ pred/gt/err shapes differ: {path}"
        raise ArtifactCacheError(msg)
    if prediction.shape[0] != expected_prediction_channels:
        msg = f"Cached NPZ prediction channels do not match the task artifact schema: {path}"
        raise ArtifactCacheError(msg)
    if inputs.shape[0] != len(contract.input_fields) or raw_targets.shape[0] != len(contract.output_fields):
        msg = f"Cached NPZ raw tensor channels do not match declared task fields: {path}"
        raise ArtifactCacheError(msg)
    if inputs.shape[1:] != raw_targets.shape[1:] or prediction.shape[1:] != raw_targets.shape[1:]:
        msg = f"Cached NPZ raw and prediction spatial shapes differ: {path}"
        raise ArtifactCacheError(msg)
    if not np.allclose(error_array, prediction - target, rtol=1e-6, atol=1e-8):
        msg = f"Cached NPZ err is not pred - gt: {path}"
        raise ArtifactCacheError(msg)

    kappa_names: tuple[str, ...] | None = None
    if contract.is_steady_brinkman:
        spatial_shape = raw_targets.shape[1:]
        kappa_encoded = _require_finite_npz_array(
            payload["kappa_encoded"],
            label=f"Cached NPZ {path} kappa_encoded",
            rank=3,
        )
        kappa = _require_finite_npz_array(payload["kappa"], label=f"Cached NPZ {path} kappa", rank=3)
        kappa_names = _require_npz_string_vector(
            payload["kappa_names"],
            label=f"Cached NPZ {path} kappa_names",
        )
        if kappa.shape != kappa_encoded.shape or kappa.shape != (len(kappa_names), *spatial_shape):
            msg = f"Cached NPZ permeability arrays do not match kappa_names or spatial shape: {path}"
            raise ArtifactCacheError(msg)
        p_bc = _require_finite_npz_array(payload["p_bc"], label=f"Cached NPZ {path} p_bc", rank=3)
        if p_bc.shape != (1, *spatial_shape):
            msg = f"Cached NPZ p_bc shape does not match the task grid: {path}"
            raise ArtifactCacheError(msg)
        coordinates = _require_finite_npz_array(
            payload["coordinates"],
            label=f"Cached NPZ {path} coordinates",
            rank=3,
        )
        if coordinates.shape != (2, *spatial_shape):
            msg = f"Cached NPZ coordinates must contain x/y fields on the task grid: {path}"
            raise ArtifactCacheError(msg)
        for name in ("Rx", "Ry", "div_u", "div_eps_u"):
            residual = _require_finite_npz_array(
                payload[name],
                label=f"Cached NPZ {path} {name}",
                rank=2,
            )
            if residual.shape != spatial_shape:
                msg = f"Cached NPZ {name} shape does not match the task grid: {path}"
                raise ArtifactCacheError(msg)

    return metadata, kappa_names


def _require_finite_parquet_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Require real, finite scalar metric values in every selected Parquet column."""
    for column in columns:
        values = df.loc[:, column].tolist()
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            msg = f"Cached Parquet column {column!r} must contain only real numbers."
            raise ArtifactCacheError(msg)
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            msg = f"Cached Parquet column {column!r} contains non-finite values."
            raise ArtifactCacheError(msg)


def _require_optional_inference_times(df: pd.DataFrame) -> None:
    """Require inference times to be null (CPU) or finite non-negative numbers."""
    for value in df.loc[:, "inference_time_ms"].tolist():
        if value is None or (isinstance(value, Real) and np.isnan(float(value))):
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) < 0.0:
            msg = "Cached Parquet inference_time_ms values must be null or finite non-negative numbers."
            raise ArtifactCacheError(msg)


def _parquet_schema(contract: _EvaluatorArtifactContract) -> tuple[set[str], tuple[str, ...]]:
    """
    Derive the closed Parquet schema and numeric validation set for one task.

    Generic tasks receive task-named physical and normalized sufficient evidence,
    predictive diagnostics, and declared group metrics. The steady-Brinkman
    contract additionally requires speed, permeability, residual, continuity,
    momentum, and pressure-boundary diagnostics.
    """
    common = {
        "artifact_schema_version",
        "task_id",
        "output_fields",
        "output_units",
        "case_index",
        "source_index",
        "split_local_index",
        "npz_path",
        "meta",
        "inference_time_ms",
    }
    physical_columns = tuple(column for field in contract.output_fields for column in contracts.physical_statistic_columns(field))
    normalized_columns = tuple(column for field in contract.output_fields for column in contracts.normalized_statistic_columns(field))
    if contract.is_steady_brinkman:
        metrics: tuple[str, ...] = (
            "rel_l2",
            "rel_h1",
            *physical_columns,
            *normalized_columns,
            *contract.group_metric_columns,
            "physical_rmse_speed_magnitude",
            "momentum_residual_mse",
            "div_velocity_mse",
            "div_eps_velocity_mse",
            "pressure_boundary_mse",
            "pressure_inlet_mse",
            "pressure_outlet_mean_square",
        )
        return common | set(metrics) | {"kappa_names"}, metrics
    metrics = (
        "rel_l2",
        "rel_h1",
        *physical_columns,
        *normalized_columns,
        *contract.group_metric_columns,
    )
    return common | set(metrics), metrics


def _require_parquet_string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    """Normalize a Parquet nested string sequence without accepting scalar text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        msg = f"{label} must contain a sequence of non-empty strings."
        raise ArtifactCacheError(msg)
    return tuple(value)


def _load_validated_artifact_cache(  # noqa: C901, PLR0912, PLR0915
    *,
    save_root: Path,
    parquet_path: Path,
    npz_dir: Path,
    request: ArtifactRequest,
) -> pd.DataFrame:
    """
    Load an artifact cache only after validating all persisted evidence.

    Validation covers the provenance schema and scientific request, the digest
    manifest, the exact Parquet schema and aggregate, ordered saved-split
    membership, one-to-one NPZ membership, per-case payloads, and metadata
    agreement between both storage formats. No partial or merely readable cache
    is accepted as reusable.

    Parameters
    ----------
    save_root, parquet_path, npz_dir : pathlib.Path
        Exact cache root and its staged or published payload locations.
    request : ArtifactRequest
        Current semantic identity and ordered source membership.

    Returns
    -------
    pandas.DataFrame
        Validated table annotated with its artifact root and full provenance.

    Raises
    ------
    ArtifactCacheError
        If any provenance, digest, schema, aggregate, identity, membership, or
        payload invariant is violated.

    """
    provenance_path = contracts.artifact_provenance_path(save_root)
    if not provenance_path.is_file():
        msg = f"Existing artifact cache has no provenance sidecar: {provenance_path}. Refusing to trust or overwrite invalid/partial artifacts."
        raise ArtifactCacheError(msg)
    if not parquet_path.is_file():
        msg = f"Artifact cache is incomplete (Parquet missing): {parquet_path}"
        raise ArtifactCacheError(msg)

    stored_provenance = dict(_read_artifact_provenance(provenance_path))
    _require_current_provenance_schema(stored_provenance)
    stored_outputs = stored_provenance.pop("outputs", None)
    stored_aggregate = stored_provenance.pop("aggregate", None)
    expected_scientific = _scientific_provenance(request.provenance)
    actual_scientific = _scientific_provenance(stored_provenance)
    if actual_scientific != expected_scientific:
        expected_json = json.dumps(expected_scientific, indent=2, sort_keys=True)
        actual_json = json.dumps(actual_scientific, indent=2, sort_keys=True)
        msg = (
            f"Artifact provenance is incompatible: {provenance_path}\n"
            f"Expected:\n{expected_json}\nActual:\n{actual_json}\n"
            "Refusing to trust or overwrite the existing cache."
        )
        raise ArtifactCacheError(msg)

    try:
        computed_outputs = contracts.artifact_output_manifest(save_root)
    except (OSError, RuntimeError) as error:
        msg = f"Artifact payload digest manifest cannot be recomputed for {save_root}: {error}"
        raise ArtifactCacheError(msg) from error
    if stored_outputs != computed_outputs:
        msg = (
            f"Artifact payload digest manifest mismatch: {provenance_path}. "
            "Refusing to trust or overwrite changed, missing, or unexpected payload files."
        )
        raise ArtifactCacheError(msg)

    contract = _evaluator_artifact_contract(request.provenance)
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as error:
        msg = f"Cached Parquet is unreadable: {parquet_path}: {error}"
        raise ArtifactCacheError(msg) from error

    expected_columns, finite_metric_columns = _parquet_schema(contract)
    actual_columns = list(df.columns)
    if not df.columns.is_unique or set(actual_columns) != expected_columns:
        missing = sorted(expected_columns.difference(actual_columns))
        unexpected = sorted(set(actual_columns).difference(expected_columns))
        msg = f"Cached Parquet schema mismatch: missing={missing}, unexpected={unexpected}, duplicate_columns={not df.columns.is_unique}."
        raise ArtifactCacheError(msg)
    _require_finite_parquet_columns(df, finite_metric_columns)
    _require_optional_inference_times(df)
    schema_values = _require_identity_values(df, "artifact_schema_version")
    if schema_values != (contracts.ARTIFACT_SCHEMA_VERSION,) * len(df):
        msg = "Cached Parquet rows do not declare the current artifact schema version."
        raise ArtifactCacheError(msg)
    if tuple(df.loc[:, "task_id"].tolist()) != (contract.task_id,) * len(df):
        msg = "Cached Parquet task_id values do not match evaluator provenance."
        raise ArtifactCacheError(msg)
    for row_position in range(len(df)):
        row_fields = _require_parquet_string_sequence(
            df.iloc[row_position].loc["output_fields"],
            label=f"Cached Parquet row {row_position} output_fields",
        )
        row_units = _require_parquet_string_sequence(
            df.iloc[row_position].loc["output_units"],
            label=f"Cached Parquet row {row_position} output_units",
        )
        if row_fields != contract.output_fields or row_units != contract.output_units:
            msg = f"Cached Parquet row {row_position} output fields/units do not match evaluator provenance."
            raise ArtifactCacheError(msg)

    output_groups = [{"id": group_id, "fields": list(fields)} for group_id, fields in contract.output_groups]
    try:
        computed_aggregate = contracts.aggregate_normalized_group_macro_rmse(
            df,
            output_groups=output_groups,
            train_standard_deviations=contract.train_standard_deviations,
            normalization_denominator_floor=contract.normalization_denominator_floor,
        )
    except (KeyError, TypeError, ValueError, RuntimeError, FloatingPointError) as error:
        msg = f"Cached Parquet group-objective evidence is invalid: {error}"
        raise ArtifactCacheError(msg) from error
    if stored_aggregate != computed_aggregate:
        msg = "Artifact aggregate does not match physical Parquet sufficient statistics and saved train scales."
        raise ArtifactCacheError(msg)
    evaluator = request.provenance.get("evaluator")
    objective = evaluator.get("objective") if isinstance(evaluator, Mapping) else None
    if not isinstance(objective, Mapping):
        msg = "Artifact evaluator must declare the resolved primary objective."
        raise ArtifactCacheError(msg)
    objective_fields = objective.get("fields")
    resolved_objective_fields = contract.output_fields if objective_fields == "all" else tuple(objective_fields or ())
    expected_semantics = {
        "id": computed_aggregate["objective_id"],
        "kind": computed_aggregate["kind"],
        "space": computed_aggregate["space"],
        "reduction": computed_aggregate["reduction"],
        "direction": computed_aggregate["direction"],
    }
    actual_semantics = {key: objective.get(key) for key in expected_semantics}
    if actual_semantics != expected_semantics or resolved_objective_fields != contract.output_fields:
        msg = "Artifact aggregate definition contradicts the resolved primary objective."
        raise ArtifactCacheError(msg)

    expected_source_indices = request.source_indices
    expected_split_local_indices = tuple(range(len(expected_source_indices)))
    expected_case_indices = tuple(source_index + 1 for source_index in expected_source_indices)
    if len(df) != len(expected_source_indices):
        msg = f"Cached Parquet has {len(df)} rows, but {len(expected_source_indices)} were expected."
        raise ArtifactCacheError(msg)
    if _require_identity_values(df, "source_index") != expected_source_indices:
        msg = "Cached Parquet ordered source_index values do not match the selected saved split."
        raise ArtifactCacheError(msg)
    if _require_identity_values(df, "split_local_index") != expected_split_local_indices:
        msg = "Cached Parquet split_local_index values are not the expected contiguous saved-split order."
        raise ArtifactCacheError(msg)
    if _require_identity_values(df, "case_index") != expected_case_indices:
        msg = "Cached Parquet case_index values do not equal source_index + 1."
        raise ArtifactCacheError(msg)

    expected_npz_paths = tuple(npz_dir / f"case_{case_index:04d}.npz" for case_index in expected_case_indices)
    actual_npz_paths = tuple(sorted(npz_dir.glob("*.npz")))
    if {_normalise_path(path) for path in actual_npz_paths} != {_normalise_path(path) for path in expected_npz_paths}:
        msg = f"Cached NPZ membership/count does not match the selected split: expected {len(expected_npz_paths)}, found {len(actual_npz_paths)}."
        raise ArtifactCacheError(msg)

    for row_position, (source_index, split_local_index, case_index, expected_npz_path) in enumerate(
        zip(
            expected_source_indices,
            expected_split_local_indices,
            expected_case_indices,
            expected_npz_paths,
            strict=True,
        )
    ):
        row = df.iloc[row_position]
        raw_npz_path = row.loc["npz_path"]
        try:
            resolved_npz_path = contracts.resolve_case_payload_path(
                save_root,
                raw_npz_path,
                expected_filename=expected_npz_path.name,
            )
        except (FileNotFoundError, TypeError, ValueError) as error:
            msg = f"Cached Parquet npz_path mismatch at row {row_position}: {raw_npz_path!r}: {error}"
            raise ArtifactCacheError(msg) from error
        if _normalise_path(resolved_npz_path) != _normalise_path(expected_npz_path):
            msg = f"Cached Parquet npz_path resolves outside current membership at row {row_position}."
            raise ArtifactCacheError(msg)
        parquet_metadata = _require_metadata_without_identity(
            row.loc["meta"],
            label=f"Cached Parquet row {row_position}",
        )
        npz_metadata, npz_kappa_names = _validate_npz_payload(
            expected_npz_path,
            case_index=case_index,
            source_index=source_index,
            split_local_index=split_local_index,
            contract=contract,
        )
        if parquet_metadata != npz_metadata:
            msg = f"Cached Parquet and NPZ metadata differ at row {row_position}."
            raise ArtifactCacheError(msg)
        if contract.is_steady_brinkman:
            parquet_kappa_names = _require_parquet_string_sequence(
                row.loc["kappa_names"],
                label=f"Cached Parquet row {row_position} kappa_names",
            )
            if parquet_kappa_names != npz_kappa_names:
                msg = f"Cached Parquet and NPZ kappa_names differ at row {row_position}."
                raise ArtifactCacheError(msg)

    df.loc[:, "npz_path"] = [str(path.resolve()) for path in expected_npz_paths]
    df.attrs["artifact_root"] = str(save_root.resolve())
    df.attrs["artifact_provenance"] = {
        **stored_provenance,
        "aggregate": stored_aggregate,
        "outputs": stored_outputs,
    }
    return df


def _create_artifact_staging_root(save_root: Path) -> Path:
    """Create one unique sibling staging directory for a locked target build."""
    save_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            dir=save_root.parent,
            prefix=f".{save_root.name}.staging.",
        )
    )


def _publish_staged_artifact(*, run_dir: Path, save_root: Path, staging_root: Path) -> None:
    """
    Atomically publish one validated sibling stage at an exact run-owned target.

    The stage must carry its provenance completion marker. An existing target is
    first renamed to a unique backup. If publication fails, that backup is moved
    back before the exception escapes. A successful replacement removes the
    backup on a best-effort basis.

    Raises
    ------
    ValueError
        If the stage is not the expected uniquely named sibling of the target.
    ArtifactCacheError
        If the stage lacks its completion marker.

    """
    _analysis_root, target = _validated_artifact_target(run_dir=run_dir, save_root=save_root)
    stage = _normalise_path(staging_root)
    if stage.parent != target.parent or not stage.name.startswith(f".{target.name}.staging."):
        msg = f"Artifact staging root is not the expected sibling of the exact target: {stage}"
        raise ValueError(msg)
    if not contracts.artifact_provenance_path(stage).is_file():
        msg = f"Refusing to publish an artifact stage without its completion marker: {stage}"
        raise ArtifactCacheError(msg)

    backup = target.with_name(f".{target.name}.backup.{uuid4().hex}")
    moved_previous = False
    try:
        if target.exists():
            target.replace(backup)
            moved_previous = True
        stage.replace(target)
    except BaseException:
        if moved_previous and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    else:
        if moved_previous:
            with suppress(OSError):
                shutil.rmtree(backup)


def _rebuild_artifact_target_locked(*, run_dir: Path, save_root: Path) -> None:
    """
    Remove exactly one computed artifact target for an explicit rebuild.

    Parameters
    ----------
    run_dir : Path
        Owning completed run directory.
    save_root : Path
        Exact ID or named OOD artifact target below ``run_dir/analysis``.

    """
    _analysis_root, target = _validated_artifact_target(
        run_dir=run_dir,
        save_root=save_root,
    )
    if target.exists():
        shutil.rmtree(target)


def rebuild_artifact_target(*, run_dir: Path, save_root: Path) -> None:
    """
    Remove one exact run-owned artifact target under both writer leases.

    Parameters
    ----------
    run_dir : pathlib.Path
        Owning completed run directory.
    save_root : pathlib.Path
        Exact ``analysis/id`` or ``analysis/ood/<dataset>`` target.

    Raises
    ------
    ValueError
        If the target is outside the owning run, is a symlink alias, or is not an
        exact ID/named-OOD artifact leaf.

    Notes
    -----
    This destructive helper is used only by explicit ``--rebuild`` handling. It
    never scans or removes sibling cache targets.

    """
    lock_path = _artifact_lock_path(run_dir=run_dir, save_root=save_root)
    with (
        experiments.run.run_reader_lease(run_dir),
        common.locking.exclusive_file_lock(lock_path, blocking=True),
    ):
        _rebuild_artifact_target_locked(run_dir=run_dir, save_root=save_root)


def _run_or_load_artifacts_locked(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    metadata_root: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load or generate artifacts for one run and saved split.

    Reuses existing Parquet+NPZ artifacts only after exact provenance and
    payload validation. If artifacts do not exist, runs split-aware inference via
    learning.inference.context.load_inference_context() and generates
    Parquet+NPZ artifacts via generation.generate_artifacts().

    Parameters
    ----------
    run_dir : Path
        Current run directory satisfying the saved-run contract.
    dataset_name : str
        Logical dataset name used for artifact file naming.
    split : {"eval", "ood"}
        Saved split role to load. ID evaluation uses "eval", while OOD uses "ood".
    device_resolution : DeviceResolution
        Immutable device decision resolved once at the artifact boundary.
    dataset_root : Path
        Current final-dataset root.
    metadata_root : Path
        Current validated dataset-metadata root.
    rebuild : bool, optional
        Force staged regeneration and atomically replace only this target.

    Returns
    -------
    pandas.DataFrame
        Validated non-empty artifact summary DataFrame.

    Notes
    -----
    - Artifacts cached in analysis/id or analysis/ood/<dataset_name>/.
    - Normal eval and OOD evaluation always use saved split membership.
    - Missing, incompatible or partial cache provenance fails loudly and is never overwritten.
    - Inference and generation failures propagate to the caller.

    """
    dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")

    save_root = _artifact_save_root(run_dir=run_dir, dataset_name=dataset_name, split=split)
    npz_dir = save_root / "npz"
    parquet_path = save_root / f"{dataset_name}.parquet"

    request = _build_artifact_request(
        run_dir=run_dir,
        dataset_name=dataset_name,
        split=split,
        device_resolution=device_resolution,
        dataset_root=dataset_root,
        metadata_root=metadata_root,
    )
    task = experiments.config.loader.validate_resolved_task_contract(_load_run_config(run_dir))

    print(f"[RUN] {run_dir.name} | split={split} | dataset={dataset_name}")
    print(f"      run_dir={run_dir}")
    print(f"      save_root={save_root}")

    if not rebuild and _cache_has_outputs(save_root=save_root, parquet_path=parquet_path, npz_dir=npz_dir):
        print(f"[VALIDATE] {run_dir.name} | {split} | {dataset_name} (existing cache)")
        df = _load_validated_artifact_cache(
            save_root=save_root,
            parquet_path=parquet_path,
            npz_dir=npz_dir,
            request=request,
        )
        _report_runtime_comparison(save_root=save_root, request=request)
        print(f"[LOAD] {run_dir.name} | {split} | {dataset_name} (validated scientific cache)")
        return df

    staging_root = _create_artifact_staging_root(save_root)
    staging_npz_dir = staging_root / "npz"
    staging_parquet_path = staging_root / f"{dataset_name}.parquet"
    timing_cases: list[dict[str, Any]] = []
    timing_enabled = bool(request.case_ids)
    try:
        from src import learning  # noqa: PLC0415

        model, loader, processor, device = learning.inference.context.load_inference_context_with_resolution(
            run_dir=run_dir,
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            split=split,
            batch_size=request.batch_size,
        )
        try:
            if timing_enabled:
                try:
                    representative_batch = next(iter(loader))
                    timing.warm_up_forward(
                        representative_batch=representative_batch,
                        model=model,
                        processor=processor,
                        device=device,
                        passes=timing.WARMUP_PASSES,
                    )
                except (KeyError, OSError, RuntimeError, StopIteration, TypeError, ValueError) as error:
                    timing_enabled = False
                    print(f"[TIMING] warmup unavailable. Scientific generation continues: {error}")
            generation.generate_artifacts(
                task=task,
                model=model,
                loader=loader,
                processor=processor,
                device=device,
                save_root=staging_root,
                publication_root=save_root,
                dataset_name=dataset_name,
                provenance=request.provenance,
                timing_cases=timing_cases if timing_enabled else None,
                timing_case_ids=request.case_ids if timing_enabled else None,
            )
            if timing_enabled:
                try:
                    dataset_identity, model_identity = _runtime_identities(request)
                    comsol_payload, manifest_sha256, unavailable_reason = _resolve_comsol_timing(request)
                    comparison = timing.build_runtime_comparison(
                        split_role=split,
                        dataset_identity=dataset_identity,
                        model_identity=model_identity,
                        neural_runtime=timing.neural_runtime_metadata(
                            device_metadata=device_resolution.as_dict(),
                            model=model,
                        ),
                        batch_size=request.batch_size,
                        cases=timing_cases,
                        comsol_timing=comsol_payload,
                        batch_manifest_sha256=manifest_sha256,
                        unavailable_reason=unavailable_reason,
                    )
                    timing.write_runtime_comparison(staging_root, comparison)
                except (ArtifactCacheError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                    print(f"[TIMING] sidecar publication unavailable. Scientific generation continues: {error}")
        finally:
            del model, loader, processor
            cleanup_runtime(device)

        _load_validated_artifact_cache(
            save_root=staging_root,
            parquet_path=staging_parquet_path,
            npz_dir=staging_npz_dir,
            request=request,
        )
        _publish_staged_artifact(
            run_dir=run_dir,
            save_root=save_root,
            staging_root=staging_root,
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    result = _load_validated_artifact_cache(
        save_root=save_root,
        parquet_path=parquet_path,
        npz_dir=npz_dir,
        request=request,
    )
    _report_runtime_comparison(save_root=save_root, request=request)
    return result


def validate_artifact_upload_source(
    *,
    run_dir: Path,
    artifact_root: Path,
) -> Mapping[str, Any]:
    """
    Validate one explicit complete current artifact target for bounded upload.

    Parameters
    ----------
    run_dir : pathlib.Path
        Authoritative completed run that must own ``artifact_root``.
    artifact_root : pathlib.Path
        Exact ``analysis/id`` or named ``analysis/ood/<dataset>`` target.

    Returns
    -------
    Mapping[str, Any]
        Admitted current provenance, including the verified payload manifest.

    Raises
    ------
    ArtifactCacheError
        If schemas, digests, payloads, or completed-run identities disagree.
    ValueError
        If the target escapes the run analysis tree or is not an exact leaf.

    Notes
    -----
    The gate never scans sibling targets, regenerates artifacts, or contacts W&B.

    """
    run_path = Path(run_dir).resolve()
    root = Path(artifact_root).resolve()
    _analysis_root, validated_target = _validated_artifact_target(
        run_dir=run_path,
        save_root=root,
    )
    if _normalise_path(validated_target) != _normalise_path(root):
        msg = f"Artifact upload target does not resolve to the explicit artifact root: {root}"
        raise ArtifactCacheError(msg)
    provenance_path = contracts.artifact_provenance_path(root)
    provenance = dict(_read_artifact_provenance(provenance_path))
    _require_current_provenance_schema(provenance)
    stored_outputs = provenance.get("outputs")
    try:
        computed_outputs = contracts.artifact_output_manifest(root)
    except (OSError, RuntimeError) as error:
        msg = f"Artifact upload source payload manifest cannot be recomputed: {root}: {error}"
        raise ArtifactCacheError(msg) from error
    if stored_outputs != computed_outputs:
        msg = f"Artifact upload source payload manifest mismatch: {provenance_path}"
        raise ArtifactCacheError(msg)

    completed = experiments.run.validate_completed_run(run_path)
    summary = completed["summary"]
    config = completed["config"]
    task = experiments.config.loader.validate_resolved_task_contract(config)
    run_identity = provenance.get("run")
    expected_identity = {
        "name": config["run"]["name"],
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "effective_config_digest": summary["effective_config_digest"],
        "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
        "normalizer_sha256": summary["normalizer_sha256"],
    }
    if not isinstance(run_identity, Mapping) or {key: run_identity.get(key) for key in expected_identity} != expected_identity:
        msg = "Artifact upload source is not current for the authoritative completed run."
        raise ArtifactCacheError(msg)
    return provenance


def upload_completed_artifact(
    session: experiments.tracking.WandbSession,
    *,
    run_dir: Path,
    artifact_root: Path,
    media_files: Mapping[str, Path] | None = None,
    tables: Mapping[str, Any] | None = None,
) -> None:
    """
    Upload validated provenance and explicitly supplied curated media.

    Parameters
    ----------
    session : experiments.tracking.WandbSession
        Initialized tracking observer whose persisted upload policy is honored.
    run_dir : pathlib.Path
        Completed run that owns the artifact target.
    artifact_root : pathlib.Path
        Exact current artifact root validated before any remote operation.
    media_files : Mapping[str, pathlib.Path] | None, optional
        Caller-rendered curated media. No files are discovered implicitly.
    tables : Mapping[str, Any] | None, optional
        Caller-built curated table payloads.

    Raises
    ------
    ArtifactCacheError, ValueError
        If local artifact admission or containment fails.

    Notes
    -----
    Local validation always completes first. Plot construction and W&B session
    lifecycle remain owned by callers and the tracking adapter respectively.

    """
    validate_artifact_upload_source(
        run_dir=run_dir,
        artifact_root=artifact_root,
    )
    if bool(session.upload_settings["evaluation_artifacts"]):
        session.upload_files({"artifact_provenance": contracts.artifact_provenance_path(artifact_root)})
    session.upload_post_artifact(
        artifact_root=artifact_root,
        media_files=media_files,
        tables=tables,
    )


def run_or_load_artifacts(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    device_resolution: DeviceResolution,
    dataset_root: Path,
    metadata_root: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Validate, reuse, or atomically generate one split-qualified artifact cache.

    Parameters
    ----------
    run_dir : pathlib.Path
        Current evaluable run with immutable config/checkpoint/split identity.
    dataset_name : str
        Logical dataset required by the saved split metadata.
    split : {"eval", "ood"}
        Saved membership role. ``eval`` publishes under ``analysis/id``.
    device_resolution : DeviceResolution
        Device decision resolved once before any inference allocation.
    dataset_root : pathlib.Path
        Final training-dataset root.
    metadata_root : pathlib.Path
        Validated dataset provenance and timing root.
    rebuild : bool, optional
        Replace only the observed target. A concurrent newer publication wins and
        is validated instead of being deleted.

    Returns
    -------
    pandas.DataFrame
        Strict current artifact table carrying validated provenance attrs.

    Raises
    ------
    ArtifactCacheError
        If existing schema, identity, aggregate, or payload evidence is invalid.
    FileNotFoundError
        If required evaluable-run evidence or a final-dataset artifact is absent.
    TypeError, ValueError, RuntimeError
        If request semantics, saved membership, device resolution, or generated
        payloads violate the current contract.

    Notes
    -----
    Both the run writer lease and target-specific lock serialize publication.
    Rebuilds generate and validate a sibling stage before replacing the target.
    Provenance remains the final completion marker.

    """
    logical_dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    save_root = _artifact_save_root(
        run_dir=run_dir,
        dataset_name=logical_dataset_name,
        split=split,
    )
    completion_path = contracts.artifact_provenance_path(save_root)
    observed_completion = _completion_marker_identity(completion_path) if rebuild else None
    lock_path = _artifact_lock_path(run_dir=run_dir, save_root=save_root)
    with (
        experiments.run.run_reader_lease(run_dir),
        common.locking.exclusive_file_lock(lock_path, blocking=True),
    ):
        current_completion = _completion_marker_identity(completion_path)
        effective_rebuild = rebuild and (current_completion is None or current_completion == observed_completion)
        return _run_or_load_artifacts_locked(
            run_dir=run_dir,
            dataset_name=logical_dataset_name,
            split=split,
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
            rebuild=effective_rebuild,
        )


def _upload_published_artifacts(
    *,
    plan: RunArtifactPlan,
    device_resolution: DeviceResolution,
    id_frame: pd.DataFrame,
    ood_frame: pd.DataFrame,
) -> None:
    """Upload only explicitly enabled completed-run provenance and curated media."""
    if not plan.is_completed:
        return
    config = _load_run_config(plan.run_dir)
    settings = config.get("tracking", {}).get("wandb")
    if not isinstance(settings, Mapping):
        msg = "Completed run config must contain tracking.wandb."
        raise TypeError(msg)
    upload = settings.get("upload")
    if settings.get("mode") == "disabled" or not isinstance(upload, Mapping) or not bool(upload.get("evaluation_artifacts")):
        return

    started_at = datetime.now(UTC)
    runtime_session_id = uuid4().hex
    experiments.run.append_runtime_session(
        plan.run_dir,
        device_resolution,
        started_at=started_at,
        session_id=runtime_session_id,
        tracking_state=experiments.run.initial_tracking_state(config),
    )
    summary = experiments.run.read_run_summary(plan.run_dir)
    persisted_run_id, last_logged_epoch = experiments.tracking.persisted_wandb_identity(summary)

    def state_updater(updates: Mapping[str, Any]) -> None:
        """Persist observer-only facts in this artifact-upload runtime session."""
        experiments.run.update_runtime_session(
            plan.run_dir,
            runtime_session_id,
            updates,
        )

    session = experiments.tracking.initialize_wandb(
        config,
        run_dir=plan.run_dir,
        semantic_config={},
        resume=True,
        persisted_run_id=persisted_run_id,
        previous_last_logged_epoch=last_logged_epoch,
        state_updater=state_updater,
    )
    upload_error: BaseException | None = None
    try:
        artifact_specs: tuple[tuple[ArtifactSplit, str], ...] = (
            ("eval", plan.id_dataset_name),
            ("ood", plan.ood_dataset_name),
        )
        artifact_roots = {
            split: _artifact_save_root(
                run_dir=plan.run_dir,
                dataset_name=dataset_name,
                split=split,
            )
            for split, dataset_name in artifact_specs
        }
        for artifact_root in artifact_roots.values():
            validate_artifact_upload_source(
                run_dir=plan.run_dir,
                artifact_root=artifact_root,
            )
            session.upload_files({"artifact_provenance": contracts.artifact_provenance_path(artifact_root)})

        from src.analysis.evaluation import evaluation_dataframe  # noqa: PLC0415
        from src.analysis.presentation import curated  # noqa: PLC0415

        datasets_eval = {
            f"{plan.run_dir.name} ID": evaluation_dataframe.build_eval_df(id_frame),
            f"{plan.run_dir.name} OOD": evaluation_dataframe.build_eval_df(ood_frame),
        }
        with tempfile.TemporaryDirectory(prefix="grainlegumes-curated-analysis-") as temporary_directory:
            bundle = curated.render_curated_analysis(
                datasets=datasets_eval,
                output_dir=temporary_directory,
            )
            session.upload_post_artifact(
                artifact_root=artifact_roots["eval"],
                media_files=bundle.media_files,
                tables=bundle.tables,
            )
    except BaseException as error:
        upload_error = error
        raise
    finally:
        local_summary: Mapping[str, Any] | None = None
        with suppress(Exception):
            local_summary = experiments.run.read_run_summary(plan.run_dir)
        status = str(local_summary.get("status", "completed")) if local_summary is not None else "completed"
        result = None
        if local_summary is not None:
            result = {
                "completed_epoch": local_summary.get("completed_epoch"),
                "global_step": local_summary.get("global_step"),
                "selected_epoch": local_summary.get("selected_epoch"),
                "selected_metrics": local_summary.get("selected_metrics", {}),
                "terminal_epoch": local_summary.get("terminal_epoch"),
                "terminal_metrics": local_summary.get("terminal_metrics", {}),
            }
        try:
            session.finish(
                status=status,
                result=result,
                local_summary=local_summary,
                error=upload_error,
            )
        except experiments.tracking.TrackingError:
            if upload_error is None:
                raise


@dataclass(frozen=True)
class PreparedRunArtifacts:
    """
    Report path-based artifact loading and any explicit local generation.

    Parameters
    ----------
    loaded_run : LoadedRunArtifacts
        Validated run and the selected role-local artifacts.
    role_actions : dict[str, str]
        Selected role to ``reused``, ``generated``, or ``rebuilt`` action.
    artifact_device : str | None
        Exact concrete device supplied to or resolved for generation. ``None``
        means every role was reused without a supplied device decision.

    """

    loaded_run: LoadedRunArtifacts
    role_actions: dict[str, str]
    artifact_device: str | None


def load_or_build_run_artifacts(
    run_dir: Path | str,
    *,
    artifact_roles: tuple[Literal["id", "ood"], ...] = ("id", "ood"),
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
    auto_build_missing: bool = True,
    rebuild_incompatible: bool = False,
    device_policy: str = "cpu",
    device_resolution: DeviceResolution | None = None,
) -> PreparedRunArtifacts:
    """
    Load selected valid artifacts or generate only missing roles locally.

    Existing valid roles are admitted through the read-only path loader before
    any model reconstruction. Missing roles use ``device_resolution`` unchanged
    when supplied; otherwise the explicit device policy is resolved once. Partial,
    corrupt, stale, or incompatible roles fail unless ``rebuild_incompatible``
    deliberately authorizes exact-target replacement. This service never uploads
    to W&B or changes lifecycle status.
    """
    from src import learning  # noqa: PLC0415
    from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader  # noqa: PLC0415

    roles = tuple(artifact_roles)
    if not roles or len(set(roles)) != len(roles) or set(roles).difference({"id", "ood"}):
        msg = "artifact_roles must contain unique values from {'id', 'ood'}."
        raise ValueError(msg)
    path = Path(run_dir).expanduser().resolve()
    current_dataset_root = Path(dataset_root) if dataset_root is not None else common.paths.get_training_raw_root()
    current_metadata_root = Path(metadata_root) if metadata_root is not None else common.paths.get_training_meta_root()
    actions: dict[str, str] = {}
    if device_resolution is not None and not isinstance(device_resolution, learning.device.DeviceResolution):
        msg = f"device_resolution must be a DeviceResolution, got {device_resolution!r}."
        raise TypeError(msg)
    resolution = device_resolution
    split_by_role: dict[str, ArtifactSplit] = {"id": "eval", "ood": "ood"}

    for _attempt in range(len(roles) + 1):
        try:
            loaded = artifact_loader.load_run_artifacts(path, artifact_roles=roles)
            for role in roles:
                actions.setdefault(role, "reused")
            return PreparedRunArtifacts(
                loaded_run=loaded,
                role_actions=actions,
                artifact_device=(str(resolution.device) if resolution is not None else None),
            )
        except artifact_loader.MissingEvaluationArtifactsError as error:
            split_role = getattr(error, "role", None)
            selected_role = "id" if split_role == "eval" else "ood" if split_role == "ood" else None
            if not auto_build_missing or selected_role not in roles:
                raise
            rebuild = False
            action = "generated"
        except artifact_loader.IncompatibleEvaluationArtifactsError as error:
            split_role = getattr(error, "role", None)
            selected_role = "id" if split_role == "eval" else "ood" if split_role == "ood" else None
            if not rebuild_incompatible or selected_role not in roles:
                raise
            rebuild = True
            action = "rebuilt"

        if resolution is None:
            resolution = learning.device.resolve_device(device_policy, path="device_policy")
        plan = load_run_artifact_plan(path)
        dataset_name = plan.id_dataset_name if selected_role == "id" else plan.ood_dataset_name
        run_or_load_artifacts(
            run_dir=path,
            dataset_name=dataset_name,
            split=split_by_role[selected_role],
            device_resolution=resolution,
            dataset_root=current_dataset_root,
            metadata_root=current_metadata_root,
            rebuild=rebuild,
        )
        actions[selected_role] = action
    msg = f"Artifact preparation did not converge for selected roles {roles}: {path}"
    raise RuntimeError(msg)


def build_artifacts(
    *,
    runs_root: Path,
    dataset_root: Path,
    metadata_root: Path | None = None,
    run_names: Iterable[str] | None = None,
    device_policy: str = "auto",
    rebuild: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build or validate ID and OOD artifacts for every selected run.

    Parameters
    ----------
    runs_root : Path
        One evaluable run or a container of run directories.
    dataset_root : Path
        Derived raw root containing immutable final task datasets.
    metadata_root : Path | None, optional
        Bounded validated metadata-root override. Defaults to training ``meta``.
    run_names : Iterable[str] | None, optional
        Explicit run names under ``runs_root``.
    device_policy : {"auto", "cuda", "cpu"}, optional
        Runtime policy. Auto selects usable CUDA and then CPU. CUDA is strict.
        CPU avoids CUDA queries.
    rebuild : bool, optional
        Stage and validate a replacement for each exact selected target. A newer
        concurrent publication is preserved and validated instead.

    Returns
    -------
    dict[str, dict[str, pandas.DataFrame]]
        Validated ``eval`` and ``ood`` frames keyed by run name.

    Raises
    ------
    FileNotFoundError, ArtifactCacheError, TypeError, ValueError, RuntimeError
        If device resolution, run admission, saved dataset identity, cache
        validation, inference, or publication fails.

    Notes
    -----
    Each run's ID and OOD caches are locally authoritative. When persisted W&B
    settings explicitly request evaluation-artifact upload, the function appends
    an observer runtime session and uploads only after both local targets validate.
    Any requested online or offline observer failure propagates.

    """
    from src import learning  # noqa: PLC0415

    device_resolution = learning.device.resolve_device(
        device_policy,
        path="device_policy",
    )
    resolved_metadata_root = Path(metadata_root) if metadata_root is not None else common.paths.get_training_meta_root()
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for run_dir in iter_run_dirs(runs_root, run_names=run_names):
        plan = load_run_artifact_plan(run_dir)
        id_frame = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.id_dataset_name,
            split="eval",
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            metadata_root=resolved_metadata_root,
            rebuild=rebuild,
        )
        ood_frame = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.ood_dataset_name,
            split="ood",
            device_resolution=device_resolution,
            dataset_root=dataset_root,
            metadata_root=resolved_metadata_root,
            rebuild=rebuild,
        )
        results[run_dir.name] = {"eval": id_frame, "ood": ood_frame}
        _upload_published_artifacts(
            plan=plan,
            device_resolution=device_resolution,
            id_frame=id_frame,
            ood_frame=ood_frame,
        )
        cleanup_runtime(device_resolution.device)
    return results
