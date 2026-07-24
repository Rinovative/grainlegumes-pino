"""
===============================================================================
analysis_artifact_service.py
===============================================================================
Discover, validate, rebuild, generate, and reuse split-aware run artifacts.

Responsibilities:
  - Admit only completed and fully validated runs
  - Bind caches to task/config/checkpoint/dataset/split/evaluator identity
  - Reject partial or semantically mismatched caches without substitution
  - Provide an explicit exact-target rebuild operation
  - Propagate every material generation failure to callers

Design principles:
  - Saved split membership is always required
  - Provenance is published last as the cache completion marker
  - Rebuild removes only the requested run analysis target
  - CLI parsing remains outside this reusable service
===============================================================================
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from src import common, datasets, domain, experiments, learning

from . import analysis_artifacts as artifacts

DEFAULT_BATCH_SIZE = 1
_ARTIFACT_LOCKS_DIRNAME = ".locks"

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
    """Raised when existing artifacts cannot be proven identical and complete."""


@dataclass(frozen=True)
class RunArtifactPlan:
    """Artifact generation plan derived from a current saved run."""

    run_dir: Path
    id_dataset_name: str
    ood_dataset_name: str


@dataclass(frozen=True)
class ArtifactRequest:
    """Expected provenance and ordered source membership for one cache."""

    provenance: dict[str, Any]
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class _EvaluatorArtifactContract:
    """Strict task-declared field and payload contract for cached artifacts."""

    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    output_units: tuple[str, ...]
    physics_kind: str

    @property
    def is_steady_brinkman(self) -> bool:
        """Return whether the concrete steady-flow diagnostic schema applies."""
        return self.physics_kind == domain.physics.brinkman.STEADY_BRINKMAN_KIND


# ======================================================================
# Utilities
# ======================================================================


def cleanup_gpu() -> None:
    """
    Aggressively clean GPU memory after inference.

    Performs garbage collection, clears CUDA cache, and collects IPC handles
    to free GPU memory for the next run. Safe to call when CUDA is unavailable.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _contains_current_run_marker(path: Path) -> bool:
    """Return whether one directory contains any current run-contract marker."""
    return path.is_dir() and any((path / filename).exists() for filename in common.paths.CURRENT_RUN_REQUIRED_FILES)


def iter_run_dirs(root: Path, *, run_names: Iterable[str] | None = None) -> Iterable[Path]:
    """
    Iterate over current-contract run directories in sorted order.

    A directory is considered a valid current run when it contains the complete
    config, split, normalizer, best-checkpoint, last-checkpoint, and summary contract.

    Parameters
    ----------
    root : Path
        Directory containing run subdirectories, or a single current run
        directory when no run names are provided.
    run_names : Iterable[str] | None, optional
        Optional run names to select under root. Requested runs must satisfy
        the current run contract.

    Yields
    ------
    pathlib.Path
        Valid current run directories.

    """
    root = Path(root)
    selected_run_names = list(run_names or [])

    if selected_run_names:
        for selected_run_name in selected_run_names:
            run_name = common.paths.validate_logical_name(selected_run_name, label="run_name")
            run_dir = root / run_name
            experiments.run.validate_completed_run(run_dir)
            yield run_dir
        return

    if common.paths.is_current_run_dir(root):
        experiments.run.validate_completed_run(root)
        yield root
        return

    if not root.is_dir():
        msg = f"Run discovery root not found: {root}"
        raise FileNotFoundError(msg)

    if _contains_current_run_marker(root):
        experiments.run.validate_completed_run(root)
        msg = f"Run validation returned without identifying a current run: {root}"
        raise RuntimeError(msg)

    for candidate in sorted(root.iterdir()):
        if not _contains_current_run_marker(candidate):
            continue
        experiments.run.validate_completed_run(candidate)
        if not common.paths.is_current_run_dir(candidate):
            msg = f"Run validation returned without identifying a current run: {candidate}"
            raise RuntimeError(msg)
        yield candidate


def _dataset_name_from_identity(metadata: Mapping[str, Any], *, identity_key: str) -> str:
    """Return one logical dataset id from strict saved split identity."""
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
    """Load the saved split contract used by inference and provenance."""
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
    Build the artifact plan for a current saved run.

    The dataset names used for artifact paths come from split_indices.pt
    metadata, then are checked against config.yaml data settings. This keeps
    output names aligned with the same saved split identity consumed by
    learning.inference.context.load_inference_context().
    """
    run_dir = Path(run_dir)
    experiments.run.validate_completed_run(run_dir)
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
    """Return exact lexical analysis/target paths after rejecting symlink aliases."""
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
    """Return one lock path outside an exact deletable artifact target."""
    analysis_root, target = _validated_artifact_target(run_dir=run_dir, save_root=save_root)
    relative = target.relative_to(analysis_root)
    return analysis_root / _ARTIFACT_LOCKS_DIRNAME / relative.parent / f"{relative.name}.lock"


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
    return artifacts.ordered_indices_sha256(indices)


def _build_artifact_request(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    max_cases: int | None,
    dataset_root: Path,
) -> ArtifactRequest:
    """Build semantic cache provenance without path, size, or mtime identity."""
    run_dir = Path(run_dir)
    if max_cases is not None and max_cases <= 0:
        msg = f"max_cases must be positive when provided, got {max_cases}."
        raise ValueError(msg)
    completed = experiments.run.validate_completed_run(run_dir)
    split_contract = completed["split_indices"]
    run_config = completed["config"]
    summary = completed["summary"]
    metadata = _split_metadata(split_contract, run_dir=run_dir)
    data_cfg = run_config.get("data")
    run_cfg = run_config.get("run")
    evaluation_cfg = run_config.get("evaluation")
    if not isinstance(data_cfg, Mapping) or not isinstance(run_cfg, Mapping) or not isinstance(evaluation_cfg, Mapping):
        msg = f"config.yaml must contain data, run, and evaluation mappings: {common.paths.resolve_run_config_path(run_dir)}"
        raise TypeError(msg)
    task = experiments.config.loader.validate_resolved_task_contract(run_config)
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

    effective_count = len(full_source_indices) if max_cases is None else min(len(full_source_indices), max_cases)
    effective_source_indices = full_source_indices[:effective_count]
    membership_digests = metadata.get("membership_digests")
    if not isinstance(membership_digests, Mapping):
        msg = "split_indices.pt metadata.membership_digests must be a mapping."
        raise TypeError(msg)
    metrics = evaluation_cfg.get("metrics")
    if not isinstance(metrics, list):
        msg = "config.yaml evaluation.metrics must be a list."
        raise TypeError(msg)

    provenance: dict[str, Any] = {
        "provenance_schema_version": artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
        "run": {
            "name": run_cfg.get("name"),
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "effective_config_digest": summary["effective_config_digest"],
            "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
        },
        "split_role": split,
        "dataset": {
            "name": dataset_name,
            "full_case_count": dataset_full_count,
            "fingerprint": expected_identity.fingerprint,
            "task_contract_digest": expected_identity.task_contract_digest,
            "saved_membership_digest": membership_digests[split],
        },
        "selection": {
            "index_key": _SPLIT_INDEX_KEYS[split],
            "full_selected_case_count": len(full_source_indices),
            "effective_case_count": effective_count,
            "generation_limit": max_cases,
            "full_ordered_source_indices_sha256": _indices_sha256(full_source_indices),
            "effective_ordered_source_indices_sha256": _indices_sha256(effective_source_indices),
        },
        "evaluator": {
            "metrics": metrics,
            "objective": evaluation_cfg.get("objective"),
            "input_fields": list(task.input_names),
            "output_fields": list(task.output_names),
            "output_units": {field.name: field.unit for field in task.outputs},
            "physics_kind": task.physics.kind,
        },
    }
    return ArtifactRequest(provenance=provenance, source_indices=effective_source_indices)


def _read_artifact_provenance(path: Path) -> Mapping[str, Any]:
    """Read a provenance JSON object or fail as an incompatible cache."""
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


def _cache_has_outputs(*, save_root: Path, parquet_path: Path, npz_dir: Path) -> bool:
    """Return whether any complete or interrupted cache output is present."""
    return any(
        (
            parquet_path.exists(),
            artifacts.artifact_provenance_path(save_root).exists(),
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
    """Return JSON object metadata after excluding reserved identity fields."""
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
    """Resolve strict cached field names, units, and physics kind from provenance."""
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, Mapping):
        msg = "Artifact provenance must contain an evaluator mapping."
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
    physics_kind = evaluator.get("physics_kind")
    if not isinstance(physics_kind, str) or not physics_kind:
        msg = "Artifact provenance evaluator.physics_kind must be a non-empty string."
        raise ArtifactCacheError(msg)
    return _EvaluatorArtifactContract(
        input_fields=input_fields,
        output_fields=output_fields,
        output_units=tuple(raw_units[name] for name in output_fields),
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
    """Validate identity, task fields, tensors, and concrete diagnostics in one NPZ."""
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
        "Rx",
        "Ry",
        "Rc",
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
    expected_artifact_units = (*contract.output_units, "m/s") if contract.is_steady_brinkman else contract.output_units
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
        for name in ("Rx", "Ry", "Rc", "div_u", "div_eps_u"):
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
    """Return exact Parquet columns and finite metric columns for a task contract."""
    common = {
        "case_index",
        "source_index",
        "split_local_index",
        "npz_path",
        "meta",
        "inference_time_ms",
    }
    if contract.is_steady_brinkman:
        metrics: tuple[str, ...] = (
            "rel_l2",
            "rel_h1",
            "rmse_p",
            "rmse_u",
            "rmse_v",
            "rmse_U",
            "mom_mse",
            "cont_mse",
            "bc_mse",
        )
        return common | set(metrics) | {"kappa_names"}, metrics
    metrics = tuple(f"rmse_{field}" for field in contract.output_fields)
    return common | set(metrics), metrics


def _require_parquet_string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    """Normalize a Parquet nested string sequence without accepting scalar text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
        msg = f"{label} must contain a sequence of non-empty strings."
        raise ArtifactCacheError(msg)
    return tuple(value)


def _load_validated_artifact_cache(
    *,
    save_root: Path,
    parquet_path: Path,
    npz_dir: Path,
    request: ArtifactRequest,
) -> pd.DataFrame:
    """Load a cache only after exact provenance, membership and payload checks."""
    provenance_path = artifacts.artifact_provenance_path(save_root)
    if not provenance_path.is_file():
        msg = f"Existing artifact cache has no provenance sidecar: {provenance_path}. Refusing to trust or overwrite invalid/partial artifacts."
        raise ArtifactCacheError(msg)
    if not parquet_path.is_file():
        msg = f"Artifact cache is incomplete (Parquet missing): {parquet_path}"
        raise ArtifactCacheError(msg)

    stored_provenance = dict(_read_artifact_provenance(provenance_path))
    stored_outputs = stored_provenance.pop("outputs", None)
    if stored_provenance != request.provenance:
        expected_json = json.dumps(request.provenance, indent=2, sort_keys=True)
        actual_json = json.dumps(stored_provenance, indent=2, sort_keys=True)
        msg = (
            f"Artifact provenance is incompatible: {provenance_path}\n"
            f"Expected:\n{expected_json}\nActual:\n{actual_json}\n"
            "Refusing to trust or overwrite the existing cache."
        )
        raise ArtifactCacheError(msg)

    try:
        computed_outputs = artifacts.artifact_output_manifest(save_root)
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

    expected_source_indices = request.source_indices
    expected_split_local_indices = tuple(range(len(expected_source_indices)))
    expected_case_indices = tuple(source_index + 1 for source_index in expected_source_indices)
    if len(df) != len(expected_source_indices):
        msg = f"Cached Parquet has {len(df)} rows; expected {len(expected_source_indices)}."
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
        zip(expected_source_indices, expected_split_local_indices, expected_case_indices, expected_npz_paths, strict=True)
    ):
        row = df.iloc[row_position]
        raw_npz_path = row.loc["npz_path"]
        if not isinstance(raw_npz_path, str) or _normalise_path(Path(raw_npz_path)) != _normalise_path(expected_npz_path):
            msg = f"Cached Parquet npz_path mismatch at row {row_position}: {raw_npz_path!r}."
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

    return df


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
    """Remove one exact artifact target under its persistent exclusive lock."""
    lock_path = _artifact_lock_path(run_dir=run_dir, save_root=save_root)
    with (
        experiments.run.run_writer_lease(run_dir, blocking=True),
        common.locking.exclusive_file_lock(lock_path, blocking=True),
    ):
        _rebuild_artifact_target_locked(run_dir=run_dir, save_root=save_root)


def _run_or_load_artifacts_locked(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    max_cases: int | None,
    batch_size: int,
    prefer_cuda: bool,
    dataset_root: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load or generate artifacts for one run and saved split.

    Reuses existing Parquet+NPZ artifacts only after exact provenance and
    payload validation. If artifacts do not exist, runs split-aware inference via
    learning.inference.context.load_inference_context() and generates
    Parquet+NPZ artifacts via artifacts.generate_artifacts().

    Parameters
    ----------
    run_dir : Path
        Current run directory satisfying the saved-run contract.
    dataset_name : str
        Logical dataset name used for artifact file naming.
    split : {"eval", "ood"}
        Saved split role to load. ID evaluation uses "eval"; OOD uses "ood".
    max_cases : int or None
        Maximum cases to process. If None, processes the entire saved split.
    batch_size : int
        Inference DataLoader batch size.
    prefer_cuda : bool
        Prefer CUDA when available.
    dataset_root : Path
        Current independent dataset root.
    rebuild : bool, optional
        Explicitly remove only this computed artifact target before generation.

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
    if batch_size != 1:
        msg = "Artifact generation currently requires batch_size=1 to preserve one output row and NPZ file per case."
        raise ValueError(msg)
    if max_cases is not None and max_cases <= 0:
        msg = f"max_cases must be positive when provided, got {max_cases}."
        raise ValueError(msg)
    dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")

    save_root = _artifact_save_root(run_dir=run_dir, dataset_name=dataset_name, split=split)
    npz_dir = save_root / "npz"
    parquet_path = save_root / f"{dataset_name}.parquet"

    request = _build_artifact_request(
        run_dir=run_dir,
        dataset_name=dataset_name,
        split=split,
        max_cases=max_cases,
        dataset_root=dataset_root,
    )
    if rebuild:
        rebuild_artifact_target(run_dir=run_dir, save_root=save_root)
    task = experiments.config.loader.validate_resolved_task_contract(_load_run_config(run_dir))

    print(f"[RUN] {run_dir.name} | split={split} | dataset={dataset_name}")
    print(f"      run_dir={run_dir}")
    print(f"      save_root={save_root}")

    if _cache_has_outputs(save_root=save_root, parquet_path=parquet_path, npz_dir=npz_dir):
        print(f"[VALIDATE] {run_dir.name} | {split} | {dataset_name} (existing cache)")
        df = _load_validated_artifact_cache(
            save_root=save_root,
            parquet_path=parquet_path,
            npz_dir=npz_dir,
            request=request,
        )
        print(f"[LOAD] {run_dir.name} | {split} | {dataset_name} (validated cache)")
        return df

    model, loader, processor, device = learning.inference.context.load_inference_context(
        run_dir=run_dir,
        dataset_root=dataset_root,
        split=split,
        batch_size=batch_size,
        prefer_cuda=prefer_cuda,
    )

    try:
        artifacts.generate_artifacts(
            task=task,
            model=model,
            loader=loader,
            processor=processor,
            device=device,
            save_root=save_root,
            dataset_name=dataset_name,
            provenance=request.provenance,
            max_cases=max_cases,
        )
    finally:
        del model, loader, processor
        cleanup_gpu()

    return _load_validated_artifact_cache(
        save_root=save_root,
        parquet_path=parquet_path,
        npz_dir=npz_dir,
        request=request,
    )


def run_or_load_artifacts(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    max_cases: int | None,
    batch_size: int,
    prefer_cuda: bool,
    dataset_root: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load, rebuild, or generate one artifact cache under its target lock."""
    logical_dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    save_root = _artifact_save_root(
        run_dir=run_dir,
        dataset_name=logical_dataset_name,
        split=split,
    )
    completion_path = artifacts.artifact_provenance_path(save_root)
    observed_completion = _completion_marker_identity(completion_path) if rebuild else None
    lock_path = _artifact_lock_path(run_dir=run_dir, save_root=save_root)
    with (
        experiments.run.run_writer_lease(run_dir, blocking=True),
        common.locking.exclusive_file_lock(lock_path, blocking=True),
    ):
        current_completion = _completion_marker_identity(completion_path)
        effective_rebuild = rebuild and (current_completion is None or current_completion == observed_completion)
        return _run_or_load_artifacts_locked(
            run_dir=run_dir,
            dataset_name=logical_dataset_name,
            split=split,
            max_cases=max_cases,
            batch_size=batch_size,
            prefer_cuda=prefer_cuda,
            dataset_root=dataset_root,
            rebuild=effective_rebuild,
        )


def build_artifacts(
    *,
    runs_root: Path,
    dataset_root: Path,
    run_names: Iterable[str] | None = None,
    max_cases: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    prefer_cuda: bool = True,
    rebuild: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build or validate ID and OOD artifacts for every selected run.

    Parameters
    ----------
    runs_root : Path
        One completed run or a container of run directories.
    dataset_root : Path
        Independent root containing immutable task datasets.
    run_names : Iterable[str] | None, optional
        Explicit run names under ``runs_root``.
    max_cases : int | None, optional
        Positive effective saved-split case limit.
    batch_size : int, optional
        Required per-case batch size, currently one.
    prefer_cuda : bool, optional
        Prefer CUDA for inference when available.
    rebuild : bool, optional
        Remove only each exact selected artifact target before regeneration.

    Returns
    -------
    dict[str, dict[str, pandas.DataFrame]]
        Validated evaluation and OOD frames keyed by run name.

    """
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for run_dir in iter_run_dirs(runs_root, run_names=run_names):
        plan = load_run_artifact_plan(run_dir)
        id_frame = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.id_dataset_name,
            split="eval",
            max_cases=max_cases,
            batch_size=batch_size,
            prefer_cuda=prefer_cuda,
            dataset_root=dataset_root,
            rebuild=rebuild,
        )
        ood_frame = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.ood_dataset_name,
            split="ood",
            max_cases=max_cases,
            batch_size=batch_size,
            prefer_cuda=prefer_cuda,
            dataset_root=dataset_root,
            rebuild=rebuild,
        )
        results[run_dir.name] = {"eval": id_frame, "ood": ood_frame}
        cleanup_gpu()
    return results
