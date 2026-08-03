"""
===============================================================================
build_training_dataset.py
===============================================================================
Build and atomically publish one canonical final training dataset.

Responsibilities:
  - Construct bounded preallocated tensors from an admitted generated batch
  - Derive final dataset identity and self-contained metadata snapshots
  - Coordinate locking, transaction recovery, validation, and atomic publication

Design principles:
  - Generated-source admission delegates to ``generated_batch``
  - Dataset and metadata publication succeed or recover as one transaction
  - Immutable identities bind tensors, TaskSpec semantics, and source snapshots

This module does NOT:
  - Reimplement generated manifest, case, grid, unit, or numerical admission
  - Run COMSOL, MATLAB, model training, evaluation, or artifact generation
  - Create splits, normalizers, dataloaders, runs, or checkpoints
===============================================================================
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from src import common, datasets, domain
from tqdm import tqdm

from . import generated_batch as generated

if TYPE_CHECKING:
    from src.datasets.dataset_identity import DatasetIdentity
    from src.datasets.dataset_metadata import DatasetMetadata
    from src.domain.tasks.domain_task_spec import TaskSpec

_PUBLICATION_TRANSACTION_SCHEMA_KIND = "training_dataset_publication_transaction"
_PUBLICATION_TRANSACTION_SCHEMA_VERSION = 1
_PUBLICATION_TRANSACTION_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "dataset_id",
        "phase",
        "staging_root",
        "dataset_sha256",
        "dataset_size",
        "dataset_metadata_sha256",
    }
)


def _snapshot_metadata_entry(path: Path, *, required: bool, role: str) -> dict[str, Any]:
    """Describe one independently hashable metadata snapshot."""
    return {
        "sha256": generated.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "required": required,
        "role": role,
    }


def _stage_metadata_package(
    destination: Path,
    *,
    dataset_identity: datasets.identity.DatasetIdentity,
    dataset_sha256: str,
    dataset_size: int,
    task: TaskSpec,
    manifest_snapshot: bytes,
    manifest_sha256: str,
    manifest_identity_sha256: str,
    sample_csv_snapshot: bytes,
    sample_json_snapshot: bytes,
    sample_csv_sha256: str,
    sample_json_sha256: str,
    timing_snapshot: bytes | None,
    timing_summary: dict[str, Any],
) -> None:
    """Stage one coherent set of validated model-training metadata."""
    destination.mkdir(parents=True)
    snapshots = {
        datasets.metadata.SOURCE_MANIFEST_FILENAME: (manifest_snapshot, True, "validated_generation_manifest"),
        datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME: (sample_csv_snapshot, True, "validated_parameter_sample_csv"),
        datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME: (sample_json_snapshot, True, "validated_parameter_sample_json"),
    }
    if timing_snapshot is not None:
        snapshots[datasets.metadata.COMSOL_TIMING_FILENAME] = (timing_snapshot, False, "validated_operational_comsol_timing")
    for filename, (snapshot, _required, _role) in snapshots.items():
        common.serialization.atomic_write_bytes(destination / filename, snapshot)
    if generated.sha256_file(destination / datasets.metadata.SOURCE_MANIFEST_FILENAME) != manifest_sha256:
        msg = "Staged generation manifest does not match its admitted snapshot."
        raise RuntimeError(msg)
    if generated.sha256_file(destination / datasets.metadata.SOURCE_SAMPLE_CSV_FILENAME) != sample_csv_sha256:
        msg = "Staged parameter-sample CSV does not match its admitted snapshot."
        raise RuntimeError(msg)
    if generated.sha256_file(destination / datasets.metadata.SOURCE_SAMPLE_JSON_FILENAME) != sample_json_sha256:
        msg = "Staged parameter-sample JSON does not match its admitted snapshot."
        raise RuntimeError(msg)
    snapshot_artifacts = {
        filename: _snapshot_metadata_entry(destination / filename, required=required, role=role)
        for filename, (_snapshot, required, role) in snapshots.items()
    }
    scientific_identity = {
        "dataset_schema_version": datasets.identity.TRAINING_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_identity.fingerprint,
        "task_id": task.id,
        "task_contract_digest": task.contract_digest,
        "source_batch_id": dataset_identity.dataset_id,
        "generated_batch_identity_sha256": manifest_identity_sha256,
        "sample_count": dataset_identity.sample_count,
        "spatial_shape": list(dataset_identity.spatial_shape),
        "tensors": {
            "inputs": {
                "dtype": "float32",
                "shape": [dataset_identity.sample_count, task.in_channels, *dataset_identity.spatial_shape],
            },
            "outputs": {
                "dtype": "float32",
                "shape": [dataset_identity.sample_count, task.out_channels, *dataset_identity.spatial_shape],
            },
        },
    }
    metadata = {
        "schema_kind": datasets.metadata.METADATA_SCHEMA_KIND,
        "schema_version": datasets.metadata.METADATA_SCHEMA_VERSION,
        "dataset_id": dataset_identity.dataset_id,
        "scientific_identity": scientific_identity,
        "artifacts": {
            "dataset": {
                "filename": f"{dataset_identity.dataset_id}.pt",
                "sha256": dataset_sha256,
                "size_bytes": dataset_size,
            },
            "snapshots": snapshot_artifacts,
        },
        "operational_provenance": {
            "builder_module": datasets.metadata.BUILDER_MODULE,
            "publication_method": datasets.metadata.PUBLICATION_METHOD,
            "source_manifest_sha256": manifest_sha256,
            "timing": timing_summary,
        },
    }
    common.serialization.atomic_write_json(destination / datasets.metadata.METADATA_FILENAME, metadata)


def _publication_transaction_record(
    *,
    dataset_id: str,
    phase: str,
    staging_root: Path,
    dataset_sha256: str = "",
    dataset_size: int = 0,
    dataset_metadata_sha256: str = "",
) -> dict[str, Any]:
    """Build one exact operational transaction marker."""
    if phase not in {"building", "ready"}:
        msg = f"Unsupported dataset publication phase: {phase!r}."
        raise ValueError(msg)
    return {
        "schema_kind": _PUBLICATION_TRANSACTION_SCHEMA_KIND,
        "schema_version": _PUBLICATION_TRANSACTION_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "phase": phase,
        "staging_root": str(staging_root.resolve(strict=False)),
        "dataset_sha256": dataset_sha256,
        "dataset_size": dataset_size,
        "dataset_metadata_sha256": dataset_metadata_sha256,
    }


def _load_publication_transaction(
    transaction_path: Path,
    *,
    training_root: Path,
    dataset_id: str,
) -> tuple[dict[str, Any], Path]:
    """Load and constrain a recovery marker to this builder's staging area."""
    try:
        loaded = json.loads(transaction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Dataset publication transaction is unreadable: {transaction_path}"
        raise RuntimeError(msg) from error
    record = generated.require_exact_mapping_keys(loaded, _PUBLICATION_TRANSACTION_KEYS, label="Dataset publication transaction")
    if (
        not isinstance(record["schema_kind"], str)
        or record["schema_kind"] != _PUBLICATION_TRANSACTION_SCHEMA_KIND
        or isinstance(record["schema_version"], bool)
        or not isinstance(record["schema_version"], int)
        or record["schema_version"] != _PUBLICATION_TRANSACTION_SCHEMA_VERSION
        or not isinstance(record["dataset_id"], str)
        or record["dataset_id"] != dataset_id
        or not isinstance(record["phase"], str)
        or record["phase"] not in {"building", "ready"}
        or not isinstance(record["staging_root"], str)
        or not record["staging_root"]
        or isinstance(record["dataset_size"], bool)
        or not isinstance(record["dataset_size"], int)
        or record["dataset_size"] < 0
        or not isinstance(record["dataset_sha256"], str)
        or not isinstance(record["dataset_metadata_sha256"], str)
    ):
        msg = f"Dataset publication transaction has invalid identity or scalar fields: {transaction_path}"
        raise RuntimeError(msg)
    staging_root = Path(record["staging_root"])
    expected_parent = training_root.resolve(strict=False)
    if (
        not staging_root.is_absolute()
        or staging_root.parent.resolve(strict=False) != expected_parent
        or not staging_root.name.startswith(f".{dataset_id}.dataset-build.")
        or not staging_root.name.endswith(".tmp")
        or staging_root.is_symlink()
    ):
        msg = f"Dataset publication transaction names an unsafe staging root: {staging_root}"
        raise RuntimeError(msg)
    if record["phase"] == "building":
        if record["dataset_sha256"] or record["dataset_size"] or record["dataset_metadata_sha256"]:
            msg = "Building publication transaction cannot claim completed staged content."
            raise RuntimeError(msg)
    else:
        generated.require_sha256(record["dataset_sha256"], label="Dataset publication transaction dataset_sha256")
        generated.require_sha256(
            record["dataset_metadata_sha256"],
            label="Dataset publication transaction dataset_metadata_sha256",
        )
        if record["dataset_size"] <= 0:
            msg = "Ready publication transaction dataset_size must be positive."
            raise RuntimeError(msg)
    return record, staging_root


def _single_publication_component(
    staged: Path,
    final: Path,
    *,
    label: str,
) -> tuple[Path, bool]:
    """Resolve exactly one staged-or-final transaction component."""
    for candidate in (staged, final):
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            msg = f"Dataset publication {label} target has an invalid filesystem type: {candidate}"
            raise RuntimeError(msg)
    present = [candidate for candidate in (staged, final) if candidate.is_dir()]
    if len(present) != 1:
        msg = f"Ready dataset publication must have exactly one staged-or-final {label} directory."
        raise RuntimeError(msg)
    return present[0], present[0] == final


def _recover_interrupted_publication(
    transaction_path: Path,
    *,
    training_root: Path,
    destination_dir: Path,
    metadata_destination: Path,
    raw_dir: Path,
    dataset_id: str,
    task: TaskSpec,
) -> tuple[DatasetIdentity, DatasetMetadata] | None:
    """Discard an incomplete build or finish an exact validated ready publication."""
    if transaction_path.is_symlink():
        msg = f"Dataset publication marker cannot be a symlink: {transaction_path}"
        raise RuntimeError(msg)
    if not transaction_path.is_file():
        if transaction_path.exists():
            msg = f"Dataset publication marker is not a regular file: {transaction_path}"
            raise RuntimeError(msg)
        return None
    record, staging_root = _load_publication_transaction(
        transaction_path,
        training_root=training_root,
        dataset_id=dataset_id,
    )
    if record["phase"] == "building":
        if destination_dir.exists() or destination_dir.is_symlink() or metadata_destination.exists() or metadata_destination.is_symlink():
            msg = "Incomplete building transaction unexpectedly has an authoritative target."
            raise RuntimeError(msg)
        if staging_root.exists():
            shutil.rmtree(staging_root)
        transaction_path.unlink()
        return None

    staged_dataset_dir = staging_root / "raw" / dataset_id
    staged_metadata_dir = staging_root / "meta" / dataset_id
    dataset_dir, dataset_is_final = _single_publication_component(
        staged_dataset_dir,
        destination_dir,
        label="dataset",
    )
    metadata_dir, metadata_is_final = _single_publication_component(
        staged_metadata_dir,
        metadata_destination,
        label="metadata",
    )
    dataset_path = dataset_dir / f"{dataset_id}.pt"
    if not dataset_path.is_file() or dataset_path.is_symlink() or set(dataset_dir.iterdir()) != {dataset_path}:
        msg = f"Recovered dataset directory does not contain exactly one regular payload: {dataset_dir}"
        raise RuntimeError(msg)
    if dataset_path.stat().st_size != record["dataset_size"] or generated.sha256_file(dataset_path) != record["dataset_sha256"]:
        msg = "Recovered staged/final dataset does not match its ready transaction digest and size."
        raise RuntimeError(msg)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    try:
        dataset_identity = datasets.identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    finally:
        del payload
        gc.collect()
    if dataset_identity.dataset_id != dataset_id:
        msg = "Recovered dataset identity does not match its publication transaction."
        raise RuntimeError(msg)
    metadata_path = metadata_dir / datasets.metadata.METADATA_FILENAME
    if not metadata_path.is_file() or generated.sha256_file(metadata_path) != record["dataset_metadata_sha256"]:
        msg = "Recovered dataset metadata does not match its ready transaction digest."
        raise RuntimeError(msg)
    datasets.metadata.validate_dataset_metadata_directory(
        metadata_dir,
        dataset_identity=dataset_identity,
        dataset_path=dataset_path,
    )
    if not (metadata_is_final and dataset_is_final):
        source_manifest_snapshot = (metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME).read_bytes()
        generated.assert_generation_snapshot_current(
            raw_dir,
            raw_dir / "batch_manifest.json",
            source_manifest_snapshot,
        )

    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    moved_metadata = False
    try:
        if not metadata_is_final:
            metadata_dir.replace(metadata_destination)
            moved_metadata = True
        if not dataset_is_final:
            dataset_dir.replace(destination_dir)
    except BaseException:
        if moved_metadata and metadata_destination.is_dir() and not staged_metadata_dir.exists():
            metadata_destination.replace(staged_metadata_dir)
        raise

    final_dataset_path = destination_dir / f"{dataset_id}.pt"
    if final_dataset_path.stat().st_size != record["dataset_size"] or generated.sha256_file(final_dataset_path) != record["dataset_sha256"]:
        msg = "Recovered final dataset changed during publication."
        raise RuntimeError(msg)
    package = datasets.metadata.validate_dataset_metadata_directory(
        metadata_destination,
        dataset_identity=dataset_identity,
        dataset_path=final_dataset_path,
    )
    transaction_path.unlink()
    if staging_root.exists():
        shutil.rmtree(staging_root)
    return dataset_identity, package


def build_batch_dataset(  # noqa: C901, PLR0912, PLR0915
    batch_name: str,
    verbose: bool = False,
    *,
    dataset_id: str | None = None,
    task_id: str = "steady_flow",
    generated_data_root: Path | str | None = None,
    model_training_data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one final dataset plus metadata package."""
    task = domain.tasks.registry.get_task(task_id)
    if task.id != "steady_flow":
        msg = f"The COMSOL batch builder supports only the current steady_flow task, got {task.id!r}."
        raise ValueError(msg)
    batch_name = common.paths.validate_logical_name(batch_name, label="batch_name")
    resolved_dataset_id = common.paths.validate_logical_name(dataset_id or batch_name, label="dataset_id")
    if resolved_dataset_id != batch_name:
        msg = "Current one-batch datasets must use the source batch name as dataset_id."
        raise ValueError(msg)
    generated_root = Path(generated_data_root).expanduser() if generated_data_root is not None else common.paths.get_generated_data_root()
    training_root = (
        Path(model_training_data_root).expanduser() if model_training_data_root is not None else common.paths.get_model_training_data_root()
    )
    meta_dir = generated_root / "meta"
    raw_dir = generated_root / "raw" / batch_name
    processed_dir = generated_root / "processed" / batch_name
    manifest_path = raw_dir / "batch_manifest.json"
    training_meta_root = training_root / "meta"
    training_raw_root = training_root / "raw"
    destination_dir = training_raw_root / resolved_dataset_id
    destination = destination_dir / f"{resolved_dataset_id}.pt"
    metadata_destination = training_meta_root / resolved_dataset_id
    lock_path = common.paths.resolve_dataset_build_lock_path(
        resolved_dataset_id,
        model_training_data_root=training_root,
    )
    transaction_path = common.paths.resolve_dataset_build_transaction_path(
        resolved_dataset_id,
        model_training_data_root=training_root,
    )
    training_root.mkdir(parents=True, exist_ok=True)

    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        generated.assert_generation_batch_idle(raw_dir)
        recovered = _recover_interrupted_publication(
            transaction_path,
            training_root=training_root,
            destination_dir=destination_dir,
            metadata_destination=metadata_destination,
            raw_dir=raw_dir,
            dataset_id=resolved_dataset_id,
            task=task,
        )
        if recovered is not None:
            recovered_identity, recovered_metadata = recovered
            result = {
                "source_batch": batch_name,
                "generated_data_root": generated_root,
                "dataset_path": destination,
                "metadata_path": metadata_destination,
                "case_count": recovered_identity.sample_count,
                "task": task.id,
                "task_contract_digest": task.contract_digest,
                "timing_coverage": recovered_metadata.timing_summary,
                "dataset_fingerprint": recovered_identity.fingerprint,
                "status": "complete",
            }
            if verbose:
                for key, value in result.items():
                    print(f"{key}: {value}")
            return result
        if destination_dir.exists() or destination_dir.is_symlink():
            msg = f"Refusing to overwrite existing final dataset: {destination_dir}"
            raise FileExistsError(msg)
        if metadata_destination.exists() or metadata_destination.is_symlink():
            msg = f"Refusing to overwrite existing dataset metadata: {metadata_destination}"
            raise FileExistsError(msg)
        manifest = generated.load_batch_manifest(raw_dir, processed_dir, batch_name=batch_name)
        try:
            manifest_snapshot = manifest_path.read_bytes()
            snapshot_manifest = json.loads(manifest_snapshot.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            msg = f"Could not capture validated generation manifest: {manifest_path}"
            raise RuntimeError(msg) from error
        if isinstance(snapshot_manifest, dict) and isinstance(snapshot_manifest.get("cases"), dict):
            snapshot_manifest["cases"] = [snapshot_manifest["cases"]]
        if snapshot_manifest != manifest:
            msg = "Generation manifest changed while source admission was in progress."
            raise RuntimeError(msg)
        manifest_sha256 = hashlib.sha256(manifest_snapshot).hexdigest()
        generated.validate_exact_source_membership(raw_dir, processed_dir, manifest)
        (
            _sample_csv_path,
            _sample_json_path,
            sample_frame,
            _sample_json,
            portable_sampling,
            sample_csv_snapshot,
            sample_json_snapshot,
        ) = generated.load_generation_metadata(
            meta_dir,
            batch_name,
            manifest,
        )
        case_ids = list(manifest["intended_case_ids"])
        if len(case_ids) != manifest["configuration"]["N"]:
            msg = "A complete batch manifest must contain exactly configuration.N intended cases."
            raise ValueError(msg)
        timing_snapshot, _timing_payload, timing_summary = generated.load_timing_snapshot(
            processed_dir,
            batch_name=batch_name,
            manifest_sha256=manifest_sha256,
            intended_case_ids=case_ids,
        )
        generated_identity = datasets.identity.build_generated_batch_identity(
            manifest,
            sampling=portable_sampling,
        )
        manifest_identity_sha256 = str(generated_identity["batch_manifest_identity_sha256"])
        sample_json_sha256 = hashlib.sha256(sample_json_snapshot).hexdigest()
        provenance = generated.source_provenance(
            manifest,
            manifest_sha256=manifest_sha256,
            sample_json_sha256=sample_json_sha256,
        )
        records_by_id = {record["case_id"]: record for record in manifest["cases"]}

        staging_root = Path(tempfile.mkdtemp(dir=training_root, prefix=f".{resolved_dataset_id}.dataset-build.", suffix=".tmp"))
        stage_dataset_dir = staging_root / "raw" / resolved_dataset_id
        stage_metadata_dir = staging_root / "meta" / resolved_dataset_id
        staged_dataset_path = stage_dataset_dir / f"{resolved_dataset_id}.pt"
        metadata_published = False
        dataset_published = False
        publication_complete = False
        transaction_active = False
        transaction_phase: str | None = None
        inputs: torch.Tensor | None = None
        outputs: torch.Tensor | None = None
        source_identities: list[dict[str, Any]] = []
        source_metadata: list[dict[str, Any]] = []
        fingerprints: list[str] = []
        reference_shape: tuple[int, int] | None = None
        try:
            common.serialization.atomic_write_json(
                transaction_path,
                _publication_transaction_record(
                    dataset_id=resolved_dataset_id,
                    phase="building",
                    staging_root=staging_root,
                ),
            )
            transaction_active = True
            transaction_phase = "building"
            stage_dataset_dir.mkdir(parents=True)
            for index, case_id in enumerate(tqdm(case_ids, desc=f"Building {batch_name}", unit="case", disable=not verbose)):
                spatial_shape, case_inputs, case_outputs, normalized_metadata, stable_source, fingerprint = generated.interpret_generated_case(
                    case_id,
                    task=task,
                    manifest=manifest,
                    manifest_record=records_by_id[case_id],
                    sample_row=sample_frame.loc[case_id],
                    raw_dir=raw_dir,
                    processed_dir=processed_dir,
                )
                if reference_shape is None:
                    reference_shape = spatial_shape
                    inputs = torch.empty((len(case_ids), task.in_channels, *spatial_shape), dtype=torch.float32)
                    outputs = torch.empty((len(case_ids), task.out_channels, *spatial_shape), dtype=torch.float32)
                elif spatial_shape != reference_shape:
                    msg = f"Inconsistent case shape for {case_id!r}: {spatial_shape} != {reference_shape}."
                    raise ValueError(msg)
                if inputs is None or outputs is None:
                    msg = "Final tensors were not allocated."
                    raise RuntimeError(msg)
                inputs[index].copy_(case_inputs)
                outputs[index].copy_(case_outputs)
                source_identities.append(stable_source)
                source_metadata.append(normalized_metadata)
                fingerprints.append(fingerprint)
                del case_inputs, case_outputs
                if verbose and index == 0:
                    print(f"Input fields: {list(task.input_names)}")
                    print(f"Output fields: {list(task.output_names)}")
                    print(f"Spatial shape: {spatial_shape}")
                    print(f"First case fingerprint: {fingerprint}")
            if inputs is None or outputs is None:
                msg = f"No complete generated cases found for {batch_name!r}."
                raise RuntimeError(msg)
            payload = datasets.identity.build_training_dataset_payload(
                task=task,
                dataset_id=resolved_dataset_id,
                sample_ids=case_ids,
                generated_batch_identity=generated_identity,
                source_identities=source_identities,
                source_metadata=source_metadata,
                source_provenance=provenance,
                case_fingerprints=fingerprints,
                inputs=inputs,
                outputs=outputs,
            )
            fingerprint = payload["dataset_fingerprint"]
            common.serialization.atomic_torch_save(payload, staged_dataset_path)
            del payload, inputs, outputs
            inputs = None
            outputs = None
            gc.collect()
            staged_payload = torch.load(staged_dataset_path, map_location="cpu", weights_only=False)
            staged_identity = datasets.identity.validate_training_dataset_payload(staged_payload, task=task, verify_content=True)
            del staged_payload
            gc.collect()
            staged_dataset_sha256 = generated.sha256_file(staged_dataset_path)
            staged_dataset_size = staged_dataset_path.stat().st_size
            _stage_metadata_package(
                stage_metadata_dir,
                dataset_identity=staged_identity,
                dataset_sha256=staged_dataset_sha256,
                dataset_size=staged_dataset_size,
                task=task,
                manifest_snapshot=manifest_snapshot,
                manifest_sha256=manifest_sha256,
                manifest_identity_sha256=manifest_identity_sha256,
                sample_csv_snapshot=sample_csv_snapshot,
                sample_json_snapshot=sample_json_snapshot,
                sample_csv_sha256=manifest["configuration"]["sample_sha256"],
                sample_json_sha256=sample_json_sha256,
                timing_snapshot=timing_snapshot,
                timing_summary=timing_summary,
            )
            datasets.metadata.validate_dataset_metadata_directory(
                stage_metadata_dir,
                dataset_identity=staged_identity,
                dataset_path=staged_dataset_path,
            )
            staged_metadata_sha256 = generated.sha256_file(stage_metadata_dir / datasets.metadata.METADATA_FILENAME)
            generated.assert_generation_snapshot_current(raw_dir, manifest_path, manifest_snapshot)
            common.serialization.atomic_write_json(
                transaction_path,
                _publication_transaction_record(
                    dataset_id=resolved_dataset_id,
                    phase="ready",
                    staging_root=staging_root,
                    dataset_sha256=staged_dataset_sha256,
                    dataset_size=staged_dataset_size,
                    dataset_metadata_sha256=staged_metadata_sha256,
                ),
            )
            transaction_phase = "ready"
            training_meta_root.mkdir(parents=True, exist_ok=True)
            training_raw_root.mkdir(parents=True, exist_ok=True)
            if destination_dir.exists() or destination_dir.is_symlink() or metadata_destination.exists() or metadata_destination.is_symlink():
                msg = "Final dataset or metadata target appeared during the build transaction."
                raise FileExistsError(msg)
            generated.assert_generation_snapshot_current(raw_dir, manifest_path, manifest_snapshot)
            stage_metadata_dir.replace(metadata_destination)
            metadata_published = True
            stage_dataset_dir.replace(destination_dir)
            dataset_published = True
            datasets.metadata.validate_dataset_metadata_directory(
                metadata_destination,
                dataset_identity=staged_identity,
                dataset_path=destination,
            )
            if (
                destination.stat().st_size != staged_dataset_size
                or generated.sha256_file(destination) != staged_dataset_sha256
                or generated.sha256_file(metadata_destination / datasets.metadata.METADATA_FILENAME) != staged_metadata_sha256
            ):
                msg = "Final dataset publication changed after its ready transaction was recorded."
                raise RuntimeError(msg)
            publication_complete = True
            transaction_path.unlink()
            transaction_active = False
        finally:
            del inputs, outputs
            if not publication_complete and transaction_phase == "ready":
                if dataset_published and destination_dir.is_dir() and not stage_dataset_dir.exists():
                    stage_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
                    destination_dir.replace(stage_dataset_dir)
                if metadata_published and metadata_destination.is_dir() and not stage_metadata_dir.exists():
                    stage_metadata_dir.parent.mkdir(parents=True, exist_ok=True)
                    metadata_destination.replace(stage_metadata_dir)
            if (publication_complete or not transaction_active) and staging_root.exists():
                shutil.rmtree(staging_root)

    result = {
        "source_batch": batch_name,
        "generated_data_root": generated_root,
        "dataset_path": destination,
        "metadata_path": metadata_destination,
        "case_count": len(case_ids),
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "timing_coverage": timing_summary,
        "dataset_fingerprint": fingerprint,
        "status": "complete",
    }
    if verbose:
        for key, value in result.items():
            print(f"{key}: {value}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one final training dataset directly from a completed COMSOL batch.")
    parser.add_argument("batch_id", help="Completed generated batch and final dataset identifier")
    parser.add_argument("--task", default="steady_flow", help="Registered task identifier")
    parser.add_argument("--generated-data-root", type=Path, default=None, help="Override GENERATED_DATA_ROOT for this invocation")
    parser.add_argument("--model-training-data-root", type=Path, default=None, help="Override MODEL_TRAINING_DATA_ROOT for this invocation")
    parser.add_argument("--verbose", action="store_true", help="Show bounded progress and final identity")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the maintained direct final-dataset build command."""
    args = _build_parser().parse_args(argv)
    try:
        result = build_batch_dataset(
            args.batch_id,
            task_id=args.task,
            generated_data_root=args.generated_data_root,
            model_training_data_root=args.model_training_data_root,
            verbose=args.verbose,
        )
    except Exception as error:  # noqa: BLE001
        print(f"Dataset build failed: {type(error).__name__}: {error}")
        return 1
    print(f"Source batch: {result['source_batch']}")
    print(f"Generated data root: {result['generated_data_root']}")
    print(f"Destination dataset: {result['dataset_path']}")
    print(f"Metadata destination: {result['metadata_path']}")
    print(f"Case count: {result['case_count']}")
    print(f"Task identity: {result['task']} ({result['task_contract_digest']})")
    coverage = result["timing_coverage"]
    print(f"COMSOL timing coverage: {coverage['measured_case_count']}/{coverage['intended_case_count']} ({coverage['status']})")
    print(f"Dataset fingerprint: {result['dataset_fingerprint']}")
    print(f"Status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
