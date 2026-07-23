"""
===============================================================================
cli_build_artifacts.py
===============================================================================
Generate split-aware analysis artifacts for current saved training runs.

Responsibilities:
  - Discover current run directories by the saved-run artifact contract
  - Load split-aware inference contexts for ID and OOD evaluation
  - Generate or reuse Parquet and NPZ analysis artifacts
  - Release GPU memory between run evaluations

Design principles:
  - CLI orchestration stays thin
  - Existing artifacts are reused when available
  - Inference and artifact generation are delegated to reusable modules
  - Run and artifact paths are resolved through common.paths

Boundaries:
  - Inference context loading belongs to learning.inference.context
  - Artifact writing belongs to analysis.artifacts
  - Run-directory and artifact path conventions belong to common.paths

Notes:
  - Runs are discovered by the current contract files:
      config.yaml, best_checkpoint.pt, normalizer.pt, split_indices.pt
  - ID artifacts use the saved eval split and are written under analysis/id
  - OOD artifacts use the saved OOD split and are written under
    analysis/ood/<dataset_name>
===============================================================================

"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from src import analysis, common, datasets, experiments, learning

# ======================================================================
# Global config
# ======================================================================

DEFAULT_TASK = "steady_flow"
DEFAULT_BATCH_SIZE = 1

# Case limit for testing / debugging (set to None for full split evaluation)
MAX_CASES: int | None = None

ArtifactSplit = Literal["eval", "ood"]
_SPLIT_INDEX_KEYS: dict[ArtifactSplit, str] = {
    "eval": "eval_indices",
    "ood": "ood_indices",
}
_SPLIT_DATASET_KEYS: dict[ArtifactSplit, str] = {
    "eval": "train_dataset",
    "ood": "ood_dataset",
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
    """Raised when existing artifacts cannot be proven compatible and complete."""


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


def _format_missing_files(paths: Iterable[Path]) -> str:
    """Format missing required run files for error messages."""
    return "\n".join(f"  - {path}" for path in paths)


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        msg = f"Expected a positive integer, got {value!r}."
        raise argparse.ArgumentTypeError(msg)
    return parsed


def iter_run_dirs(root: Path, *, run_names: Iterable[str] | None = None) -> Iterable[Path]:
    """
    Iterate over current-contract run directories in sorted order.

    A directory is considered a valid current run when it contains:
    config.yaml, best_checkpoint.pt, normalizer.pt and split_indices.pt.

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
        for run_name in selected_run_names:
            run_dir = root / run_name
            missing = common.paths.missing_current_run_files(run_dir)
            if missing:
                msg = f"Requested run does not satisfy the current run contract: {run_dir}\n{_format_missing_files(missing)}"
                raise FileNotFoundError(msg)
            yield run_dir
        return

    if common.paths.is_current_run_dir(root):
        yield root
        return

    if not root.is_dir():
        msg = f"Run discovery root not found: {root}"
        raise FileNotFoundError(msg)

    for candidate in sorted(root.iterdir()):
        if common.paths.is_current_run_dir(candidate):
            yield candidate


def _dataset_name_from_saved_path(value: Any, *, label: str) -> str:
    """Extract a logical dataset name from saved split metadata path."""
    if not isinstance(value, str) or not value:
        msg = f"split_indices.pt metadata missing non-empty {label}."
        raise RuntimeError(msg)

    dataset_path = Path(value)
    if dataset_path.suffix:
        return dataset_path.stem
    return dataset_path.name


def _load_split_contract(run_dir: Path) -> Mapping[str, Any]:
    """Load the saved split contract used by inference and provenance."""
    split_indices_path = common.paths.resolve_split_indices_path(run_dir)
    split_indices = torch.load(split_indices_path, map_location="cpu")
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
    if not isinstance(value, str) or not value:
        msg = f"config.yaml data.{key} must be a non-empty string."
        raise TypeError(msg)
    return value


def _required_config_ood_dataset_names(data_cfg: Mapping[str, Any]) -> list[str]:
    """Return configured OOD dataset names from config data."""
    value = data_cfg.get("ood_datasets")
    if not isinstance(value, list) or not value:
        msg = "config.yaml data.ood_datasets must be a non-empty list."
        raise TypeError(msg)
    if not all(isinstance(name, str) and name for name in value):
        msg = "config.yaml data.ood_datasets must contain non-empty strings."
        raise TypeError(msg)
    return value


def load_run_artifact_plan(run_dir: Path) -> RunArtifactPlan:
    """
    Build the artifact plan for a current saved run.

    The dataset names used for artifact paths come from split_indices.pt
    metadata, then are checked against config.yaml data settings. This keeps
    output names aligned with the same saved split identity consumed by
    learning.inference.context.load_inference_context().
    """
    run_dir = Path(run_dir)
    data_cfg = _load_data_config(run_dir)
    split_metadata = _load_split_metadata(run_dir)

    id_dataset_name = _dataset_name_from_saved_path(split_metadata.get("train_dataset"), label="train_dataset")
    ood_dataset_name = _dataset_name_from_saved_path(split_metadata.get("ood_dataset"), label="ood_dataset")

    configured_id_dataset = _required_config_dataset_name(data_cfg, "train_dataset")
    configured_ood_datasets = _required_config_ood_dataset_names(data_cfg)

    if configured_id_dataset != id_dataset_name:
        msg = (
            "config.yaml data.train_dataset does not match split_indices.pt metadata.\n"
            f"config:   {configured_id_dataset}\n"
            f"metadata: {id_dataset_name}"
        )
        raise RuntimeError(msg)

    if ood_dataset_name not in configured_ood_datasets:
        msg = (
            "split_indices.pt metadata OOD dataset is not listed in config.yaml data.ood_datasets.\n"
            f"config:   {configured_ood_datasets}\n"
            f"metadata: {ood_dataset_name}"
        )
        raise RuntimeError(msg)

    extra_ood_datasets = [name for name in configured_ood_datasets if name != ood_dataset_name]
    if extra_ood_datasets:
        print(f"[INFO] Current split_indices.pt stores one OOD split; building artifacts only for saved OOD dataset {ood_dataset_name!r}.")

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
    """Return a canonical absolute path without requiring the target to exist."""
    return path.expanduser().resolve(strict=False)


def _required_metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    """Return one non-negative integer from saved split metadata."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"split_indices.pt metadata {key!r} must be a non-negative integer."
        raise TypeError(msg)
    return value


def _indices_sha256(indices: Iterable[int]) -> str:
    """Hash ordered split membership without hashing dataset tensor contents."""
    return analysis.artifacts.ordered_indices_sha256(indices)


def _file_identity(path: Path) -> dict[str, Any]:
    """Describe one load-bearing run file for same-directory cache invalidation."""
    stat = path.stat()
    return {
        "path": str(_normalise_path(path)),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _build_artifact_request(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    max_cases: int | None,
) -> ArtifactRequest:
    """Build deterministic expected provenance from the current saved run."""
    run_dir = Path(run_dir)
    if max_cases is not None and max_cases <= 0:
        msg = f"max_cases must be positive when provided, got {max_cases}."
        raise ValueError(msg)

    missing = common.paths.missing_current_run_files(run_dir)
    if missing:
        msg = f"Run does not satisfy the artifact contract: {run_dir}\n{_format_missing_files(missing)}"
        raise FileNotFoundError(msg)

    split_contract = _load_split_contract(run_dir)
    metadata = _split_metadata(split_contract, run_dir=run_dir)
    run_config = _load_run_config(run_dir)
    data_cfg = run_config.get("data")
    run_cfg = run_config.get("run")
    if not isinstance(data_cfg, Mapping) or not isinstance(run_cfg, Mapping):
        msg = f"config.yaml must contain data and run mappings: {common.paths.resolve_run_config_path(run_dir)}"
        raise TypeError(msg)
    required_settings = {
        "data.train_ratio": (data_cfg, "train_ratio"),
        "data.ood_fraction": (data_cfg, "ood_fraction"),
        "run.seed": (run_cfg, "seed"),
    }
    missing_settings = [label for label, (section, key) in required_settings.items() if key not in section]
    if missing_settings:
        msg = f"config.yaml is missing saved split setting(s): {', '.join(missing_settings)}."
        raise KeyError(msg)
    validated_indices = datasets.base.validate_split_info(
        split_contract,
        expected_train_ratio=data_cfg["train_ratio"],
        expected_ood_fraction=data_cfg["ood_fraction"],
        expected_split_seed=run_cfg["seed"],
    )
    saved_indices = validated_indices[_SPLIT_INDEX_KEYS[split]]
    full_source_indices = tuple(int(value) for value in saved_indices.tolist())

    count_key = _SPLIT_COUNT_KEYS[split]
    saved_selected_count = _required_metadata_int(metadata, count_key)
    if saved_selected_count != len(full_source_indices):
        msg = f"split_indices.pt metadata {count_key!r}={saved_selected_count} does not match the saved index count {len(full_source_indices)}."
        raise RuntimeError(msg)

    full_count_key = _SPLIT_FULL_COUNT_KEYS[split]
    dataset_full_count = _required_metadata_int(metadata, full_count_key)
    if dataset_full_count <= 0:
        msg = f"split_indices.pt metadata {full_count_key!r} must be positive."
        raise ValueError(msg)
    if min(full_source_indices) < 0 or max(full_source_indices) >= dataset_full_count:
        msg = f"Saved {split!r} indices are out of bounds for metadata {full_count_key!r}={dataset_full_count}."
        raise RuntimeError(msg)

    dataset_key = _SPLIT_DATASET_KEYS[split]
    saved_dataset_path_raw = metadata.get(dataset_key)
    saved_dataset_name = _dataset_name_from_saved_path(saved_dataset_path_raw, label=dataset_key)
    if saved_dataset_name != dataset_name:
        msg = f"Requested dataset {dataset_name!r} does not match saved {split!r} dataset {saved_dataset_name!r}."
        raise RuntimeError(msg)
    if not isinstance(saved_dataset_path_raw, str):
        msg = f"split_indices.pt metadata {dataset_key!r} must be a path string."
        raise TypeError(msg)
    saved_dataset_path = _normalise_path(Path(saved_dataset_path_raw))
    if not saved_dataset_path.is_file():
        msg = f"Saved {split!r} dataset file is missing: {saved_dataset_path}"
        raise FileNotFoundError(msg)

    effective_count = len(full_source_indices) if max_cases is None else min(len(full_source_indices), max_cases)
    effective_source_indices = full_source_indices[:effective_count]
    contract_paths = common.paths.resolve_current_run_required_paths(run_dir)
    contract_files = {path.name: _file_identity(path) for path in contract_paths}

    provenance: dict[str, Any] = {
        "provenance_schema_version": analysis.artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
        "run": {
            "name": run_dir.name,
            "path": str(_normalise_path(run_dir)),
            "contract_files": contract_files,
        },
        "split_role": split,
        "dataset": {
            "name": dataset_name,
            "path": str(saved_dataset_path),
            "full_case_count": dataset_full_count,
            "current_file": _file_identity(saved_dataset_path),
            "identity_basis": "saved_path_full_case_count_and_current_file_stat",
        },
        "selection": {
            "index_key": _SPLIT_INDEX_KEYS[split],
            "full_selected_case_count": len(full_source_indices),
            "effective_case_count": effective_count,
            "generation_limit": max_cases,
            "full_ordered_source_indices_sha256": _indices_sha256(full_source_indices),
            "effective_ordered_source_indices_sha256": _indices_sha256(effective_source_indices),
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
            analysis.artifacts.artifact_provenance_path(save_root).exists(),
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


def _require_identity_meta(raw_meta: Any, *, case_index: int, source_index: int, split_local_index: int, label: str) -> None:
    """Validate identity fields embedded in one JSON metadata payload."""
    if not isinstance(raw_meta, str):
        msg = f"{label} metadata must be a JSON string."
        raise ArtifactCacheError(msg)
    try:
        metadata = json.loads(raw_meta)
    except json.JSONDecodeError as error:
        msg = f"{label} metadata is invalid JSON: {error}"
        raise ArtifactCacheError(msg) from error
    if not isinstance(metadata, Mapping):
        msg = f"{label} metadata must decode to an object."
        raise ArtifactCacheError(msg)

    expected = {
        "case_index": case_index,
        "source_index": source_index,
        "split_local_index": split_local_index,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        msg = f"{label} metadata identity mismatch: expected {expected}, got {actual}."
        raise ArtifactCacheError(msg)


def _read_npz_identity(path: Path) -> tuple[set[str], dict[str, int], Any]:
    """Read only the small identity payload from one NPZ archive."""
    try:
        with np.load(path, allow_pickle=False) as artifact:
            fields = set(artifact.files)
            actual = {
                "case_index": int(artifact["case_index"].item()),
                "source_index": int(artifact["source_index"].item()),
                "split_local_index": int(artifact["split_local_index"].item()),
            }
            raw_meta = artifact["meta"].item()
    except (OSError, ValueError, KeyError) as error:
        msg = f"Cached NPZ is unreadable or incompatible: {path}: {error}"
        raise ArtifactCacheError(msg) from error
    return fields, actual, raw_meta


def _validate_npz_identity(
    path: Path,
    *,
    case_index: int,
    source_index: int,
    split_local_index: int,
) -> None:
    """Validate scalar and metadata identity stored in one NPZ payload."""
    fields, actual, raw_meta = _read_npz_identity(path)
    required = {"case_index", "source_index", "split_local_index", "meta"}
    missing = required.difference(fields)
    if missing:
        msg = f"Cached NPZ is missing identity fields {sorted(missing)}: {path}"
        raise ArtifactCacheError(msg)

    expected = {
        "case_index": case_index,
        "source_index": source_index,
        "split_local_index": split_local_index,
    }
    if actual != expected:
        msg = f"Cached NPZ identity mismatch for {path}: expected {expected}, got {actual}."
        raise ArtifactCacheError(msg)

    _require_identity_meta(
        raw_meta,
        case_index=case_index,
        source_index=source_index,
        split_local_index=split_local_index,
        label=f"Cached NPZ {path}",
    )


def _load_validated_artifact_cache(
    *,
    save_root: Path,
    parquet_path: Path,
    npz_dir: Path,
    request: ArtifactRequest,
) -> pd.DataFrame:
    """Load a cache only after exact provenance, membership and payload checks."""
    provenance_path = analysis.artifacts.artifact_provenance_path(save_root)
    if not provenance_path.is_file():
        msg = f"Existing artifact cache has no provenance sidecar: {provenance_path}. Refusing to trust or overwrite legacy/partial artifacts."
        raise ArtifactCacheError(msg)
    if not parquet_path.is_file():
        msg = f"Artifact cache is incomplete (Parquet missing): {parquet_path}"
        raise ArtifactCacheError(msg)

    actual_provenance = _read_artifact_provenance(provenance_path)
    if actual_provenance != request.provenance:
        expected_json = json.dumps(request.provenance, indent=2, sort_keys=True)
        actual_json = json.dumps(dict(actual_provenance), indent=2, sort_keys=True)
        msg = (
            f"Artifact provenance is incompatible: {provenance_path}\n"
            f"Expected:\n{expected_json}\nActual:\n{actual_json}\n"
            "Refusing to trust or overwrite the existing cache."
        )
        raise ArtifactCacheError(msg)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as error:
        msg = f"Cached Parquet is unreadable: {parquet_path}: {error}"
        raise ArtifactCacheError(msg) from error

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
    if "npz_path" not in df.columns or "meta" not in df.columns:
        msg = "Cached Parquet must contain npz_path and meta columns."
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
        _require_identity_meta(
            row.loc["meta"],
            case_index=case_index,
            source_index=source_index,
            split_local_index=split_local_index,
            label=f"Cached Parquet row {row_position}",
        )
        _validate_npz_identity(
            expected_npz_path,
            case_index=case_index,
            source_index=source_index,
            split_local_index=split_local_index,
        )

    return df


def run_or_load_artifacts(
    *,
    run_dir: Path,
    dataset_name: str,
    split: ArtifactSplit,
    max_cases: int | None,
    batch_size: int,
    prefer_cuda: bool,
) -> pd.DataFrame:
    """
    Load or generate artifacts for one run and saved split.

    Reuses existing Parquet+NPZ artifacts only after exact provenance and
    payload validation. If artifacts do not exist, runs split-aware inference via
    learning.inference.context.load_inference_context() and generates
    Parquet+NPZ artifacts via analysis.artifacts.generate_artifacts().

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

    Returns
    -------
    pandas.DataFrame
        Artifact summary DataFrame, or empty if artifacts were skipped.

    Notes
    -----
    - Artifacts cached in analysis/id or analysis/ood/<dataset_name>/.
    - No full-dataset fallback is used for normal eval or OOD evaluation.
    - Missing, incompatible or partial cache provenance fails loudly and is never overwritten.
    - Exceptions during inference are logged and return empty DataFrame.

    """
    if batch_size != 1:
        msg = "Artifact generation currently requires batch_size=1 to preserve one output row and NPZ file per case."
        raise ValueError(msg)
    if max_cases is not None and max_cases <= 0:
        msg = f"max_cases must be positive when provided, got {max_cases}."
        raise ValueError(msg)

    save_root = _artifact_save_root(run_dir=run_dir, dataset_name=dataset_name, split=split)
    npz_dir = save_root / "npz"
    parquet_path = save_root / f"{dataset_name}.parquet"

    request = _build_artifact_request(
        run_dir=run_dir,
        dataset_name=dataset_name,
        split=split,
        max_cases=max_cases,
    )

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

    try:
        model, loader, processor, device = learning.inference.context.load_inference_context(
            run_dir=run_dir,
            split=split,
            batch_size=batch_size,
            prefer_cuda=prefer_cuda,
        )
    except Exception as error:  # noqa: BLE001
        print(f"[SKIP] {run_dir.name} | {split} | {dataset_name}")
        print(f"       Reason: {type(error).__name__}: {error}")
        return pd.DataFrame()

    try:
        analysis.artifacts.generate_artifacts(
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


# ======================================================================
# CLI
# ======================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the artifact generation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate split-aware analysis artifacts for current saved training runs.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task name used to resolve the default run discovery root.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory containing run directories, or a single current run directory.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=None,
        help="Run directory name under --runs-root to process. May be repeated.",
    )
    parser.add_argument(
        "--max-cases",
        type=_positive_int,
        default=MAX_CASES,
        help="Optional positive maximum number of cases to process from each saved split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1,),
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size for artifact generation (currently fixed at 1 for per-case artifacts).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable CUDA preference for inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Generate analysis artifacts for current saved runs.

    Orchestrates evaluation across all current run directories found under the
    resolved runs root. For each run, ID artifacts are built from the saved eval
    split and OOD artifacts are built from the saved OOD split recorded in
    split_indices.pt.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    runs_root = args.runs_root if args.runs_root is not None else common.paths.resolve_runs_root(args.task)
    run_dirs = list(iter_run_dirs(runs_root, run_names=args.run_names))
    print(f"[INFO] Found {len(run_dirs)} current runs in {runs_root}")

    for run_dir in run_dirs:
        print(f"\n=== {run_dir.name} ===")

        # Invalid or ambiguous saved-run metadata is a contract failure, not a
        # skippable run. Let the exception produce a non-zero CLI exit.
        plan = load_run_artifact_plan(run_dir)

        # -----------------
        # ID / eval split
        # -----------------
        df_raw_id = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.id_dataset_name,
            split="eval",
            max_cases=args.max_cases,
            batch_size=args.batch_size,
            prefer_cuda=not args.cpu,
        )
        if df_raw_id.empty:
            print(f"[SKIP] {run_dir.name} | ID evaluation skipped")
            continue
        _ = analysis.evaluation.dataframe.build_eval_df(df_raw_id)

        # -----------------
        # OOD split
        # -----------------
        df_raw_ood = run_or_load_artifacts(
            run_dir=plan.run_dir,
            dataset_name=plan.ood_dataset_name,
            split="ood",
            max_cases=args.max_cases,
            batch_size=args.batch_size,
            prefer_cuda=not args.cpu,
        )

        if df_raw_ood.empty:
            print(f"[SKIP] {run_dir.name} | OOD {plan.ood_dataset_name} skipped")
            cleanup_gpu()
            continue

        _ = analysis.evaluation.dataframe.build_eval_df(df_raw_ood)
        cleanup_gpu()

    print("\n[DONE] All artifacts generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
