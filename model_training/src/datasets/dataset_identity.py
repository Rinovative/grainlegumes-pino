"""
===============================================================================
dataset_identity.py
===============================================================================
Define and verify persisted final-dataset and split identity contracts.

Responsibilities:
  - Validate the single current task-aware training-dataset schema
  - Compute stable case, dataset, and split-membership fingerprints
  - Bind persisted tensors and ordered sample identities to task contracts

Design principles:
  - Identity is content-addressed, deterministic, and fail-closed
  - Fingerprints are computed before publication and verified on consumption
  - Ordered membership remains distinct from dataset-level content identity

This module does NOT:
  - Load final datasets into model-ready samples; ``dataset_simulation`` owns that
  - Choose training/evaluation ratios or random seeds; experiment services own them
  - Publish dataset files; the generation-domain builder owns publication
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import domain

if TYPE_CHECKING:
    from torch import Tensor

    from src.domain.tasks.domain_task_spec import TaskSpec

TRAINING_DATASET_SCHEMA_VERSION = 1
TRAINING_DATASET_SCHEMA_KIND = "training_dataset"
GENERATED_BATCH_IDENTITY_SCHEMA_VERSION = 1
CASE_FINGERPRINT_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = 64
_TENSOR_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_FINITE_CHECK_CHUNK_ELEMENTS = 1024 * 1024

_GENERATED_BATCH_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "batch_name",
        "configuration",
        "field_schema",
        "intended_case_ids",
        "scientific_case_sources",
        "sampling",
        "batch_manifest_identity_sha256",
    }
)
_GENERATED_CONFIGURATION_KEYS = frozenset(
    {
        "method",
        "variation",
        "N",
        "seed",
        "Lx",
        "Ly",
        "res",
        "save_model",
        "template_name",
        "template_sha256",
    }
)
_GENERATED_FIELD_SCHEMA_KEYS = frozenset({"input_columns", "solution_columns"})
_GENERATED_CASE_SOURCE_KEYS = frozenset({"case_id", "raw_csv_sha256", "solution_csv_sha256", "solution_model_sha256"})
_GENERATED_SAMPLING_KEYS = frozenset({"method", "variation", "N", "seed", "base", "param_names"})

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "schema_kind",
        "dataset_id",
        "task",
        "task_contract_digest",
        "fields",
        "tensor_layout",
        "sample_count",
        "spatial_shape",
        "sample_ids",
        "generated_batch_identity",
        "source_identities",
        "source_metadata",
        "source_provenance",
        "case_fingerprints",
        "tensor_metadata",
        "inputs",
        "outputs",
        "dataset_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Portable ordered identity returned after strict dataset validation."""

    dataset_id: str
    task: str
    task_contract_digest: str
    fingerprint: str
    sample_ids: tuple[str, ...]
    sample_count: int
    spatial_shape: tuple[int, ...]
    generated_batch_identity_sha256: str | None = field(default=None, compare=False, repr=False)
    generated_batch_identity: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    source_metadata: tuple[dict[str, Any], ...] | None = field(default=None, compare=False, repr=False)
    source_provenance: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation used by split identity."""
        return {
            "dataset_id": self.dataset_id,
            "task": self.task,
            "task_contract_digest": self.task_contract_digest,
            "fingerprint": self.fingerprint,
            "sample_ids": list(self.sample_ids),
            "sample_count": self.sample_count,
            "spatial_shape": list(self.spatial_shape),
        }


def _canonical_json(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        msg = f"{label} must be JSON-serializable without non-finite values."
        raise TypeError(msg) from error


def _update_hash(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _tensor_dtype(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _update_tensor_hash(hasher: Any, tensor: Tensor) -> None:
    import torch  # noqa: PLC0415

    contiguous = tensor.detach().cpu()
    if not contiguous.is_contiguous():
        contiguous = contiguous.contiguous()
    byte_count = contiguous.numel() * contiguous.element_size()
    hasher.update(byte_count.to_bytes(8, byteorder="big", signed=False))
    byte_view = contiguous.view(torch.uint8).numpy().data.cast("B")
    for offset in range(0, len(byte_view), _TENSOR_HASH_CHUNK_BYTES):
        hasher.update(byte_view[offset : offset + _TENSOR_HASH_CHUNK_BYTES])


def _content_fingerprint(metadata: Mapping[str, Any], tensors: Sequence[tuple[str, Tensor]]) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, _canonical_json(dict(metadata), label="Fingerprint metadata"))
    for label, tensor in tensors:
        _update_hash(hasher, label.encode("utf-8"))
        _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def _require_non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string."
        raise TypeError(msg)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _require_non_empty_string(value, label=label)
    if len(digest) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in digest):
        msg = f"{label} must be a 64-character lowercase hexadecimal SHA-256 digest."
        raise ValueError(msg)
    return digest


def _require_string_sequence(value: Any, *, label: str, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        msg = f"{label} must be a list or tuple of strings."
        raise TypeError(msg)
    values = tuple(_require_non_empty_string(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if unique and len(values) != len(set(values)):
        duplicates = sorted({item for item in values if values.count(item) > 1})
        msg = f"{label} contains duplicate identifiers: {duplicates}."
        raise ValueError(msg)
    return values


def _require_sha256_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    values = _require_string_sequence(value, label=label, unique=False)
    return tuple(_require_sha256(item, label=f"{label}[{index}]") for index, item in enumerate(values))


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    if value <= 0:
        msg = f"{label} must be positive, got {value}."
        raise ValueError(msg)
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _json_mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    mapping = dict(_require_mapping(value, label=label))
    return json.loads(_canonical_json(mapping, label=label).decode("utf-8"))


def _require_exact_mapping(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Return an isolated JSON mapping with exactly the declared keys."""
    normalized = _json_mapping_copy(value, label=label)
    missing = sorted(set(expected).difference(normalized))
    unexpected = sorted(set(normalized).difference(expected))
    if missing or unexpected:
        msg = f"{label} keys do not match. Missing: {missing}; unexpected: {unexpected}."
        raise ValueError(msg)
    return normalized


def _require_manifest_number(value: Any, *, label: str, positive: bool) -> float:
    """Validate one finite manifest number without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real number."
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        relation = "positive" if positive else "non-negative"
        msg = f"{label} must be finite and {relation}."
        raise ValueError(msg)
    return result


def _validate_generated_configuration(value: Any, *, label: str) -> dict[str, Any]:
    """Validate the scientific subset of one admitted manifest configuration."""
    configuration = _require_exact_mapping(value, _GENERATED_CONFIGURATION_KEYS, label=label)
    if configuration["method"] not in {"uniform", "lhs", "sobol"}:
        msg = f"{label}.method is invalid."
        raise ValueError(msg)
    for name in ("N", "seed"):
        numeric = configuration[name]
        if isinstance(numeric, bool) or not isinstance(numeric, int):
            msg = f"{label}.{name} must be an integer."
            raise TypeError(msg)
    if configuration["N"] <= 0 or configuration["seed"] < 0:
        msg = f"{label} N and seed must be non-negative with positive N."
        raise ValueError(msg)
    _require_manifest_number(
        configuration["variation"],
        label=f"{label}.variation",
        positive=False,
    )
    lengths = {
        name: _require_manifest_number(
            configuration[name],
            label=f"{label}.{name}",
            positive=True,
        )
        for name in ("Lx", "Ly", "res")
    }
    if lengths["res"] > min(lengths["Lx"], lengths["Ly"]):
        msg = f"{label}.res cannot exceed the shorter domain length."
        raise ValueError(msg)
    if not isinstance(configuration["save_model"], bool):
        msg = f"{label}.save_model must be boolean."
        raise TypeError(msg)
    template_name = _require_non_empty_string(
        configuration["template_name"],
        label=f"{label}.template_name",
    )
    if Path(template_name).name != template_name or "/" in template_name or "\\" in template_name or not template_name.endswith(".mph"):
        msg = f"{label}.template_name must be an .mph basename."
        raise ValueError(msg)
    _require_sha256(configuration["template_sha256"], label=f"{label}.template_sha256")
    return configuration


def _validate_generated_field_schema(value: Any, *, label: str) -> dict[str, Any]:
    """Validate exact generated source-column declarations."""
    field_schema = _require_exact_mapping(value, _GENERATED_FIELD_SCHEMA_KEYS, label=label)
    for name in ("input_columns", "solution_columns"):
        columns = _require_string_sequence(
            field_schema[name],
            label=f"{label}.{name}",
            unique=True,
        )
        if not columns:
            msg = f"{label}.{name} must not be empty."
            raise ValueError(msg)
    return field_schema


def _validate_generated_case_sources(
    value: Any,
    *,
    intended: Sequence[str],
    save_model: bool,
    label: str,
) -> list[dict[str, Any]]:
    """Validate ordered scientific source digests for every intended case."""
    if not isinstance(value, list) or len(value) != len(intended):
        msg = f"{label} must align one-to-one with intended_case_ids."
        raise ValueError(msg)
    normalized: list[dict[str, Any]] = []
    for index, (case_id, source_value) in enumerate(zip(intended, value, strict=True)):
        item_label = f"{label}[{index}]"
        source = _require_exact_mapping(
            source_value,
            _GENERATED_CASE_SOURCE_KEYS,
            label=item_label,
        )
        if source["case_id"] != case_id:
            msg = f"{label} must follow intended_case_ids exactly."
            raise ValueError(msg)
        _require_sha256(source["raw_csv_sha256"], label=f"{item_label}.raw_csv_sha256")
        _require_sha256(
            source["solution_csv_sha256"],
            label=f"{item_label}.solution_csv_sha256",
        )
        model_digest = source["solution_model_sha256"]
        if save_model:
            _require_sha256(model_digest, label=f"{item_label}.solution_model_sha256")
        elif model_digest != "":
            msg = f"{item_label}.solution_model_sha256 must be empty when save_model is false."
            raise ValueError(msg)
        normalized.append(source)
    return normalized


def _validate_generated_sampling(
    value: Any,
    *,
    configuration: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Validate portable parameter-sampling identity against the manifest."""
    sampling = _require_exact_mapping(value, _GENERATED_SAMPLING_KEYS, label=label)
    for name in ("N", "seed"):
        numeric = sampling[name]
        if isinstance(numeric, bool) or not isinstance(numeric, int):
            msg = f"{label}.{name} must be an integer."
            raise TypeError(msg)
    _require_manifest_number(
        sampling["variation"],
        label=f"{label}.variation",
        positive=False,
    )
    for name in ("method", "variation", "N", "seed"):
        if sampling[name] != configuration[name]:
            msg = f"{label}.{name} must match the generated configuration."
            raise ValueError(msg)
    sampling["base"] = _json_mapping_copy(sampling["base"], label=f"{label}.base")
    parameter_names = _require_string_sequence(
        sampling["param_names"],
        label=f"{label}.param_names",
        unique=True,
    )
    if not parameter_names:
        msg = f"{label}.param_names must not be empty."
        raise ValueError(msg)
    return sampling


def _validate_generated_batch_identity(
    value: Any,
    *,
    sample_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Validate the complete version-1 scientific generated-batch identity."""
    identity = _require_exact_mapping(value, _GENERATED_BATCH_IDENTITY_KEYS, label=label)
    schema_version = identity["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != GENERATED_BATCH_IDENTITY_SCHEMA_VERSION:
        msg = f"{label}.schema_version must be integer {GENERATED_BATCH_IDENTITY_SCHEMA_VERSION}."
        raise ValueError(msg)
    _require_non_empty_string(identity["batch_name"], label=f"{label}.batch_name")
    configuration = _validate_generated_configuration(
        identity["configuration"],
        label=f"{label}.configuration",
    )
    identity["field_schema"] = _validate_generated_field_schema(
        identity["field_schema"],
        label=f"{label}.field_schema",
    )
    intended = _require_string_sequence(
        identity["intended_case_ids"],
        label=f"{label}.intended_case_ids",
        unique=True,
    )
    if intended != tuple(sample_ids):
        msg = f"{label}.intended_case_ids must equal the dataset sample_ids in order."
        raise ValueError(msg)
    if len(intended) != configuration["N"]:
        msg = f"{label}.intended_case_ids must contain exactly configuration.N cases."
        raise ValueError(msg)
    identity["scientific_case_sources"] = _validate_generated_case_sources(
        identity["scientific_case_sources"],
        intended=intended,
        save_model=configuration["save_model"],
        label=f"{label}.scientific_case_sources",
    )
    identity["sampling"] = _validate_generated_sampling(
        identity["sampling"],
        configuration=configuration,
        label=f"{label}.sampling",
    )
    expected_digest = hashlib.sha256(
        _canonical_json(
            {key: identity[key] for key in identity if key != "batch_manifest_identity_sha256"},
            label=label,
        )
    ).hexdigest()
    actual_digest = _require_sha256(
        identity["batch_manifest_identity_sha256"],
        label=f"{label}.batch_manifest_identity_sha256",
    )
    if actual_digest != expected_digest:
        msg = f"{label}.batch_manifest_identity_sha256 does not match its scientific content."
        raise ValueError(msg)
    return identity


def build_generated_batch_identity(
    source_manifest: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the version-1 scientific identity from admitted sources."""
    manifest = _require_mapping(source_manifest, label="Source manifest")
    configuration = _require_mapping(
        manifest.get("configuration"),
        label="Source manifest.configuration",
    )
    intended = _require_string_sequence(
        manifest.get("intended_case_ids"),
        label="Source manifest.intended_case_ids",
        unique=True,
    )
    records = manifest.get("cases")
    if not isinstance(records, (list, tuple)):
        msg = "Source manifest.cases must be a list or tuple."
        raise TypeError(msg)
    scientific_records: list[dict[str, Any]] = []
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, label=f"Source manifest.cases[{index}]")
        files = _require_mapping(record.get("files"), label=f"Source manifest.cases[{index}].files")
        scientific_records.append(
            {
                "case_id": record.get("case_id"),
                "raw_csv_sha256": files.get("raw_csv_sha256"),
                "solution_csv_sha256": files.get("solution_csv_sha256"),
                "solution_model_sha256": files.get("solution_model_sha256"),
            }
        )
    scientific_configuration = {key: value for key, value in configuration.items() if key != "sample_sha256"}
    content = {
        "schema_version": manifest.get("schema_version"),
        "batch_name": manifest.get("batch_name"),
        "configuration": scientific_configuration,
        "field_schema": manifest.get("field_schema"),
        "intended_case_ids": list(intended),
        "scientific_case_sources": scientific_records,
        "sampling": dict(sampling),
    }
    identity = dict(content)
    identity["batch_manifest_identity_sha256"] = hashlib.sha256(_canonical_json(content, label="Generated batch identity")).hexdigest()
    return _validate_generated_batch_identity(
        identity,
        sample_ids=intended,
        label="Generated batch identity",
    )


def _require_tensor(value: Any, *, label: str, rank: int) -> Tensor:
    import torch  # noqa: PLC0415

    if not isinstance(value, torch.Tensor):
        msg = f"{label} must be a torch.Tensor."
        raise TypeError(msg)
    tensor = value
    if tensor.layout != torch.strided:
        msg = f"{label} must be a dense strided tensor."
        raise TypeError(msg)
    if tensor.ndim != rank:
        msg = f"{label} must have rank {rank}, got shape {tuple(tensor.shape)}."
        raise ValueError(msg)
    if tensor.dtype != torch.float32:
        msg = f"{label} must use torch.float32, got {tensor.dtype}."
        raise TypeError(msg)
    cpu_tensor = tensor.detach().cpu()
    flat = cpu_tensor.reshape(-1)
    for offset in range(0, flat.numel(), _FINITE_CHECK_CHUNK_ELEMENTS):
        if not bool(torch.isfinite(flat[offset : offset + _FINITE_CHECK_CHUNK_ELEMENTS]).all().item()):
            msg = f"{label} must contain only finite values."
            raise ValueError(msg)
    return cpu_tensor


def _tensor_metadata(tensor: Tensor) -> dict[str, Any]:
    return {"dtype": _tensor_dtype(tensor), "shape": list(tensor.shape)}


def _require_exact_keys(payload: Mapping[str, Any], *, label: str) -> None:
    missing = sorted(_REQUIRED_KEYS.difference(payload))
    unexpected = sorted(set(payload).difference(_REQUIRED_KEYS))
    if missing or unexpected:
        msg = f"{label} schema keys do not match. Missing: {missing}; unexpected: {unexpected}."
        raise ValueError(msg)


def _validate_task_header(payload: Mapping[str, Any], task: TaskSpec, *, label: str) -> None:
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != TRAINING_DATASET_SCHEMA_VERSION:
        msg = f"{label} schema_version must be integer {TRAINING_DATASET_SCHEMA_VERSION}."
        raise ValueError(msg)
    if payload.get("schema_kind") != TRAINING_DATASET_SCHEMA_KIND:
        msg = f"{label} schema_kind must be {TRAINING_DATASET_SCHEMA_KIND!r}."
        raise ValueError(msg)
    if payload.get("task") != task.id:
        msg = f"{label} task must be {task.id!r}, got {payload.get('task')!r}."
        raise ValueError(msg)
    if payload.get("task_contract_digest") != task.contract_digest:
        msg = f"{label} task-contract digest does not match registered task {task.id!r}."
        raise ValueError(msg)
    fields = _require_mapping(payload.get("fields"), label=f"{label}.fields")
    if set(fields) != {"inputs", "outputs"}:
        msg = f"{label}.fields must contain exactly 'inputs' and 'outputs'."
        raise ValueError(msg)
    inputs = _require_string_sequence(fields["inputs"], label=f"{label}.fields.inputs", unique=True)
    outputs = _require_string_sequence(fields["outputs"], label=f"{label}.fields.outputs", unique=True)
    domain.field_sets.validate_ordered_fields(inputs, task.input_names, label=f"{label}.fields.inputs")
    domain.field_sets.validate_ordered_fields(outputs, task.output_names, label=f"{label}.fields.outputs")
    layout = _require_string_sequence(payload.get("tensor_layout"), label=f"{label}.tensor_layout", unique=True)
    if layout != task.tensor_layout:
        msg = f"{label}.tensor_layout must equal {list(task.tensor_layout)}, got {list(layout)}."
        raise ValueError(msg)


def source_file_identity(path: Path | str) -> dict[str, Any]:
    """Return basename, exact size, and streaming SHA-256 for one source file."""
    source_path = Path(path)
    hasher = hashlib.sha256()
    size = 0
    with source_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
    return {"name": source_path.name, "size_bytes": size, "sha256": hasher.hexdigest()}


def canonical_metadata_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical byte count and SHA-256 of portable JSON metadata."""
    encoded = _canonical_json(dict(value), label="Source metadata identity")
    return {"canonical_size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def compute_case_fingerprint(
    *,
    task: TaskSpec,
    case_id: str,
    source_identity: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    inputs: Tensor,
    outputs: Tensor,
) -> str:
    """Compute one in-memory scientific case fingerprint without persistence."""
    normalized_case_id = _require_non_empty_string(case_id, label="case_id")
    rank = len(task.tensor_layout) - 1
    normalized_inputs = _require_tensor(inputs, label="case inputs", rank=rank)
    normalized_outputs = _require_tensor(outputs, label="case outputs", rank=rank)
    if normalized_inputs.shape[0] != task.in_channels or normalized_outputs.shape[0] != task.out_channels:
        msg = "Case tensor channel counts do not match the task contract."
        raise ValueError(msg)
    if normalized_inputs.shape[1:] != normalized_outputs.shape[1:]:
        msg = "Case input/output spatial shapes differ."
        raise ValueError(msg)
    if normalized_inputs.dtype != normalized_outputs.dtype:
        msg = "Case input/output tensor dtypes differ."
        raise ValueError(msg)
    metadata = {
        "case_fingerprint_version": CASE_FINGERPRINT_VERSION,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "fields": {"inputs": list(task.input_names), "outputs": list(task.output_names)},
        "tensor_layout": list(task.tensor_layout[1:]),
        "case_id": normalized_case_id,
        "source_identity": _json_mapping_copy(source_identity, label="source_identity"),
        "source_metadata": _json_mapping_copy(source_metadata, label="source_metadata"),
        "tensor_metadata": {
            "inputs": _tensor_metadata(normalized_inputs),
            "outputs": _tensor_metadata(normalized_outputs),
        },
    }
    return _content_fingerprint(metadata, (("inputs", normalized_inputs), ("outputs", normalized_outputs)))


def build_training_dataset_payload(
    *,
    task: TaskSpec,
    dataset_id: str,
    sample_ids: Sequence[str],
    generated_batch_identity: Mapping[str, Any],
    source_identities: Sequence[Mapping[str, Any]],
    source_metadata: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
    case_fingerprints: Sequence[str],
    inputs: Tensor,
    outputs: Tensor,
) -> dict[str, Any]:
    """Build the only maintained final training-dataset payload."""
    normalized_dataset_id = _require_non_empty_string(dataset_id, label="dataset_id")
    normalized_sample_ids = _require_string_sequence(sample_ids, label="sample_ids", unique=True)
    normalized_case_fingerprints = _require_sha256_sequence(case_fingerprints, label="case_fingerprints")
    normalized_batch_identity = _validate_generated_batch_identity(
        generated_batch_identity,
        sample_ids=normalized_sample_ids,
        label="generated_batch_identity",
    )
    normalized_sources = [_json_mapping_copy(value, label=f"source_identities[{index}]") for index, value in enumerate(source_identities)]
    normalized_metadata = [_json_mapping_copy(value, label=f"source_metadata[{index}]") for index, value in enumerate(source_metadata)]
    normalized_provenance = _json_mapping_copy(source_provenance, label="source_provenance")
    sample_count = len(normalized_sample_ids)
    if sample_count <= 0:
        msg = "Training datasets must contain at least one sample."
        raise ValueError(msg)
    if not (len(normalized_sources) == len(normalized_metadata) == len(normalized_case_fingerprints) == sample_count):
        msg = "sample_ids, source identities, source metadata, and case fingerprints must align one-to-one."
        raise ValueError(msg)
    rank = len(task.tensor_layout)
    normalized_inputs = _require_tensor(inputs, label="inputs", rank=rank)
    normalized_outputs = _require_tensor(outputs, label="outputs", rank=rank)
    if normalized_inputs.shape[0] != sample_count or normalized_outputs.shape[0] != sample_count:
        msg = f"Dataset tensor sample counts must equal {sample_count}."
        raise ValueError(msg)
    if normalized_inputs.shape[1] != task.in_channels or normalized_outputs.shape[1] != task.out_channels:
        msg = "Dataset tensor channel counts do not match the task contract."
        raise ValueError(msg)
    if normalized_inputs.shape[2:] != normalized_outputs.shape[2:]:
        msg = "Dataset input/output spatial shapes differ."
        raise ValueError(msg)
    if normalized_inputs.dtype != normalized_outputs.dtype:
        msg = "Dataset input/output tensor dtypes differ."
        raise ValueError(msg)
    payload: dict[str, Any] = {
        "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "schema_kind": TRAINING_DATASET_SCHEMA_KIND,
        "dataset_id": normalized_dataset_id,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "fields": {"inputs": list(task.input_names), "outputs": list(task.output_names)},
        "tensor_layout": list(task.tensor_layout),
        "sample_count": sample_count,
        "spatial_shape": list(normalized_inputs.shape[2:]),
        "sample_ids": list(normalized_sample_ids),
        "generated_batch_identity": normalized_batch_identity,
        "source_identities": normalized_sources,
        "source_metadata": normalized_metadata,
        "source_provenance": normalized_provenance,
        "case_fingerprints": list(normalized_case_fingerprints),
        "tensor_metadata": {
            "inputs": _tensor_metadata(normalized_inputs),
            "outputs": _tensor_metadata(normalized_outputs),
        },
        "inputs": normalized_inputs,
        "outputs": normalized_outputs,
        "dataset_fingerprint": "",
    }
    fingerprint_metadata = {
        key: value for key, value in payload.items() if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id", "source_provenance"}
    }
    payload["dataset_fingerprint"] = _content_fingerprint(
        fingerprint_metadata,
        (("inputs", normalized_inputs), ("outputs", normalized_outputs)),
    )
    validate_training_dataset_payload(payload, task=task)
    return payload


def validate_training_dataset_payload(
    payload: Any,
    *,
    task: TaskSpec,
    verify_content: bool = False,
) -> DatasetIdentity:
    """Validate one version-1 training dataset and optionally rehash all content."""
    mapping = _require_mapping(payload, label="Training dataset")
    _require_exact_keys(mapping, label="Training dataset")
    _validate_task_header(mapping, task, label="Training dataset")
    dataset_id = _require_non_empty_string(mapping.get("dataset_id"), label="Training dataset.dataset_id")
    sample_count = _require_positive_int(mapping.get("sample_count"), label="Training dataset.sample_count")
    sample_ids = _require_string_sequence(mapping.get("sample_ids"), label="Training dataset.sample_ids", unique=True)
    if len(sample_ids) != sample_count:
        msg = f"Training dataset sample_count={sample_count} does not match {len(sample_ids)} sample_ids."
        raise ValueError(msg)
    generated_batch_identity = _validate_generated_batch_identity(
        mapping.get("generated_batch_identity"),
        sample_ids=sample_ids,
        label="Training dataset.generated_batch_identity",
    )
    source_identities = mapping.get("source_identities")
    source_metadata = mapping.get("source_metadata")
    if not isinstance(source_identities, (list, tuple)) or len(source_identities) != sample_count:
        msg = "Training dataset.source_identities must align one-to-one with sample_ids."
        raise ValueError(msg)
    if not isinstance(source_metadata, (list, tuple)) or len(source_metadata) != sample_count:
        msg = "Training dataset.source_metadata must align one-to-one with sample_ids."
        raise ValueError(msg)
    normalized_sources = [
        _json_mapping_copy(value, label=f"Training dataset.source_identities[{index}]") for index, value in enumerate(source_identities)
    ]
    normalized_metadata = [
        _json_mapping_copy(value, label=f"Training dataset.source_metadata[{index}]") for index, value in enumerate(source_metadata)
    ]
    source_provenance = _json_mapping_copy(
        mapping.get("source_provenance"),
        label="Training dataset.source_provenance",
    )
    case_fingerprints = _require_sha256_sequence(
        mapping.get("case_fingerprints"),
        label="Training dataset.case_fingerprints",
    )
    if len(case_fingerprints) != sample_count:
        msg = "Training dataset.case_fingerprints must align one-to-one with sample_ids."
        raise ValueError(msg)
    rank = len(task.tensor_layout)
    inputs = _require_tensor(mapping.get("inputs"), label="Training dataset.inputs", rank=rank)
    outputs = _require_tensor(mapping.get("outputs"), label="Training dataset.outputs", rank=rank)
    if inputs.shape[0] != sample_count or outputs.shape[0] != sample_count:
        msg = f"Training dataset tensor sample counts must equal {sample_count}."
        raise ValueError(msg)
    if inputs.shape[1] != task.in_channels or outputs.shape[1] != task.out_channels:
        msg = "Training dataset tensor channel counts do not match task fields."
        raise ValueError(msg)
    if inputs.shape[2:] != outputs.shape[2:]:
        msg = "Training dataset input/output spatial shapes differ."
        raise ValueError(msg)
    spatial_shape_value = mapping.get("spatial_shape")
    if not isinstance(spatial_shape_value, (list, tuple)) or len(spatial_shape_value) != rank - 2:
        msg = f"Training dataset.spatial_shape must contain exactly {rank - 2} dimensions."
        raise ValueError(msg)
    spatial_shape = tuple(_require_positive_int(value, label="spatial_shape") for value in spatial_shape_value)
    if tuple(inputs.shape[2:]) != spatial_shape:
        msg = f"Training dataset spatial_shape {spatial_shape} does not match tensors {tuple(inputs.shape[2:])}."
        raise ValueError(msg)
    tensor_metadata = _require_mapping(mapping.get("tensor_metadata"), label="Training dataset.tensor_metadata")
    expected_tensor_metadata = {"inputs": _tensor_metadata(inputs), "outputs": _tensor_metadata(outputs)}
    if dict(tensor_metadata) != expected_tensor_metadata:
        msg = "Training dataset tensor_metadata does not match its tensors."
        raise ValueError(msg)
    fingerprint = _require_sha256(mapping.get("dataset_fingerprint"), label="Training dataset.dataset_fingerprint")
    if verify_content:
        normalized_mapping = dict(mapping)
        normalized_mapping["generated_batch_identity"] = generated_batch_identity
        normalized_mapping["source_identities"] = normalized_sources
        normalized_mapping["source_metadata"] = normalized_metadata
        normalized_mapping["source_provenance"] = source_provenance
        fingerprint_metadata = {
            key: value
            for key, value in normalized_mapping.items()
            if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id", "source_provenance"}
        }
        expected = _content_fingerprint(fingerprint_metadata, (("inputs", inputs), ("outputs", outputs)))
        if fingerprint != expected:
            msg = f"Training dataset fingerprint mismatch for {dataset_id!r}."
            raise ValueError(msg)
    return DatasetIdentity(
        dataset_id=dataset_id,
        task=task.id,
        task_contract_digest=task.contract_digest,
        fingerprint=fingerprint,
        sample_ids=sample_ids,
        sample_count=sample_count,
        spatial_shape=spatial_shape,
        generated_batch_identity_sha256=str(generated_batch_identity["batch_manifest_identity_sha256"]),
        generated_batch_identity=generated_batch_identity,
        source_metadata=tuple(normalized_metadata),
        source_provenance=source_provenance,
    )


def membership_digest(
    *,
    role: str,
    dataset_fingerprint: str,
    sample_ids: Sequence[str],
    indices: Sequence[int],
) -> str:
    """Hash exact ordered split membership against one dataset fingerprint."""
    normalized_role = _require_non_empty_string(role, label="role")
    normalized_fingerprint = _require_sha256(dataset_fingerprint, label="dataset_fingerprint")
    normalized_sample_ids = _require_string_sequence(sample_ids, label="sample_ids", unique=True)
    normalized_indices: list[int] = []
    for position, value in enumerate(indices):
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"indices[{position}] must be an integer."
            raise TypeError(msg)
        if value < 0 or value >= len(normalized_sample_ids):
            msg = f"indices[{position}]={value} is out of bounds for {len(normalized_sample_ids)} sample_ids."
            raise IndexError(msg)
        normalized_indices.append(value)
    if len(normalized_indices) != len(set(normalized_indices)):
        msg = "indices must not contain duplicates."
        raise ValueError(msg)
    content = {
        "role": normalized_role,
        "dataset_fingerprint": normalized_fingerprint,
        "indices": normalized_indices,
        "sample_ids": [normalized_sample_ids[index] for index in normalized_indices],
    }
    return hashlib.sha256(_canonical_json(content, label="Membership payload")).hexdigest()
