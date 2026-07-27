"""
===============================================================================
dataset_identity.py
===============================================================================
Define and validate current task-aware case, dataset, and split identities.

Responsibilities:
  - Build strict versioned case and merged-dataset payloads
  - Validate task fields, tensor metadata, sample identity, and source identity
  - Compute deterministic content and ordered-membership fingerprints

Design principles:
  - Saved field declarations and sample membership are explicit and order-sensitive
  - Physical paths and merged dataset names are excluded from portable fingerprints
  - Routine validation trusts stored content identity after strict structure checks
  - Explicit verification recomputes exact metadata and tensor-byte identity
  - Invalid keys, task digests, names, and schema versions fail without translation

This module does NOT:
  - Resolve paths, write payload files, or enforce producer overwrite policy
  - Define task semantics, split selection, normalizers, or DataLoader behavior
  - Upgrade historical schemas, repair fingerprints, or accept field aliases
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from src import domain

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec

CASE_SCHEMA_VERSION = 1
MERGED_DATASET_SCHEMA_VERSION = 2
SPLIT_SCHEMA_VERSION = 1
CASE_SCHEMA_KIND = "task_case"
MERGED_DATASET_SCHEMA_KIND = "merged_dataset"
_SHA256_HEX_LENGTH = 64
_TENSOR_HASH_CHUNK_BYTES = 8 * 1024 * 1024

_CASE_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "schema_kind",
        "task",
        "task_contract_digest",
        "fields",
        "tensor_layout",
        "sample_count",
        "spatial_shape",
        "sample_ids",
        "case_id",
        "source_identity",
        "source_metadata",
        "tensor_metadata",
        "input_fields",
        "output_fields",
        "dataset_fingerprint",
    }
)
_MERGED_REQUIRED_KEYS = frozenset(
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
        "source_identities",
        "source_metadata",
        "case_fingerprints",
        "tensor_metadata",
        "inputs",
        "outputs",
        "dataset_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """
    Describe the portable ordered identity of one validated dataset.

    Attributes
    ----------
    dataset_id : str
        Logical dataset identifier used for resolution.
    task : str
        Registered task identifier.
    task_contract_digest : str
        Digest of the exact task contract.
    fingerprint : str
        Producer-computed dataset content fingerprint. Strict validation
        recomputes it; routine validation checks and reuses the stored value.
    sample_ids : tuple[str, ...]
        Unique ordered sample identifiers.
    sample_count : int
        Number of ordered samples.
    spatial_shape : tuple[int, ...]
        Spatial tensor shape after batch and channel axes.

    Notes
    -----
    This frozen dataclass is the portable identity returned by validators; path,
    loader, split, and normalization state are deliberately absent. Its fields
    are not an alternative to validating a serialized payload at a public load
    boundary.

    """

    dataset_id: str
    task: str
    task_contract_digest: str
    fingerprint: str
    sample_ids: tuple[str, ...]
    sample_count: int
    spatial_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        """
        Return the persisted split-identity representation.

        Returns
        -------
        dict[str, Any]
            JSON-serializable logical and ordered dataset identity.

        """
        return {
            "dataset_id": self.dataset_id,
            "task": self.task,
            "task_contract_digest": self.task_contract_digest,
            "fingerprint": self.fingerprint,
            "sample_ids": list(self.sample_ids),
            "sample_count": self.sample_count,
            "spatial_shape": list(self.spatial_shape),
        }


@dataclass(frozen=True, slots=True)
class CaseIdentity:
    """
    Describe the portable identity and metadata of one validated case payload.

    Attributes
    ----------
    case_id : str
        Stable sample identifier bound to ``sample_ids``.
    source_identity : dict[str, Any]
        Isolated path-independent producer identity included in the fingerprint.
    source_metadata : dict[str, Any]
        Isolated reproducibility metadata retained outside the case fingerprint.
    fingerprint : str
        Stored producer-computed case content SHA-256 digest.
    spatial_shape : tuple[int, ...]
        Positive spatial dimensions shared by all case fields.

    Notes
    -----
    The dataclass is frozen, while its dictionaries remain ordinary isolated
    mappings. Callers receive it only after structural validation and optionally
    exact tensor-byte verification.

    """

    case_id: str
    source_identity: dict[str, Any]
    source_metadata: dict[str, Any]
    fingerprint: str
    spatial_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ValidatedCase:
    """
    Hold validated single-case tensors and identity metadata.

    Attributes
    ----------
    case_id : str
        Stable case identifier.
    inputs : Tensor
        Input tensor with shape ``[channel, ...spatial]``.
    outputs : Tensor
        Output tensor with shape ``[channel, ...spatial]``.
    source_identity : dict[str, Any]
        Path-independent source identity.
    source_metadata : dict[str, Any]
        Reproducibility metadata carried by the producer.
    fingerprint : str
        Stored producer-computed case content fingerprint.

    Notes
    -----
    Input and output channels are newly stacked CPU tensors in TaskSpec order;
    source dictionaries are isolated JSON-compatible copies. Strict content
    verification is controlled by the validating function, not this container.

    """

    case_id: str
    inputs: Tensor
    outputs: Tensor
    source_identity: dict[str, Any]
    source_metadata: dict[str, Any]
    fingerprint: str

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the case spatial shape without its channel axis."""
        return tuple(self.inputs.shape[1:])


@dataclass(frozen=True, slots=True)
class _ValidatedCaseParts:
    """
    Carry validated case identity and ordered field tensors before stacking.

    This internal transport avoids a channel-stack allocation when a caller
    needs identity only; materializing ``ValidatedCase`` owns the later stacks.
    """

    identity: CaseIdentity
    input_tensors: tuple[Tensor, ...]
    output_tensors: tuple[Tensor, ...]


def _canonical_json(value: Any, *, label: str) -> bytes:
    """
    Encode canonical identity metadata as deterministic UTF-8 JSON bytes.

    Keys are sorted, separators compact, non-ASCII escaped, and NaN/infinity
    rejected so equal semantic metadata produces equal digest input.

    Raises
    ------
    TypeError
        If ``value`` is not JSON-serializable without non-finite numbers.

    """
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
    """
    Append one eight-byte-length-prefixed value to an incremental hash.

    Length framing prevents concatenation ambiguity between metadata, labels,
    and tensor payloads that share the same SHA-256 stream.
    """
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _tensor_dtype(tensor: Tensor) -> str:
    """Return a stable persisted PyTorch dtype name."""
    return str(tensor.dtype).removeprefix("torch.")


def _update_raw_tensor_hash(hasher: Any, tensor: Tensor) -> None:
    """
    Stream one tensor's exact contiguous CPU bytes into an existing hash.

    The tensor is detached, moved to CPU, made contiguous only when necessary,
    viewed as bytes, and processed in bounded chunks. Shape and dtype framing
    belong to surrounding canonical metadata and group helpers.
    """
    contiguous = tensor.detach().cpu()
    if not contiguous.is_contiguous():
        contiguous = contiguous.contiguous()
    byte_array = contiguous.view(torch.uint8).numpy()
    byte_view = byte_array.data.cast("B")
    for offset in range(0, len(byte_view), _TENSOR_HASH_CHUNK_BYTES):
        hasher.update(byte_view[offset : offset + _TENSOR_HASH_CHUNK_BYTES])


def _update_tensor_hash(hasher: Any, tensor: Tensor) -> None:
    """
    Append a tensor byte count followed by its exact chunked content.

    The explicit count frames each standalone tensor without allocating one
    complete ``bytes`` object.
    """
    byte_count = tensor.numel() * tensor.element_size()
    hasher.update(byte_count.to_bytes(8, byteorder="big", signed=False))
    _update_raw_tensor_hash(hasher, tensor)


def _update_tensor_group_hash(hasher: Any, tensors: Sequence[Tensor]) -> None:
    """
    Hash ordered fields as one virtual contiguous channel stack.

    A single aggregate byte count followed by each field's raw bytes is exactly
    compatible with hashing their materialized task-order ``torch.stack`` while
    avoiding that allocation during identity-only validation.
    """
    byte_count = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    hasher.update(byte_count.to_bytes(8, byteorder="big", signed=False))
    for tensor in tensors:
        _update_raw_tensor_hash(hasher, tensor)


def _tensor_metadata(tensor: Tensor) -> dict[str, Any]:
    """Return persisted tensor dtype and shape metadata."""
    return {
        "dtype": _tensor_dtype(tensor),
        "shape": list(tensor.shape),
    }


def _require_exact_keys(payload: Mapping[str, Any], required: frozenset[str], *, label: str) -> None:
    """
    Require exact schema keys and report both missing and unexpected names.

    Additional compatibility or producer-private keys fail closed rather than
    being silently ignored during identity validation.
    """
    missing = sorted(required.difference(payload))
    unexpected = sorted(set(payload).difference(required))
    if missing or unexpected:
        msg = f"{label} schema keys do not match. Missing: {missing}; unexpected: {unexpected}."
        raise ValueError(msg)


def _require_non_empty_string(value: Any, *, label: str) -> str:
    """Return a required non-empty string."""
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string."
        raise TypeError(msg)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    """
    Parse one canonical stored SHA-256 digest.

    Admission requires exactly 64 lowercase hexadecimal characters; uppercase,
    prefixes, whitespace, and other hash formats are not normalized.
    """
    digest = _require_non_empty_string(value, label=label)
    if len(digest) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in digest):
        msg = f"{label} must be a 64-character lowercase hexadecimal SHA-256 digest."
        raise ValueError(msg)
    return digest


def _require_string_sequence(value: Any, *, label: str, unique: bool) -> tuple[str, ...]:
    """
    Parse an ordered list/tuple of non-empty strings without coercion.

    Order is preserved in the immutable return value. When ``unique`` is true,
    duplicate identifiers are reported explicitly because order-sensitive
    sample and field identity cannot admit repeated members.
    """
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
    """
    Parse an order-preserving sequence of canonical SHA-256 digests.

    Duplicate digests are not rejected by this format helper; ordered sample
    alignment and identity are validated separately by the enclosing schema.
    """
    values = _require_string_sequence(value, label=label, unique=False)
    return tuple(_require_sha256(item, label=f"{label}[{index}]") for index, item in enumerate(values))


def _require_positive_int(value: Any, *, label: str) -> int:
    """Return a required positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    if value <= 0:
        msg = f"{label} must be positive, got {value}."
        raise ValueError(msg)
    return value


def _require_spatial_shape(value: Any, *, rank: int, label: str) -> tuple[int, ...]:
    """
    Parse an exact-rank spatial shape into positive integer dimensions.

    Lists and tuples are accepted for serialized/in-memory symmetry; booleans,
    zero, negative, and dimension-count mismatches are rejected.
    """
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        msg = f"{label} must contain exactly {rank} dimensions."
        raise ValueError(msg)
    return tuple(_require_positive_int(item, label=f"{label}[{index}]") for index, item in enumerate(value))


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Return a required mapping."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _json_mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    """
    Canonicalize a mapping through JSON and return an isolated plain dictionary.

    The round trip validates nested JSON compatibility and removes references to
    producer-owned mutable containers before identity objects are exposed.
    """
    mapping = dict(_require_mapping(value, label=label))
    encoded = _canonical_json(mapping, label=label)
    return json.loads(encoded.decode("utf-8"))


def _require_tensor(value: Any, *, label: str, rank: int) -> Tensor:
    """
    Admit one dense finite real tensor and expose a detached CPU view/copy.

    Non-tensor array-like values are converted with ``torch.as_tensor``. Sparse
    layouts, wrong rank, non-floating/complex dtypes, and non-finite content fail
    before hashing or model materialization; contiguity is deferred to hashing.
    """
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.layout != torch.strided:
        msg = f"{label} must be a dense strided tensor."
        raise TypeError(msg)
    if tensor.ndim != rank:
        msg = f"{label} must have rank {rank}, got shape {tuple(tensor.shape)}."
        raise ValueError(msg)
    if not tensor.is_floating_point():
        msg = f"{label} must use a real floating-point dtype, got {tensor.dtype}."
        raise TypeError(msg)
    cpu_tensor = tensor.detach().cpu()
    if not bool(torch.isfinite(cpu_tensor).all().item()):
        msg = f"{label} must contain only finite values."
        raise ValueError(msg)
    return cpu_tensor


def _validate_task_header(payload: Mapping[str, Any], task: TaskSpec, *, schema_kind: str, schema_version: int, label: str) -> None:
    """
    Bind a payload header to one exact current TaskSpec and schema kind.

    Version, kind, task ID, complete task-contract digest, ordered input/output
    declarations, and tensor layout must all agree. No aliasing, field reordering,
    or schema compatibility translation occurs.
    """
    if payload.get("schema_version") != schema_version:
        msg = f"{label} schema_version must be the current value {schema_version}."
        raise ValueError(msg)
    if payload.get("schema_kind") != schema_kind:
        msg = f"{label} schema_kind must be {schema_kind!r}."
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
    input_fields = _require_string_sequence(fields["inputs"], label=f"{label}.fields.inputs", unique=False)
    output_fields = _require_string_sequence(fields["outputs"], label=f"{label}.fields.outputs", unique=False)
    domain.field_sets.validate_ordered_fields(input_fields, task.input_names, label=f"{label}.fields.inputs")
    domain.field_sets.validate_ordered_fields(output_fields, task.output_names, label=f"{label}.fields.outputs")

    layout = _require_string_sequence(payload.get("tensor_layout"), label=f"{label}.tensor_layout", unique=True)
    if layout != task.tensor_layout:
        msg = f"{label}.tensor_layout must equal {list(task.tensor_layout)}, got {list(layout)}."
        raise ValueError(msg)


def _validate_field_tensors(
    fields: Any,
    *,
    declarations: Sequence[str],
    expected: Sequence[str],
    spatial_rank: int,
    label: str,
) -> tuple[Tensor, ...]:
    """
    Validate and order a complete field mapping without channel stacking.

    The declaration must match TaskSpec order exactly, mapping keys must contain
    no missing or unexpected names, and every detached CPU field must share one
    spatial shape and dtype. The returned tuple follows ``expected`` order.
    """
    mapping = _require_mapping(fields, label=label)
    domain.field_sets.validate_ordered_fields(declarations, expected, label=f"{label} declaration")
    missing = [name for name in expected if name not in mapping]
    unexpected = [name for name in mapping if name not in expected]
    if missing or unexpected:
        msg = f"{label} does not match the task contract. Missing: {missing}; unexpected: {unexpected}."
        raise ValueError(msg)

    tensors = tuple(_require_tensor(mapping[name], label=f"{label}.{name}", rank=spatial_rank) for name in expected)
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        msg = f"{label} contains inconsistent spatial shapes: {sorted(shapes)}."
        raise ValueError(msg)
    dtypes = {_tensor_dtype(tensor) for tensor in tensors}
    if len(dtypes) != 1:
        msg = f"{label} contains inconsistent dtypes: {sorted(dtypes)}."
        raise ValueError(msg)
    return tensors


def _stack_fields(
    fields: Any,
    *,
    declarations: Sequence[str],
    expected: Sequence[str],
    spatial_rank: int,
    label: str,
) -> Tensor:
    """
    Validate exact field membership and materialize a task-order channel stack.

    The leading output axis corresponds one-to-one with ``expected``; field
    tensors have already been detached to CPU and checked for shared shape/dtype.
    """
    tensors = _validate_field_tensors(
        fields,
        declarations=declarations,
        expected=expected,
        spatial_rank=spatial_rank,
        label=label,
    )
    return torch.stack(tensors, dim=0)


def _content_fingerprint(metadata: Mapping[str, Any], tensors: Sequence[tuple[str, Tensor]]) -> str:
    """
    Hash framed canonical metadata and labeled exact tensor content with SHA-256.

    Tensor sequence and labels are order-sensitive. Each tensor is framed by its
    byte count; dtype and shape must therefore be present in the supplied
    metadata when those properties belong to the identity contract.
    """
    hasher = hashlib.sha256()
    _update_hash(hasher, _canonical_json(dict(metadata), label="Fingerprint metadata"))
    for label, tensor in tensors:
        _update_hash(hasher, label.encode("utf-8"))
        _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def _case_content_fingerprint(
    metadata: Mapping[str, Any],
    input_tensors: Sequence[Tensor],
    output_tensors: Sequence[Tensor],
) -> str:
    """
    Recompute a case fingerprint from virtual task-order channel stacks.

    Canonical metadata is followed by separately labeled input and output groups.
    Group hashing matches the producer's materialized stacks without allocating
    them during identity-only strict verification.
    """
    hasher = hashlib.sha256()
    _update_hash(hasher, _canonical_json(dict(metadata), label="Fingerprint metadata"))
    for label, tensors in (("inputs", input_tensors), ("outputs", output_tensors)):
        _update_hash(hasher, label.encode("utf-8"))
        _update_tensor_group_hash(hasher, tensors)
    return hasher.hexdigest()


def source_file_identity(path: Path | str) -> dict[str, Any]:
    """
    Return a path-independent content identity for one source file.

    Parameters
    ----------
    path : Path | str
        Existing source file.

    Returns
    -------
    dict[str, Any]
        Basename, exact byte count, and lowercase SHA-256 content digest. Parent
        directories are excluded, so moving an unchanged same-named source does
        not alter identity.

    Raises
    ------
    OSError
        If the source cannot be opened or read completely.

    """
    source_path = Path(path)
    hasher = hashlib.sha256()
    size = 0
    with source_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
    return {
        "name": source_path.name,
        "size_bytes": size,
        "sha256": hasher.hexdigest(),
    }


def canonical_metadata_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a path-independent digest for canonical source metadata.

    Parameters
    ----------
    value : Mapping[str, Any]
        Reproducibility metadata with producer-specific path fields removed.

    Returns
    -------
    dict[str, Any]
        Canonical UTF-8 byte count and lowercase SHA-256 digest.

    Raises
    ------
    TypeError
        If nested metadata is not JSON-compatible or contains non-finite values.

    Notes
    -----
    Callers must remove producer-specific path fields before this function; it
    canonicalizes content but does not know which metadata keys are portable.

    """
    encoded = _canonical_json(dict(value), label="Source metadata identity")
    return {
        "canonical_size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_case_payload(
    *,
    task: TaskSpec,
    case_id: str,
    input_fields: Mapping[str, Any],
    output_fields: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build one strict case payload.

    Parameters
    ----------
    task : TaskSpec
        Authoritative task contract.
    case_id : str
        Stable sample/case identifier.
    input_fields : Mapping[str, Any]
        Exact task input fields.
    output_fields : Mapping[str, Any]
        Exact task output fields.
    source_identity : Mapping[str, Any]
        Path-independent reproducibility identity.
    source_metadata : Mapping[str, Any]
        Producer metadata retained for reproducibility.

    Returns
    -------
    dict[str, Any]
        Current-schema payload with detached CPU field tensors, exact metadata,
        and a deterministic lowercase SHA-256 fingerprint.

    Raises
    ------
    TypeError
        If identifiers, source mappings, or field values violate required JSON,
        tensor, or floating-dtype types.
    ValueError
        If task fields, shapes, finiteness, declarations, or metadata violate the
        exact single-case schema.

    Notes
    -----
    The fingerprint binds schema/task identity, case/sample identity, ordered
    fields, tensor metadata, path-independent ``source_identity``, and exact
    input/output bytes. ``source_metadata`` is retained for reproducibility but
    deliberately excluded from the case content digest.

    """
    normalized_case_id = _require_non_empty_string(case_id, label="case_id")
    spatial_rank = len(task.tensor_layout) - 2
    inputs = _stack_fields(
        input_fields,
        declarations=task.input_names,
        expected=task.input_names,
        spatial_rank=spatial_rank,
        label="input_fields",
    )
    outputs = _stack_fields(
        output_fields,
        declarations=task.output_names,
        expected=task.output_names,
        spatial_rank=spatial_rank,
        label="output_fields",
    )
    if inputs.shape[1:] != outputs.shape[1:]:
        msg = f"Case input/output spatial shapes differ: {tuple(inputs.shape[1:])} != {tuple(outputs.shape[1:])}."
        raise ValueError(msg)

    normalized_source_identity = _json_mapping_copy(source_identity, label="source_identity")
    normalized_source_metadata = _json_mapping_copy(source_metadata, label="source_metadata")
    payload: dict[str, Any] = {
        "schema_version": CASE_SCHEMA_VERSION,
        "schema_kind": CASE_SCHEMA_KIND,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "fields": {
            "inputs": list(task.input_names),
            "outputs": list(task.output_names),
        },
        "tensor_layout": list(task.tensor_layout),
        "sample_count": 1,
        "spatial_shape": list(inputs.shape[1:]),
        "sample_ids": [normalized_case_id],
        "case_id": normalized_case_id,
        "source_identity": normalized_source_identity,
        "source_metadata": normalized_source_metadata,
        "tensor_metadata": {
            "inputs": _tensor_metadata(inputs.unsqueeze(0)),
            "outputs": _tensor_metadata(outputs.unsqueeze(0)),
        },
        "input_fields": {name: inputs[index] for index, name in enumerate(task.input_names)},
        "output_fields": {name: outputs[index] for index, name in enumerate(task.output_names)},
        "dataset_fingerprint": "",
    }
    excluded_fingerprint_keys = {
        "input_fields",
        "output_fields",
        "source_metadata",
        "dataset_fingerprint",
    }
    fingerprint_metadata = {key: value for key, value in payload.items() if key not in excluded_fingerprint_keys}
    payload["dataset_fingerprint"] = _content_fingerprint(
        fingerprint_metadata,
        (("inputs", inputs), ("outputs", outputs)),
    )
    validate_case_identity(payload, task=task)
    return payload


def _validate_case_parts(
    payload: Any,
    *,
    task: TaskSpec,
    verify_content: bool,
) -> _ValidatedCaseParts:
    """
    Validate one current case while deferring model channel-stack allocation.

    Exact schema/header, singleton sample identity, field tensors, spatial shape,
    tensor metadata, source mappings, and stored digest are always checked.
    ``verify_content`` additionally streams canonical identity metadata and all
    task-order tensor bytes and rejects a digest mismatch.
    """
    mapping = _require_mapping(payload, label="Case payload")
    _require_exact_keys(mapping, _CASE_REQUIRED_KEYS, label="Case payload")
    _validate_task_header(
        mapping,
        task,
        schema_kind=CASE_SCHEMA_KIND,
        schema_version=CASE_SCHEMA_VERSION,
        label="Case payload",
    )

    sample_count = _require_positive_int(mapping.get("sample_count"), label="Case payload.sample_count")
    if sample_count != 1:
        msg = f"Case payload.sample_count must be 1, got {sample_count}."
        raise ValueError(msg)
    sample_ids = _require_string_sequence(mapping.get("sample_ids"), label="Case payload.sample_ids", unique=True)
    case_id = _require_non_empty_string(mapping.get("case_id"), label="Case payload.case_id")
    if sample_ids != (case_id,):
        msg = f"Case payload.sample_ids must contain only case_id {case_id!r}."
        raise ValueError(msg)

    fields = _require_mapping(mapping["fields"], label="Case payload.fields")
    spatial_rank = len(task.tensor_layout) - 2
    input_tensors = _validate_field_tensors(
        mapping.get("input_fields"),
        declarations=_require_string_sequence(fields["inputs"], label="Case payload.fields.inputs", unique=False),
        expected=task.input_names,
        spatial_rank=spatial_rank,
        label="Case payload.input_fields",
    )
    output_tensors = _validate_field_tensors(
        mapping.get("output_fields"),
        declarations=_require_string_sequence(fields["outputs"], label="Case payload.fields.outputs", unique=False),
        expected=task.output_names,
        spatial_rank=spatial_rank,
        label="Case payload.output_fields",
    )
    input_shape = tuple(input_tensors[0].shape)
    output_shape = tuple(output_tensors[0].shape)
    if input_shape != output_shape:
        msg = f"Case payload input/output spatial shapes differ: {input_shape} != {output_shape}."
        raise ValueError(msg)
    spatial_shape = _require_spatial_shape(
        mapping.get("spatial_shape"),
        rank=spatial_rank,
        label="Case payload.spatial_shape",
    )
    if input_shape != spatial_shape:
        msg = f"Case payload spatial_shape {spatial_shape} does not match tensors {input_shape}."
        raise ValueError(msg)

    tensor_metadata = _require_mapping(mapping.get("tensor_metadata"), label="Case payload.tensor_metadata")
    expected_tensor_metadata = {
        "inputs": {
            "dtype": _tensor_dtype(input_tensors[0]),
            "shape": [1, len(input_tensors), *input_shape],
        },
        "outputs": {
            "dtype": _tensor_dtype(output_tensors[0]),
            "shape": [1, len(output_tensors), *output_shape],
        },
    }
    if dict(tensor_metadata) != expected_tensor_metadata:
        msg = f"Case payload tensor_metadata does not match its tensors. Expected {expected_tensor_metadata}, got {dict(tensor_metadata)}."
        raise ValueError(msg)

    source_identity = _json_mapping_copy(mapping.get("source_identity"), label="Case payload.source_identity")
    source_metadata = _json_mapping_copy(mapping.get("source_metadata"), label="Case payload.source_metadata")
    fingerprint = _require_sha256(mapping.get("dataset_fingerprint"), label="Case payload.dataset_fingerprint")
    if verify_content:
        fingerprint_metadata = {
            key: value for key, value in mapping.items() if key not in {"input_fields", "output_fields", "source_metadata", "dataset_fingerprint"}
        }
        expected_fingerprint = _case_content_fingerprint(
            fingerprint_metadata,
            input_tensors,
            output_tensors,
        )
        if fingerprint != expected_fingerprint:
            msg = f"Case payload dataset fingerprint mismatch for {case_id!r}."
            raise ValueError(msg)

    case_identity = CaseIdentity(
        case_id=case_id,
        source_identity=source_identity,
        source_metadata=source_metadata,
        fingerprint=fingerprint,
        spatial_shape=spatial_shape,
    )
    return _ValidatedCaseParts(
        identity=case_identity,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )


def validate_case_identity(
    payload: Any,
    *,
    task: TaskSpec,
    verify_content: bool = False,
) -> CaseIdentity:
    """
    Validate case identity without stacking or copying model tensors.

    Parameters
    ----------
    payload : Any
        Candidate serialized case payload.
    task : TaskSpec
        Authoritative task contract.
    verify_content : bool, optional
        Recompute the complete content fingerprint. Default is False.

    Returns
    -------
    CaseIdentity
        Structurally validated immutable case identity. Content is also
        reverified when ``verify_content=True``.

    Raises
    ------
    TypeError
        If container, source metadata, field tensors, or stored values have
        incompatible runtime types.
    ValueError
        If exact schema, TaskSpec binding, sample/field identity, tensor metadata,
        shapes, finiteness, or the optional content fingerprint check fails.

    Notes
    -----
    Identity-only validation does not allocate model channel stacks. With the
    default ``verify_content=False``, the stored producer digest is syntax-checked
    but tensor bytes are not rehashed.

    """
    return _validate_case_parts(
        payload,
        task=task,
        verify_content=verify_content,
    ).identity


def validate_case_payload(
    payload: Any,
    *,
    task: TaskSpec,
    verify_content: bool = False,
) -> ValidatedCase:
    """
    Validate and materialize one strict case payload.

    Routine validation reuses the immutable producer-computed fingerprint. Set
    ``verify_content=True`` to hash all tensor bytes and compare exact content.

    Parameters
    ----------
    payload : Any
        Candidate serialized case payload.
    task : TaskSpec
        Authoritative task contract.
    verify_content : bool, optional
        Recompute the complete content fingerprint. Default is False.

    Returns
    -------
    ValidatedCase
        Structurally validated, task-ordered CPU model tensors and isolated
        source mappings. Content is also reverified when
        ``verify_content=True``.

    Raises
    ------
    TypeError
        If container, source metadata, field tensors, or stored values have
        incompatible runtime types.
    ValueError
        If exact schema, TaskSpec binding, sample/field identity, tensor metadata,
        shapes, finiteness, or the optional content fingerprint check fails.

    Notes
    -----
    Materialization allocates input and output channel stacks. The default trusts
    the syntax-checked producer fingerprint after structural validation; strict
    verification rehashes all tensor bytes first.

    """
    parts = _validate_case_parts(
        payload,
        task=task,
        verify_content=verify_content,
    )
    case_identity = parts.identity
    return ValidatedCase(
        case_id=case_identity.case_id,
        inputs=torch.stack(parts.input_tensors, dim=0),
        outputs=torch.stack(parts.output_tensors, dim=0),
        source_identity=case_identity.source_identity,
        source_metadata=case_identity.source_metadata,
        fingerprint=case_identity.fingerprint,
    )


def build_merged_dataset_payload(
    *,
    task: TaskSpec,
    dataset_id: str,
    sample_ids: Sequence[str],
    source_identities: Sequence[Mapping[str, Any]],
    source_metadata: Sequence[Mapping[str, Any]],
    case_fingerprints: Sequence[str],
    inputs: Tensor,
    outputs: Tensor,
) -> dict[str, Any]:
    """
    Build one strict merged dataset payload.

    Parameters
    ----------
    task : TaskSpec
        Authoritative task contract.
    dataset_id : str
        Logical dataset identifier.
    sample_ids : Sequence[str]
        Unique ordered sample identifiers.
    source_identities : Sequence[Mapping[str, Any]]
        Ordered path-independent source identities.
    source_metadata : Sequence[Mapping[str, Any]]
        Ordered canonical producer metadata aligned with ``sample_ids``.
    case_fingerprints : Sequence[str]
        Ordered verified producer-case fingerprints.
    inputs : Tensor
        Input tensor in task layout.
    outputs : Tensor
        Output tensor in task layout.

    Returns
    -------
    dict[str, Any]
        Current-schema merged payload with detached CPU BCHW tensors and a
        deterministic lowercase SHA-256 content fingerprint.

    Raises
    ------
    TypeError
        If identifiers, source mappings, fingerprints, or tensors violate
        required runtime and JSON-compatible types.
    ValueError
        If ordered membership, source alignment, tensor rank/channel/spatial
        shape, finiteness, or exact TaskSpec declarations disagree.

    Notes
    -----
    The portable fingerprint excludes ``dataset_id`` but binds ordered sample
    IDs, task/schema metadata, source identities and metadata, case fingerprints,
    tensor metadata, and exact input/output bytes. The function returns data but
    performs no filesystem publication or overwrite check.

    """
    normalized_dataset_id = _require_non_empty_string(dataset_id, label="dataset_id")
    normalized_sample_ids = _require_string_sequence(sample_ids, label="sample_ids", unique=True)
    normalized_case_fingerprints = _require_sha256_sequence(
        case_fingerprints,
        label="case_fingerprints",
    )
    normalized_sources = [_json_mapping_copy(value, label=f"source_identities[{index}]") for index, value in enumerate(source_identities)]
    normalized_source_metadata = [_json_mapping_copy(value, label=f"source_metadata[{index}]") for index, value in enumerate(source_metadata)]
    if (
        len(normalized_sources) != len(normalized_sample_ids)
        or len(normalized_source_metadata) != len(normalized_sample_ids)
        or len(normalized_case_fingerprints) != len(normalized_sample_ids)
    ):
        msg = "sample_ids, source_identities, source_metadata, and case_fingerprints must have identical lengths."
        raise ValueError(msg)

    rank = len(task.tensor_layout)
    normalized_inputs = _require_tensor(inputs, label="inputs", rank=rank)
    normalized_outputs = _require_tensor(outputs, label="outputs", rank=rank)
    sample_count = len(normalized_sample_ids)
    if sample_count <= 0:
        msg = "Merged datasets must contain at least one sample."
        raise ValueError(msg)
    if normalized_inputs.shape[0] != sample_count or normalized_outputs.shape[0] != sample_count:
        msg = (
            f"Merged tensor sample counts must equal {sample_count}; got inputs={normalized_inputs.shape[0]}, outputs={normalized_outputs.shape[0]}."
        )
        raise ValueError(msg)
    if normalized_inputs.shape[1] != task.in_channels or normalized_outputs.shape[1] != task.out_channels:
        msg = (
            "Merged tensor channel counts do not match the task contract: "
            f"inputs={normalized_inputs.shape[1]}/{task.in_channels}, "
            f"outputs={normalized_outputs.shape[1]}/{task.out_channels}."
        )
        raise ValueError(msg)
    if normalized_inputs.shape[2:] != normalized_outputs.shape[2:]:
        msg = f"Merged input/output spatial shapes differ: {tuple(normalized_inputs.shape[2:])} != {tuple(normalized_outputs.shape[2:])}."
        raise ValueError(msg)

    payload: dict[str, Any] = {
        "schema_version": MERGED_DATASET_SCHEMA_VERSION,
        "schema_kind": MERGED_DATASET_SCHEMA_KIND,
        "dataset_id": normalized_dataset_id,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "fields": {
            "inputs": list(task.input_names),
            "outputs": list(task.output_names),
        },
        "tensor_layout": list(task.tensor_layout),
        "sample_count": sample_count,
        "spatial_shape": list(normalized_inputs.shape[2:]),
        "sample_ids": list(normalized_sample_ids),
        "source_identities": normalized_sources,
        "source_metadata": normalized_source_metadata,
        "case_fingerprints": list(normalized_case_fingerprints),
        "tensor_metadata": {
            "inputs": _tensor_metadata(normalized_inputs),
            "outputs": _tensor_metadata(normalized_outputs),
        },
        "inputs": normalized_inputs,
        "outputs": normalized_outputs,
        "dataset_fingerprint": "",
    }
    fingerprint_metadata = {key: value for key, value in payload.items() if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id"}}
    payload["dataset_fingerprint"] = _content_fingerprint(
        fingerprint_metadata,
        (("inputs", normalized_inputs), ("outputs", normalized_outputs)),
    )
    validate_merged_dataset_payload(payload, task=task)
    return payload


def validate_merged_dataset_payload(
    payload: Any,
    *,
    task: TaskSpec,
    verify_content: bool = False,
) -> DatasetIdentity:
    """
    Validate one strict merged-dataset payload.

    Routine validation reuses the immutable producer-computed fingerprint. Set
    ``verify_content=True`` to hash all tensor bytes and compare exact content.

    Parameters
    ----------
    payload : Any
        Candidate serialized merged dataset.
    task : TaskSpec
        Authoritative task contract.
    verify_content : bool, optional
        Recompute the complete content fingerprint. Default is False.

    Returns
    -------
    DatasetIdentity
        Structurally validated portable ordered dataset identity. Content is
        also reverified when ``verify_content=True``.

    Raises
    ------
    TypeError
        If containers, source metadata, fingerprints, or tensors have
        incompatible runtime types.
    ValueError
        If exact schema/TaskSpec binding, ordered membership, aligned source
        metadata, tensor geometry/metadata, or optional content verification
        fails.

    Notes
    -----
    Routine validation checks the stored digest format and complete structure but
    does not rehash tensors. Strict verification canonicalizes normalized source
    mappings and recomputes the portable metadata-and-content fingerprint.

    """
    mapping = _require_mapping(payload, label="Merged dataset")
    _require_exact_keys(mapping, _MERGED_REQUIRED_KEYS, label="Merged dataset")
    _validate_task_header(
        mapping,
        task,
        schema_kind=MERGED_DATASET_SCHEMA_KIND,
        schema_version=MERGED_DATASET_SCHEMA_VERSION,
        label="Merged dataset",
    )
    dataset_id = _require_non_empty_string(mapping.get("dataset_id"), label="Merged dataset.dataset_id")
    sample_count = _require_positive_int(mapping.get("sample_count"), label="Merged dataset.sample_count")
    sample_ids = _require_string_sequence(mapping.get("sample_ids"), label="Merged dataset.sample_ids", unique=True)
    if len(sample_ids) != sample_count:
        msg = f"Merged dataset sample_count={sample_count} does not match {len(sample_ids)} sample_ids."
        raise ValueError(msg)

    source_identities = mapping.get("source_identities")
    if not isinstance(source_identities, (list, tuple)) or len(source_identities) != sample_count:
        msg = "Merged dataset.source_identities must align one-to-one with sample_ids."
        raise ValueError(msg)
    normalized_sources = [
        _json_mapping_copy(value, label=f"Merged dataset.source_identities[{index}]") for index, value in enumerate(source_identities)
    ]
    source_metadata = mapping.get("source_metadata")
    if not isinstance(source_metadata, (list, tuple)) or len(source_metadata) != sample_count:
        msg = "Merged dataset.source_metadata must align one-to-one with sample_ids."
        raise ValueError(msg)
    normalized_source_metadata = [
        _json_mapping_copy(value, label=f"Merged dataset.source_metadata[{index}]") for index, value in enumerate(source_metadata)
    ]
    case_fingerprints = _require_sha256_sequence(
        mapping.get("case_fingerprints"),
        label="Merged dataset.case_fingerprints",
    )
    if len(case_fingerprints) != sample_count:
        msg = "Merged dataset.case_fingerprints must align one-to-one with sample_ids."
        raise ValueError(msg)

    rank = len(task.tensor_layout)
    inputs = _require_tensor(mapping.get("inputs"), label="Merged dataset.inputs", rank=rank)
    outputs = _require_tensor(mapping.get("outputs"), label="Merged dataset.outputs", rank=rank)
    if inputs.shape[0] != sample_count or outputs.shape[0] != sample_count:
        msg = f"Merged dataset tensor sample counts must equal {sample_count}; got inputs={inputs.shape[0]}, outputs={outputs.shape[0]}."
        raise ValueError(msg)
    if inputs.shape[1] != task.in_channels or outputs.shape[1] != task.out_channels:
        msg = (
            "Merged dataset tensor channel counts do not match task fields: "
            f"inputs={inputs.shape[1]}/{task.in_channels}, outputs={outputs.shape[1]}/{task.out_channels}."
        )
        raise ValueError(msg)
    if inputs.shape[2:] != outputs.shape[2:]:
        msg = f"Merged dataset input/output spatial shapes differ: {tuple(inputs.shape[2:])} != {tuple(outputs.shape[2:])}."
        raise ValueError(msg)

    spatial_shape = _require_spatial_shape(
        mapping.get("spatial_shape"),
        rank=rank - 2,
        label="Merged dataset.spatial_shape",
    )
    if tuple(inputs.shape[2:]) != spatial_shape:
        msg = f"Merged dataset spatial_shape {spatial_shape} does not match tensors {tuple(inputs.shape[2:])}."
        raise ValueError(msg)

    tensor_metadata = _require_mapping(mapping.get("tensor_metadata"), label="Merged dataset.tensor_metadata")
    expected_tensor_metadata = {
        "inputs": _tensor_metadata(inputs),
        "outputs": _tensor_metadata(outputs),
    }
    if dict(tensor_metadata) != expected_tensor_metadata:
        msg = f"Merged dataset tensor_metadata does not match its tensors. Expected {expected_tensor_metadata}, got {dict(tensor_metadata)}."
        raise ValueError(msg)

    fingerprint = _require_sha256(mapping.get("dataset_fingerprint"), label="Merged dataset.dataset_fingerprint")
    if verify_content:
        fingerprint_metadata = {key: value for key, value in mapping.items() if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id"}}
        fingerprint_metadata["source_identities"] = normalized_sources
        fingerprint_metadata["source_metadata"] = normalized_source_metadata
        expected_fingerprint = _content_fingerprint(
            fingerprint_metadata,
            (("inputs", inputs), ("outputs", outputs)),
        )
        if fingerprint != expected_fingerprint:
            msg = f"Merged dataset fingerprint mismatch for {dataset_id!r}."
            raise ValueError(msg)

    return DatasetIdentity(
        dataset_id=dataset_id,
        task=task.id,
        task_contract_digest=task.contract_digest,
        fingerprint=fingerprint,
        sample_ids=sample_ids,
        sample_count=sample_count,
        spatial_shape=spatial_shape,
    )


def membership_digest(
    *,
    role: str,
    dataset_fingerprint: str,
    sample_ids: Sequence[str],
    indices: Sequence[int],
) -> str:
    """
    Hash exact ordered split membership against one dataset fingerprint.

    Parameters
    ----------
    role : str
        Split role identifier.
    dataset_fingerprint : str
        Verified source dataset fingerprint.
    sample_ids : Sequence[str]
        Complete ordered dataset sample identity.
    indices : Sequence[int]
        Exact ordered source indices selected for the split.

    Returns
    -------
    str
        Lowercase SHA-256 digest binding split role, source fingerprint, exact
        ordered indices, and the corresponding ordered sample IDs.

    Raises
    ------
    TypeError
        If role/fingerprint/sample identifiers or indices have invalid types.
    ValueError
        If identifiers are empty/duplicated or selected indices repeat.
    IndexError
        If a selected index is negative or outside ``sample_ids``.

    Notes
    -----
    Membership order is significant. The function does not sort indices or
    inspect dataset tensors; the supplied dataset fingerprint carries content
    identity into the split digest.

    """
    normalized_role = _require_non_empty_string(role, label="role")
    normalized_fingerprint = _require_non_empty_string(dataset_fingerprint, label="dataset_fingerprint")
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
    payload = {
        "role": normalized_role,
        "dataset_fingerprint": normalized_fingerprint,
        "indices": normalized_indices,
        "sample_ids": [normalized_sample_ids[index] for index in normalized_indices],
    }
    return hashlib.sha256(_canonical_json(payload, label="Membership payload")).hexdigest()


def case_collection_identity(
    *,
    task: TaskSpec,
    dataset_id: str,
    cases: Sequence[CaseIdentity | ValidatedCase],
) -> DatasetIdentity:
    """
    Build an ordered directory identity from validated case fingerprints.

    Parameters
    ----------
    task : TaskSpec
        Authoritative task contract.
    dataset_id : str
        Logical identity for the case collection.
    cases : Sequence[CaseIdentity | ValidatedCase]
        Validated ordered case identities or materialized cases.

    Returns
    -------
    DatasetIdentity
        Portable identity for the ordered case collection, derived without
        loading or stacking case tensors.

    Raises
    ------
    TypeError
        If ``dataset_id`` is not a non-empty string.
    ValueError
        If the collection is empty, sample IDs repeat, or spatial shapes differ.

    Notes
    -----
    The collection fingerprint binds task/schema metadata, task field order,
    ordered case IDs and case fingerprints, and the shared spatial shape. Source
    paths and materialized tensor bytes are represented transitively by each
    already-validated case fingerprint.

    """
    normalized_dataset_id = _require_non_empty_string(dataset_id, label="dataset_id")
    if not cases:
        msg = "Case collection must contain at least one case."
        raise ValueError(msg)
    sample_ids = tuple(case.case_id for case in cases)
    if len(sample_ids) != len(set(sample_ids)):
        msg = f"Case collection contains duplicate sample ids: {sample_ids}."
        raise ValueError(msg)
    spatial_shapes = {case.spatial_shape for case in cases}
    if len(spatial_shapes) != 1:
        msg = f"Case collection contains inconsistent spatial shapes: {sorted(spatial_shapes)}."
        raise ValueError(msg)
    payload = {
        "schema_version": CASE_SCHEMA_VERSION,
        "task": task.id,
        "task_contract_digest": task.contract_digest,
        "fields": {
            "inputs": list(task.input_names),
            "outputs": list(task.output_names),
        },
        "tensor_layout": list(task.tensor_layout),
        "sample_ids": list(sample_ids),
        "case_fingerprints": [case.fingerprint for case in cases],
        "spatial_shape": list(next(iter(spatial_shapes))),
    }
    fingerprint = hashlib.sha256(_canonical_json(payload, label="Case collection identity")).hexdigest()
    return DatasetIdentity(
        dataset_id=normalized_dataset_id,
        task=task.id,
        task_contract_digest=task.contract_digest,
        fingerprint=fingerprint,
        sample_ids=sample_ids,
        sample_count=len(sample_ids),
        spatial_shape=next(iter(spatial_shapes)),
    )
