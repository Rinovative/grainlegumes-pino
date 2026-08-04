# ruff: noqa: S101
"""
Protect deterministic, content-bound dataset and saved-membership identity.

The tests vary case order, tensors, metadata, sample IDs, and split indices to
show that equivalent content is stable and tampering fails strict verification.
Direct-builder transactions belong to ``test_dataset_contract``. Large
production tensors are deliberately not loaded.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import datasets, domain
from support.synthetic_task import build_synthetic_generated_batch_identity

_EXPECTED_DISTINCT_FINGERPRINTS = 4
_SHA256_HEX_LENGTH = 64

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _reordered_payload(
    payload: dict[str, Any],
    *,
    order: list[int],
    task: domain.tasks.spec.TaskSpec,
) -> dict[str, Any]:
    """
    Rebuild a payload after applying one order to every sample-aligned component.

    Tensors, source evidence, metadata, and case fingerprints stay paired while
    generated-batch identity follows the reordered manifest membership.
    """
    return datasets.identity.build_training_dataset_payload(
        task=task,
        dataset_id=payload["dataset_id"],
        sample_ids=[payload["sample_ids"][index] for index in order],
        generated_batch_identity=build_synthetic_generated_batch_identity(
            batch_name=payload["generated_batch_identity"]["batch_name"],
            sample_ids=[payload["sample_ids"][index] for index in order],
        ),
        source_identities=[payload["source_identities"][index] for index in order],
        source_metadata=[payload["source_metadata"][index] for index in order],
        source_provenance=payload["source_provenance"],
        case_fingerprints=[payload["case_fingerprints"][index] for index in order],
        inputs=payload["inputs"][order],
        outputs=payload["outputs"][order],
    )


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """
    Save one strict payload below its logical dataset directory.

    This test helper intentionally uses direct ``torch.save`` rather than the
    production atomic publisher because callers exercise read-time identity only.
    """
    dataset_id = payload["dataset_id"]
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.pt"
    torch.save(payload, path)
    return path


def test_creation_computes_stable_content_identity(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Build identical strict final content twice, then verify one payload by content.

    Both constructions and strict validation must report the same fingerprint, proving
    deterministic identity for reproducible split and cache admission.
    """
    first = training_dataset_payload_factory()
    second = training_dataset_payload_factory()

    strict_identity = datasets.identity.validate_training_dataset_payload(
        first,
        task=steady_task,
        verify_content=True,
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert strict_identity.fingerprint == first["dataset_fingerprint"]


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_dataset_and_generated_identity_require_integer_version_one(
    schema_version: object,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reject alternate representations in both persisted dataset version fields."""
    payload = training_dataset_payload_factory()
    for path in (("schema_version",), ("generated_batch_identity", "schema_version")):
        invalid = copy.deepcopy(payload)
        target = invalid
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = schema_version
        with pytest.raises(ValueError, match="schema_version"):
            datasets.identity.validate_training_dataset_payload(invalid, task=steady_task)


def test_dataset_reader_requires_actual_float32_tensors(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reject array coercion and non-float32 tensors at the persisted boundary."""
    array_payload = training_dataset_payload_factory()
    array_payload["inputs"] = array_payload["inputs"].numpy()
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        datasets.identity.validate_training_dataset_payload(array_payload, task=steady_task)

    with pytest.raises(TypeError, match=r"torch\.float32"):
        training_dataset_payload_factory(dtype=torch.float64)


def test_source_metadata_is_aligned_and_fingerprint_bound(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Change and misalign ordered source metadata around an otherwise fixed payload.

    Rebuilding must change the fingerprint, while post-build tampering or length
    drift must be rejected so provenance cannot detach from sample membership.
    """
    original = training_dataset_payload_factory()
    changed_metadata = copy.deepcopy(original["source_metadata"])
    changed_metadata[0]["case_id"] = "changed_case"
    changed = datasets.identity.build_training_dataset_payload(
        task=steady_task,
        dataset_id=original["dataset_id"],
        sample_ids=original["sample_ids"],
        generated_batch_identity=original["generated_batch_identity"],
        source_identities=original["source_identities"],
        source_metadata=changed_metadata,
        source_provenance=original["source_provenance"],
        case_fingerprints=original["case_fingerprints"],
        inputs=original["inputs"],
        outputs=original["outputs"],
    )

    assert original["schema_version"] == datasets.identity.TRAINING_DATASET_SCHEMA_VERSION
    assert original["source_metadata"][0]["case_id"] == "case_0000"
    assert changed["dataset_fingerprint"] != original["dataset_fingerprint"]

    tampered = copy.deepcopy(original)
    tampered["source_metadata"][0]["case_id"] = "tampered_case"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.identity.validate_training_dataset_payload(
            tampered,
            task=steady_task,
            verify_content=True,
        )

    misaligned = copy.deepcopy(original)
    misaligned["source_metadata"].pop()
    with pytest.raises(ValueError, match="source_metadata must align"):
        datasets.identity.validate_training_dataset_payload(
            misaligned,
            task=steady_task,
        )


def test_ordered_membership_changes_fingerprint(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Vary case order, membership, and source identity independently.

    Every family must produce a distinct fingerprint because each changes the
    scientific dataset consumed by a saved run.
    """
    original = training_dataset_payload_factory()
    reordered = _reordered_payload(
        original,
        order=[1, 0, 2, 3],
        task=steady_task,
    )
    missing = datasets.identity.build_training_dataset_payload(
        task=steady_task,
        dataset_id="tiny",
        sample_ids=original["sample_ids"][:-1],
        generated_batch_identity=build_synthetic_generated_batch_identity(
            batch_name=original["generated_batch_identity"]["batch_name"],
            sample_ids=original["sample_ids"][:-1],
        ),
        source_identities=original["source_identities"][:-1],
        source_metadata=original["source_metadata"][:-1],
        source_provenance=original["source_provenance"],
        case_fingerprints=original["case_fingerprints"][:-1],
        inputs=original["inputs"][:-1],
        outputs=original["outputs"][:-1],
    )
    changed = training_dataset_payload_factory(
        source_tokens=("case_0000", "replacement", "case_0002", "case_0003"),
    )

    fingerprints = {
        original["dataset_fingerprint"],
        reordered["dataset_fingerprint"],
        missing["dataset_fingerprint"],
        changed["dataset_fingerprint"],
    }
    assert len(fingerprints) == _EXPECTED_DISTINCT_FINGERPRINTS


def test_strict_verification_rejects_reordered_samples_with_stale_fingerprint(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Swap persisted sample IDs while retaining the original stored fingerprint.

    Strict validation must reject the stale digest so ordered membership cannot
    be altered without invalidating downstream split identity.
    """
    payload = copy.deepcopy(training_dataset_payload_factory())
    payload["sample_ids"][0], payload["sample_ids"][1] = (
        payload["sample_ids"][1],
        payload["sample_ids"][0],
    )
    payload["generated_batch_identity"] = build_synthetic_generated_batch_identity(
        batch_name=payload["generated_batch_identity"]["batch_name"],
        sample_ids=payload["sample_ids"],
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.identity.validate_training_dataset_payload(
            payload,
            task=steady_task,
            verify_content=True,
        )


def test_default_dataset_load_rejects_modified_tensor_content(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Mutate saved tensor content without recomputing its persisted fingerprint.

    The default dataset loader must reject the mismatch, proving ordinary
    consumers do not silently bypass content verification.
    """
    payload = copy.deepcopy(training_dataset_payload_factory())
    payload["inputs"][0, 0, 0, 0] += 1.0
    path = tmp_path / "modified.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.simulation.create_task_dataset(path, task=steady_task)


def test_duplicate_sample_id_is_rejected(
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Duplicate one sample identifier in an otherwise current final payload.

    Validation must reject the collision because ordered membership digests rely
    on each logical sample having a unique identity.
    """
    payload = copy.deepcopy(training_dataset_payload_factory())
    payload["sample_ids"][1] = payload["sample_ids"][0]
    with pytest.raises(ValueError, match="duplicate identifiers"):
        datasets.identity.validate_training_dataset_payload(payload, task=steady_task)


def test_membership_digest_binds_indices_and_order(
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Hash fixed dataset membership while varying selected order and split role.

    Both variations must change the digest, protecting exact saved membership
    rather than only the unordered set of selected samples.
    """
    payload = training_dataset_payload_factory()
    direct = datasets.identity.membership_digest(
        role="train",
        dataset_fingerprint=payload["dataset_fingerprint"],
        sample_ids=payload["sample_ids"],
        indices=[0, 2],
    )
    reordered = datasets.identity.membership_digest(
        role="train",
        dataset_fingerprint=payload["dataset_fingerprint"],
        sample_ids=payload["sample_ids"],
        indices=[2, 0],
    )
    changed_role = datasets.identity.membership_digest(
        role="eval",
        dataset_fingerprint=payload["dataset_fingerprint"],
        sample_ids=payload["sample_ids"],
        indices=[0, 2],
    )

    assert len(direct) == _SHA256_HEX_LENGTH
    assert direct != reordered
    assert direct != changed_role


def test_saved_split_rejects_replaced_same_name_count_dataset(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Replace a saved training dataset with equal-name, equal-count new content.

    Reusing the original split must fail by fingerprint so path and cardinality
    cannot masquerade as the dataset identity used for training.
    """
    train_payload = training_dataset_payload_factory("train")
    ood_payload = training_dataset_payload_factory("ood")
    train_path = _save_dataset(tmp_path, train_payload)
    ood_path = _save_dataset(tmp_path, ood_payload)
    loader_args = {
        "dataset_factory": datasets.simulation.create_task_dataset,
        "path_train": str(train_path),
        "path_test_ood": str(ood_path),
        "task": steady_task,
        "train_dataset_id": "train",
        "ood_dataset_id": "ood",
        "batch_size": 1,
        "train_ratio": 0.5,
        "ood_fraction": 0.5,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "split_seed": 9,
    }
    *_, split_info = datasets.base.create_dataloaders(**loader_args)
    assert split_info["task_contract_digest"] == steady_task.contract_digest
    assert split_info["metadata"]["datasets"]["train"]["data_contract_digest"] == steady_task.data_contract_digest
    assert split_info["metadata"]["datasets"]["ood"]["data_contract_digest"] == steady_task.data_contract_digest
    stale_header = copy.deepcopy(split_info)
    stale_header["task_contract_digest"] = "8cdaf4de22d945e08783f118d5fa8374e37521f91b20b12c913230ba015ca91a"
    with pytest.raises(ValueError, match="current registered task"):
        datasets.base.validate_split_info(stale_header)

    replaced = training_dataset_payload_factory(
        "train",
        source_tokens=("same", "same", "replacement", "same"),
    )
    torch.save(replaced, train_path)
    with pytest.raises(ValueError, match="identity does not match"):
        datasets.base.create_dataloaders(
            **loader_args,
            split_indices=split_info,
        )


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_saved_split_requires_integer_version_one(
    schema_version: object,
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reject non-integer and unsupported saved split schema versions."""
    train_path = _save_dataset(tmp_path, training_dataset_payload_factory("split_train"))
    ood_path = _save_dataset(tmp_path, training_dataset_payload_factory("split_ood"))
    loader_args = {
        "dataset_factory": datasets.simulation.create_task_dataset,
        "path_train": str(train_path),
        "path_test_ood": str(ood_path),
        "task": steady_task,
        "train_dataset_id": "split_train",
        "ood_dataset_id": "split_ood",
        "batch_size": 1,
        "train_ratio": 0.5,
        "ood_fraction": 0.5,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "split_seed": 9,
    }
    *_, split_info = datasets.base.create_dataloaders(**loader_args)
    invalid = copy.deepcopy(split_info)
    invalid["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        datasets.base.validate_split_info(invalid)


def _valid_normalizer_state() -> dict[str, torch.Tensor]:
    """Return one strict current 2D saved-normalizer state."""
    return {
        "in_normalizer.mean": torch.zeros(1, 2, 1, 1),
        "in_normalizer.std": torch.ones(1, 2, 1, 1),
        "out_normalizer.mean": torch.zeros(1, 3, 1, 1),
        "out_normalizer.std": torch.ones(1, 3, 1, 1),
    }


@pytest.mark.parametrize(
    ("key", "replacement", "error_type", "match"),
    [
        ("out_normalizer.std", -torch.ones(1, 3, 1, 1), ValueError, "non-negative"),
        ("out_normalizer.mean", torch.full((1, 3, 1, 1), float("inf")), ValueError, "non-finite"),
        ("in_normalizer.mean", torch.zeros(1, 2, 1, dtype=torch.complex64), TypeError, "real floating-point"),
        ("in_normalizer.mean", torch.zeros(1, 2, 1), ValueError, "must have shape"),
    ],
    ids=("negative-std", "non-finite", "complex", "wrong-rank"),
)
def test_saved_normalizer_state_fails_closed(
    key: str,
    replacement: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    """
    Vary one saved statistic across negative, non-finite, complex, and wrong-rank forms.

    Each parameter family must fail with its type/domain-specific error while
    valid BCHW keys remain fixed, protecting preprocessing reconstruction.
    """
    state = _valid_normalizer_state()
    state[key] = replacement

    with pytest.raises(error_type, match=match):
        datasets.base.data_processor_from_state(state)


def test_zero_variance_normalizer_uses_a_positive_denominator_floor() -> None:
    """
    Restore zero standard deviations and normalize constant BCHW tensors.

    Results must remain finite and zero through a positive denominator floor,
    protecting legitimate constant channels without falsifying saved statistics.
    """
    state = _valid_normalizer_state()
    state["in_normalizer.std"][0, 0] = 0.0
    state["out_normalizer.std"].zero_()

    processor = datasets.base.data_processor_from_state(state)
    in_normalizer = processor.in_normalizer
    out_normalizer = processor.out_normalizer
    assert in_normalizer is not None
    assert out_normalizer is not None
    inputs = torch.zeros(2, 2, 3, 4)
    outputs = torch.zeros(2, 3, 3, 4)
    normalized_inputs = in_normalizer.transform(inputs)
    normalized_outputs = out_normalizer.transform(outputs)

    assert in_normalizer.eps > 0.0
    assert out_normalizer.eps > 0.0
    assert in_normalizer.std[0, 0, 0, 0] == 0.0
    assert torch.isfinite(normalized_inputs).all()
    assert torch.isfinite(normalized_outputs).all()
    assert torch.equal(normalized_inputs, torch.zeros_like(normalized_inputs))
    assert torch.equal(normalized_outputs, torch.zeros_like(normalized_outputs))


def test_training_loader_retains_a_partial_batch(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """
    Request a batch larger than the small valid training split.

    The loader must retain one partial batch. Dropping it would turn a valid
    dataset into an empty training lifecycle.
    """
    train_path = _save_dataset(tmp_path, training_dataset_payload_factory("partial_train"))
    ood_path = _save_dataset(tmp_path, training_dataset_payload_factory("partial_ood"))

    train_loader, *_rest = datasets.base.create_dataloaders(
        dataset_factory=datasets.simulation.create_task_dataset,
        path_train=str(train_path),
        path_test_ood=str(ood_path),
        task=steady_task,
        train_dataset_id="partial_train",
        ood_dataset_id="partial_ood",
        batch_size=8,
        train_ratio=0.5,
        ood_fraction=0.5,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        split_seed=13,
    )

    batch = next(iter(train_loader))
    assert 0 < batch["x"].shape[0] < train_loader.batch_size
    assert len(train_loader) == 1


def test_operational_provenance_is_excluded_from_scientific_fingerprint(
    training_dataset_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Changing paths/timestamps/exact operational hashes must not change science."""
    first = training_dataset_payload_factory(
        source_provenance={
            "batch_manifest_sha256": "2" * 64,
            "diagnostic_path": r"C:\\Users\\first\\OneDrive\\batch",
            "conversion_timestamp": "2026-01-01T00:00:00Z",
        },
    )
    second = training_dataset_payload_factory(
        source_provenance={
            "batch_manifest_sha256": "3" * 64,
            "diagnostic_path": "/cluster/other/batch",
            "conversion_timestamp": "2030-02-03T04:05:06Z",
        },
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    datasets.identity.validate_training_dataset_payload(second, task=domain.tasks.registry.get_task("steady_flow"), verify_content=True)
