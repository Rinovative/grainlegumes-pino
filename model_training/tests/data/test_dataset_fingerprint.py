# ruff: noqa: S101
"""Verify deterministic, tamper-evident dataset and split-membership identity."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import datasets, domain

_EXPECTED_DISTINCT_FINGERPRINTS = 5
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
    """Rebuild a payload with exact source samples in a different order."""
    return datasets.identity.build_merged_dataset_payload(
        task=task,
        dataset_id=payload["dataset_id"],
        sample_ids=[payload["sample_ids"][index] for index in order],
        source_identities=[payload["source_identities"][index] for index in order],
        source_metadata=[payload["source_metadata"][index] for index in order],
        case_fingerprints=[payload["case_fingerprints"][index] for index in order],
        inputs=payload["inputs"][order],
        outputs=payload["outputs"][order],
    )


def _save_dataset(root: Path, payload: dict[str, Any]) -> Path:
    """Save one strict payload under its logical dataset id."""
    dataset_id = payload["dataset_id"]
    directory = root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset_id}.pt"
    torch.save(payload, path)
    return path


def test_creation_computes_stable_content_identity(
    steady_task: domain.tasks.spec.TaskSpec,
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Equivalent creation inputs produce one strict-verifiable identity."""
    first = merged_payload_factory()
    second = merged_payload_factory()

    strict_identity = datasets.identity.validate_merged_dataset_payload(
        first,
        task=steady_task,
        verify_content=True,
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert strict_identity.fingerprint == first["dataset_fingerprint"]


def test_source_metadata_is_aligned_and_fingerprint_bound(
    steady_task: domain.tasks.spec.TaskSpec,
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Ordered source metadata survives schema validation and participates in identity."""
    original = merged_payload_factory()
    changed_metadata = copy.deepcopy(original["source_metadata"])
    changed_metadata[0]["case_id"] = "changed_case"
    changed = datasets.identity.build_merged_dataset_payload(
        task=steady_task,
        dataset_id=original["dataset_id"],
        sample_ids=original["sample_ids"],
        source_identities=original["source_identities"],
        source_metadata=changed_metadata,
        case_fingerprints=original["case_fingerprints"],
        inputs=original["inputs"],
        outputs=original["outputs"],
    )

    assert original["schema_version"] == datasets.identity.MERGED_DATASET_SCHEMA_VERSION
    assert original["source_metadata"][0] == {"case_id": "case_0000"}
    assert changed["dataset_fingerprint"] != original["dataset_fingerprint"]

    tampered = copy.deepcopy(original)
    tampered["source_metadata"][0]["case_id"] = "tampered_case"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.identity.validate_merged_dataset_payload(
            tampered,
            task=steady_task,
            verify_content=True,
        )

    misaligned = copy.deepcopy(original)
    misaligned["source_metadata"].pop()
    with pytest.raises(ValueError, match="source_metadata must align"):
        datasets.identity.validate_merged_dataset_payload(
            misaligned,
            task=steady_task,
        )


def test_ordered_membership_changes_fingerprint(
    steady_task: domain.tasks.spec.TaskSpec,
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Reordered, missing, changed, or replaced cases change identity."""
    original = merged_payload_factory()
    reordered = _reordered_payload(
        original,
        order=[1, 0, 2, 3],
        task=steady_task,
    )
    missing = datasets.identity.build_merged_dataset_payload(
        task=steady_task,
        dataset_id="tiny",
        sample_ids=original["sample_ids"][:-1],
        source_identities=original["source_identities"][:-1],
        source_metadata=original["source_metadata"][:-1],
        case_fingerprints=original["case_fingerprints"][:-1],
        inputs=original["inputs"][:-1],
        outputs=original["outputs"][:-1],
    )
    changed = merged_payload_factory(
        source_tokens=("case_0000", "replacement", "case_0002", "case_0003"),
    )
    different_dtype = merged_payload_factory(dtype=torch.float64)

    fingerprints = {
        original["dataset_fingerprint"],
        reordered["dataset_fingerprint"],
        missing["dataset_fingerprint"],
        changed["dataset_fingerprint"],
        different_dtype["dataset_fingerprint"],
    }
    assert len(fingerprints) == _EXPECTED_DISTINCT_FINGERPRINTS


def test_strict_verification_rejects_reordered_samples_with_stale_fingerprint(
    steady_task: domain.tasks.spec.TaskSpec,
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Strict verification rejects changed sample order with a stale fingerprint."""
    payload = copy.deepcopy(merged_payload_factory())
    payload["sample_ids"][0], payload["sample_ids"][1] = (
        payload["sample_ids"][1],
        payload["sample_ids"][0],
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.identity.validate_merged_dataset_payload(
            payload,
            task=steady_task,
            verify_content=True,
        )


def test_default_dataset_load_rejects_modified_tensor_content(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Maintained consumers reject tensor changes hidden behind stored identity."""
    payload = copy.deepcopy(merged_payload_factory())
    payload["inputs"][0, 0, 0, 0] += 1.0
    path = tmp_path / "modified.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        datasets.simulation.create_task_dataset(path, task=steady_task)


def test_duplicate_sample_id_is_rejected(
    steady_task: domain.tasks.spec.TaskSpec,
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Ordered sample identity is unique by schema."""
    payload = copy.deepcopy(merged_payload_factory())
    payload["sample_ids"][1] = payload["sample_ids"][0]
    with pytest.raises(ValueError, match="duplicate identifiers"):
        datasets.identity.validate_merged_dataset_payload(payload, task=steady_task)


def test_membership_digest_binds_indices_and_order(
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Exact indices and selected sample order are part of split identity."""
    payload = merged_payload_factory()
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
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """A same-path, same-name, same-count replacement fails by fingerprint."""
    train_payload = merged_payload_factory("train")
    ood_payload = merged_payload_factory("ood")
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

    replaced = merged_payload_factory(
        "train",
        source_tokens=("same", "same", "replacement", "same"),
    )
    torch.save(replaced, train_path)
    with pytest.raises(ValueError, match="identity does not match"):
        datasets.base.create_dataloaders(
            **loader_args,
            split_indices=split_info,
        )


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
    """Restored normalizers require finite real BCHW statistics and non-negative std."""
    state = _valid_normalizer_state()
    state[key] = replacement

    with pytest.raises(error_type, match=match):
        datasets.base.data_processor_from_state(state)


def test_zero_variance_normalizer_uses_a_positive_denominator_floor() -> None:
    """Constant channels normalize finitely through the explicit epsilon floor."""
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
    merged_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """A valid train split smaller than batch_size still yields one batch."""
    train_path = _save_dataset(tmp_path, merged_payload_factory("partial_train"))
    ood_path = _save_dataset(tmp_path, merged_payload_factory("partial_ood"))

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
