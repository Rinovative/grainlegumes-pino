"""
Provide reusable strict task and dataset fixtures for the CPU contract suite.

The factories model current TaskSpec field order, schema identity, and small
in-memory tensors while deliberately avoiding production storage and training
workloads. Scientific equations and lifecycle failures are exercised by their
own focused modules; these fixtures should not be treated as benchmark data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import datasets, domain
from support.synthetic_task import build_synthetic_generated_batch_identity, build_synthetic_task

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@pytest.fixture
def steady_task() -> domain.tasks.spec.TaskSpec:
    """
    Return the registered immutable steady-flow contract used by production.

    This fixture is appropriate when a test must exercise the exact seven-input,
    three-output fields; genericity tests should request ``synthetic_task`` instead.
    """
    return domain.tasks.registry.get_task("steady_flow")


@pytest.fixture
def synthetic_task() -> domain.tasks.spec.TaskSpec:
    """
    Return an unregistered task with different fields, units, and no physics.

    The fixture detects steady-flow constants leaking into generic consumers and
    must never be used as a production task or persisted default.
    """
    return build_synthetic_task()


@pytest.fixture
def training_dataset_payload_factory(
    steady_task: domain.tasks.spec.TaskSpec,
) -> Callable[..., dict[str, Any]]:
    """Return a factory for small version-1 final training datasets."""

    def factory(
        dataset_id: str = "tiny",
        *,
        sample_ids: Sequence[str] = ("case_0000", "case_0001", "case_0002", "case_0003"),
        dtype: torch.dtype = torch.float32,
        source_tokens: Sequence[str] | None = None,
        source_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tokens = tuple(source_tokens or sample_ids)
        inputs = torch.stack(
            [
                torch.stack([torch.full((2, 3), float(sample_index + channel), dtype=dtype) for channel in range(steady_task.in_channels)])
                for sample_index, _sample_id in enumerate(sample_ids)
            ]
        )
        outputs = torch.stack(
            [
                torch.stack([torch.full((2, 3), float(sample_index + 20 + channel), dtype=dtype) for channel in range(steady_task.out_channels)])
                for sample_index, _sample_id in enumerate(sample_ids)
            ]
        )
        identities = [{"case_id": sample_id, "token": tokens[index]} for index, sample_id in enumerate(sample_ids)]
        metadata = [{"case_id": sample_id, "parameters": {"sample_index": index}} for index, sample_id in enumerate(sample_ids)]
        fingerprints = [
            datasets.identity.compute_case_fingerprint(
                task=steady_task,
                case_id=sample_id,
                source_identity=identities[index],
                source_metadata=metadata[index],
                inputs=inputs[index],
                outputs=outputs[index],
            )
            for index, sample_id in enumerate(sample_ids)
        ]
        return datasets.identity.build_training_dataset_payload(
            task=steady_task,
            dataset_id=dataset_id,
            sample_ids=sample_ids,
            generated_batch_identity=build_synthetic_generated_batch_identity(
                batch_name="synthetic_source",
                sample_ids=sample_ids,
            ),
            source_identities=identities,
            source_metadata=metadata,
            source_provenance=source_provenance or {"batch_manifest_sha256": "2" * 64},
            case_fingerprints=fingerprints,
            inputs=inputs,
            outputs=outputs,
        )

    return factory
