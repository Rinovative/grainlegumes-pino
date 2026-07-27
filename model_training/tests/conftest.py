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
from support.synthetic_task import build_synthetic_task

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
def case_payload_factory(
    steady_task: domain.tasks.spec.TaskSpec,
) -> Callable[..., dict[str, Any]]:
    """
    Return a factory for small content-bound steady-flow case payloads.

    The factory varies identity, values, shape, dtype, and source token while
    always using production TaskSpec ordering; it does not model COMSOL files.
    """

    def factory(
        case_id: str = "case_0000",
        *,
        value: float = 0.0,
        shape: tuple[int, int] = (2, 3),
        dtype: torch.dtype = torch.float32,
        source_token: str | None = None,
    ) -> dict[str, Any]:
        """Build one strict case with deterministic per-channel constant fields."""
        input_fields = {name: torch.full(shape, value + index, dtype=dtype) for index, name in enumerate(steady_task.input_names)}
        output_fields = {name: torch.full(shape, value + 20 + index, dtype=dtype) for index, name in enumerate(steady_task.output_names)}
        return datasets.identity.build_case_payload(
            task=steady_task,
            case_id=case_id,
            input_fields=input_fields,
            output_fields=output_fields,
            source_identity={"token": source_token or case_id},
            source_metadata={"case_id": case_id},
        )

    return factory


@pytest.fixture
def merged_payload_factory(
    steady_task: domain.tasks.spec.TaskSpec,
    case_payload_factory: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """
    Return a factory for small strict merged-dataset payloads.

    Each member is validated through the production case contract before stacking,
    so callers can vary membership and dtype without bypassing identity checks.
    """

    def factory(
        dataset_id: str = "tiny",
        *,
        sample_ids: Sequence[str] = ("case_0000", "case_0001", "case_0002", "case_0003"),
        dtype: torch.dtype = torch.float32,
        source_tokens: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build one content-bound merged payload from deterministic strict cases."""
        tokens = tuple(source_tokens or sample_ids)
        cases = [
            case_payload_factory(
                sample_id,
                value=float(index),
                dtype=dtype,
                source_token=tokens[index],
            )
            for index, sample_id in enumerate(sample_ids)
        ]
        validated = [datasets.identity.validate_case_payload(case, task=steady_task) for case in cases]
        return datasets.identity.build_merged_dataset_payload(
            task=steady_task,
            dataset_id=dataset_id,
            sample_ids=sample_ids,
            source_identities=[case.source_identity for case in validated],
            source_metadata=[case.source_metadata for case in validated],
            case_fingerprints=[case.fingerprint for case in validated],
            inputs=torch.stack([case.inputs for case in validated]),
            outputs=torch.stack([case.outputs for case in validated]),
        )

    return factory
