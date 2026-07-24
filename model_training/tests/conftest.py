"""Provide strict synthetic task-dataset fixtures and factories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import datasets, domain
from support.future_task import build_future_task

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@pytest.fixture
def steady_task() -> domain.tasks.spec.TaskSpec:
    """Return the authoritative steady-flow task."""
    return domain.tasks.registry.get_task("steady_flow")


@pytest.fixture
def future_task() -> domain.tasks.spec.TaskSpec:
    """Return the shared future-task contract."""
    return build_future_task()


@pytest.fixture
def case_payload_factory(
    steady_task: domain.tasks.spec.TaskSpec,
) -> Callable[..., dict[str, Any]]:
    """Return a factory for small strict case payloads."""

    def factory(
        case_id: str = "case_0000",
        *,
        value: float = 0.0,
        shape: tuple[int, int] = (2, 3),
        dtype: torch.dtype = torch.float32,
        source_token: str | None = None,
    ) -> dict[str, Any]:
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
    """Return a factory for small strict merged payloads."""

    def factory(
        dataset_id: str = "tiny",
        *,
        sample_ids: Sequence[str] = ("case_0000", "case_0001", "case_0002", "case_0003"),
        dtype: torch.dtype = torch.float32,
        source_tokens: Sequence[str] | None = None,
    ) -> dict[str, Any]:
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
