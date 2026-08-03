# ruff: noqa: S101
"""Run bounded read-only EDA against one explicitly mounted generated batch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src import common, domain
from src.analysis.eda import eda_dataframe
from support import real_data

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not real_data.real_data_tests_enabled(),
        reason="set RUN_REAL_DATA_TESTS=1 for strict mounted generated-data acceptance",
    ),
]

_BATCH_ID = "lhs_var80_seed3001"
_MAX_CASES = 2


def test_bounded_real_generated_batch_eda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit at most two real cases without crossing into model-training data."""
    generated_root = real_data.require_real_generated_batch(_BATCH_ID)
    task = domain.tasks.registry.get_task("steady_flow")

    def reject_model_training_access(*_args: Any, **_kwargs: Any) -> None:
        message = "generated-data EDA crossed into model-training storage"
        raise AssertionError(message)

    monkeypatch.setattr(common.paths, "get_model_training_data_root", reject_model_training_access)
    monkeypatch.setattr(common.paths, "resolve_dataset_path", reject_model_training_access)
    frame, logs = eda_dataframe.generate_eda_dataframe(
        _BATCH_ID,
        task=task,
        generated_data_root=generated_root,
        max_cases=_MAX_CASES,
        show_progress=False,
    )

    assert len(frame) >= 1
    assert len(frame) <= _MAX_CASES
    assert frame.attrs["loaded_case_count"] == len(frame)
    assert frame.attrs["available_case_count"] >= len(frame)
    assert frame.attrs["task_id"] == task.id
    assert frame.attrs["field_names"] == (*task.input_names, *task.output_names, "U")
    assert frame.attrs["spatial_shape"] is not None
    for field_name in frame.attrs["field_names"]:
        assert all(np.isfinite(values).all() for values in frame[field_name])
    assert logs
    assert all(isinstance(message, str) for message in logs)
