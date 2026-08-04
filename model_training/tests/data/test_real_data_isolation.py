# ruff: noqa: S101
"""Protect the explicit opt-in boundary for mounted production packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from support import real_data

if TYPE_CHECKING:
    from pathlib import Path


def test_real_data_acceptance_requires_the_flag_and_both_explicit_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ordinary sessions and every enabled domain without its explicit root."""
    monkeypatch.delenv(real_data.REAL_DATA_FLAG, raising=False)
    assert not real_data.real_data_tests_enabled()
    with pytest.raises(RuntimeError, match="RUN_REAL_DATA_TESTS=1"):
        real_data.require_real_data_root()

    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    root_requirements = (
        (real_data.MODEL_TRAINING_DATA_ROOT, real_data.require_real_data_root),
        (real_data.GENERATED_DATA_ROOT, real_data.require_real_generated_data_root),
    )
    for variable, resolver in root_requirements:
        monkeypatch.delenv(variable, raising=False)
        with pytest.raises(RuntimeError, match=variable):
            resolver()


def test_enabled_real_data_acceptance_uses_only_mounted_roots_and_requires_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve both supplied roots and fail when requested packages are absent."""
    training_root = tmp_path / "mounted-model-training-data"
    generated_root = tmp_path / "mounted-generated-data"
    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    monkeypatch.setenv(real_data.MODEL_TRAINING_DATA_ROOT, str(training_root))
    monkeypatch.setenv(real_data.GENERATED_DATA_ROOT, str(generated_root))

    assert real_data.require_real_data_root() == training_root
    assert real_data.require_real_generated_data_root() == generated_root
    with pytest.raises(FileNotFoundError, match="artificial_missing_package"):
        real_data.require_real_metadata_package("artificial_missing_package")
    with pytest.raises(FileNotFoundError, match="artificial_missing_package"):
        real_data.require_real_generated_batch("artificial_missing_package")
