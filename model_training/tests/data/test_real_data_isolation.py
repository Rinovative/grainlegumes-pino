# ruff: noqa: S101
"""Protect the explicit opt-in boundary for mounted production packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from support import real_data

if TYPE_CHECKING:
    from pathlib import Path


def test_real_data_acceptance_is_disabled_without_the_exact_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ordinary test sessions before resolving any production path."""
    monkeypatch.delenv(real_data.REAL_DATA_FLAG, raising=False)
    assert not real_data.real_data_tests_enabled()
    with pytest.raises(RuntimeError, match="RUN_REAL_DATA_TESTS=1"):
        real_data.require_real_data_root()


def test_enabled_real_data_acceptance_requires_an_explicit_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never fall back to the repository-local model_training/data path."""
    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    monkeypatch.delenv(real_data.MODEL_TRAINING_DATA_ROOT, raising=False)
    with pytest.raises(RuntimeError, match="MODEL_TRAINING_DATA_ROOT"):
        real_data.require_real_data_root()


def test_enabled_real_data_acceptance_makes_missing_packages_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat every requested production package as mandatory once enabled."""
    root = tmp_path / "empty-mounted-model-training-data"
    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    monkeypatch.setenv(real_data.MODEL_TRAINING_DATA_ROOT, str(root))
    with pytest.raises(FileNotFoundError, match="lhs_var80_seed3001"):
        real_data.require_real_metadata_package("lhs_var80_seed3001")


def test_enabled_real_data_acceptance_uses_the_maintained_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve exactly the supplied model-training data root."""
    root = tmp_path / "mounted-model-training-data"
    monkeypatch.setenv(real_data.REAL_DATA_FLAG, "1")
    monkeypatch.setenv(real_data.MODEL_TRAINING_DATA_ROOT, str(root))
    assert real_data.require_real_data_root() == root
