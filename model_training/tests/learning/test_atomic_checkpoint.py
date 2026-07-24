# ruff: noqa: S101, EM101, TC003, TRY003
"""Verify atomic checkpoint publication and failure cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from src import common


def test_failed_atomic_save_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure after partial temporary output cannot damage the final path."""
    destination = tmp_path / "checkpoint.pt"
    common.serialization.atomic_torch_save({"epoch": 2}, destination)
    previous_bytes = destination.read_bytes()

    def fail_after_partial(payload: Any, stream: Any) -> None:
        del payload
        stream.write(b"partial-corrupt-data")
        stream.flush()
        raise OSError("injected serialization failure")

    monkeypatch.setattr(common.serialization.torch, "save", fail_after_partial)
    with pytest.raises(OSError, match="injected serialization failure"):
        common.serialization.atomic_torch_save({"epoch": 3}, destination)

    assert destination.read_bytes() == previous_bytes
    assert torch.load(destination, map_location="cpu", weights_only=False) == {"epoch": 2}
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_failed_first_publication_leaves_no_final_or_temp_file(
    tmp_path: Path,
) -> None:
    """A writer exception on first publication leaves no observable artifact."""
    destination = tmp_path / "summary.json"

    def fail_writer(temp_path: Path) -> None:
        temp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        common.serialization.atomic_path_write(destination, fail_writer)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
