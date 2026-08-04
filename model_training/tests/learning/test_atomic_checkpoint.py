# ruff: noqa: S101, EM101, TC003, TRY003
"""
Protect atomic checkpoint publication when serialization fails mid-write.

Injected writer failures prove an existing destination remains byte-identical and
a first publication leaves neither final nor temporary files. Checkpoint schema,
roles, and resume-state equivalence are covered by ``test_checkpoint_resume``.
this module isolates only the filesystem transaction.
"""

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
    """
    Inject serialization failure after corrupt bytes reach a temporary checkpoint.

    The previously published destination must remain byte-identical and loadable,
    and temporary state must be removed, protecting resume continuity.
    """
    destination = tmp_path / "checkpoint.pt"
    common.serialization.atomic_torch_save({"epoch": 2}, destination)
    previous_bytes = destination.read_bytes()

    def fail_after_partial(payload: Any, stream: Any) -> None:
        del payload
        stream.write(b"partial-corrupt-data")
        stream.flush()
        raise OSError("injected serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial)
    with pytest.raises(OSError, match="injected serialization failure"):
        common.serialization.atomic_torch_save({"epoch": 3}, destination)

    assert destination.read_bytes() == previous_bytes
    assert torch.load(destination, map_location="cpu", weights_only=False) == {"epoch": 2}
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_failed_first_publication_leaves_no_final_or_temp_file(
    tmp_path: Path,
) -> None:
    """
    Fail a generic atomic writer after creating its first partial temporary file.

    Neither final nor temporary paths may remain, preventing incomplete first
    publication from appearing authoritative to readers.
    """
    destination = tmp_path / "summary.json"

    def fail_writer(temp_path: Path) -> None:
        temp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        common.serialization.atomic_path_write(destination, fail_writer)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
