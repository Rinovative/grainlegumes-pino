"""
===============================================================================
common_serialization.py
===============================================================================
Publish authoritative files atomically and compute stable content digests.

Responsibilities:
  - Write unique sibling temporary files and atomically replace destinations
  - Flush file data and parent-directory metadata where the platform supports it
  - Remove unpublished temporary files after handled failures
  - Provide canonical JSON and file SHA-256 helpers for lifecycle identities

Design principles:
  - A failed publication never damages the previously published destination
  - Callers decide whether replacement is permitted by their lifecycle contract
  - Serialization helpers do not own experiment, checkpoint, or artifact schemas
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

PathWriter = Callable[[Path], None]


def _fsync_file(path: Path) -> None:
    """Flush one completed temporary file to stable storage where supported."""
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_parent(path: Path) -> None:
    """Flush a parent directory after replacement where supported."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Directory descriptors are not available on every supported filesystem.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Atomic replacement still holds when directory durability flushing is unsupported.
        return
    finally:
        os.close(descriptor)


def atomic_path_write(destination: Path | str, writer: PathWriter) -> Path:
    """
    Publish a file produced by ``writer`` through an atomic sibling replace.

    Parameters
    ----------
    destination : Path | str
        Final file path. Its parent is created when necessary.
    writer : Callable[[Path], None]
        Callback that must completely write the supplied temporary path.

    Returns
    -------
    Path
        The published destination.

    Raises
    ------
    RuntimeError
        If the writer removes or fails to create the temporary file.
    Exception
        Any writer, flush, or replacement failure after best-effort cleanup.

    Notes
    -----
    Replacement is atomic only when the destination filesystem supports
    ``os.replace``. The temporary file is always allocated in the same parent.

    """
    final_path = Path(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temp_path = Path(raw_temp_path)
    try:
        writer(temp_path)
        if not temp_path.is_file():
            msg = f"Atomic writer did not produce its temporary file: {temp_path}"
            raise RuntimeError(msg)  # noqa: TRY301
        _fsync_file(temp_path)
        temp_path.replace(final_path)
        _fsync_parent(final_path.parent)
    except BaseException:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise
    return final_path


def atomic_write_bytes(destination: Path | str, payload: bytes) -> Path:
    """
    Atomically publish a bytes payload.

    Parameters
    ----------
    destination : Path | str
        Final file path.
    payload : bytes
        Complete serialized payload.

    Returns
    -------
    Path
        Published destination.

    """

    def write_bytes(temp_path: Path) -> None:
        temp_path.write_bytes(payload)

    return atomic_path_write(destination, write_bytes)


def atomic_write_text(
    destination: Path | str,
    payload: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """
    Atomically publish a text payload.

    Parameters
    ----------
    destination : Path | str
        Final file path.
    payload : str
        Complete text payload.
    encoding : str, optional
        Text encoding, by default ``"utf-8"``.

    Returns
    -------
    Path
        Published destination.

    """

    def write_text(temp_path: Path) -> None:
        temp_path.write_text(payload, encoding=encoding)

    return atomic_path_write(destination, write_text)


def atomic_write_json(
    destination: Path | str,
    payload: Mapping[str, Any],
    *,
    indent: int = 2,
) -> Path:
    """
    Atomically publish a JSON object with a final newline.

    Parameters
    ----------
    destination : Path | str
        Final file path.
    payload : Mapping[str, Any]
        JSON-serializable object.
    indent : int, optional
        Pretty-print indentation, by default 2.

    Returns
    -------
    Path
        Published destination.

    """
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=True,
        indent=indent,
        sort_keys=True,
    )
    return atomic_write_text(destination, f"{serialized}\n")


def atomic_torch_save(payload: Any, destination: Path | str) -> Path:
    """
    Atomically publish one ``torch.save`` payload.

    Parameters
    ----------
    payload : Any
        PyTorch-serializable object.
    destination : Path | str
        Final checkpoint or tensor-artifact path.

    Returns
    -------
    Path
        Published destination.

    """

    def write_torch(temp_path: Path) -> None:
        with temp_path.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())

    return atomic_path_write(destination, write_torch)


def canonical_json_sha256(payload: Any) -> str:
    """
    Return a SHA-256 digest of canonical JSON-compatible content.

    Parameters
    ----------
    payload : Any
        JSON-compatible content.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    """
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """
    Return the SHA-256 digest of one file's exact bytes.

    Parameters
    ----------
    path : Path | str
        Existing file path.
    chunk_size : int, optional
        Positive streaming chunk size, by default one MiB.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    """
    if chunk_size <= 0:
        msg = f"chunk_size must be positive, got {chunk_size}."
        raise ValueError(msg)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
