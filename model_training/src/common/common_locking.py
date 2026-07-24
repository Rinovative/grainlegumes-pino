"""
===============================================================================
common_locking.py
===============================================================================
Coordinate exclusive filesystem writers with process-scoped advisory locks.

Responsibilities:
  - Acquire re-entrant exclusive locks on persistent lock files
  - Support blocking serialization and fail-fast writer leases
  - Release locks automatically on exceptions and process exit
  - Close inherited lock descriptors in forked child processes

Design principles:
  - Lock files live outside any target directory a caller may delete
  - Kernel-owned locks, rather than file existence, define lease ownership
  - Re-entrancy is limited to the owning process thread and exact lock path
===============================================================================
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class FileLockUnavailableError(RuntimeError):
    """Raised when a nonblocking exclusive file lock is already held."""


@dataclass
class _HeldLock:
    """One thread-owned advisory lock and its re-entrant depth."""

    descriptor: int
    depth: int = 1


class _ThreadLockState(threading.local):
    """Per-thread re-entrant lock ownership."""

    def __init__(self) -> None:
        self.locks: dict[Path, _HeldLock] = {}


_THREAD_LOCK_STATE = _ThreadLockState()
_OPEN_LOCK_DESCRIPTORS: set[int] = set()
_OPEN_LOCK_DESCRIPTORS_GUARD = threading.Lock()


def _before_fork() -> None:
    """Freeze descriptor registration before the process descriptor table forks."""
    _OPEN_LOCK_DESCRIPTORS_GUARD.acquire()


def _after_fork_parent() -> None:
    """Resume descriptor registration in the parent after a fork."""
    _OPEN_LOCK_DESCRIPTORS_GUARD.release()


def _after_fork_child() -> None:
    """Drop every parent-owned lock descriptor in the forked child."""
    descriptors = tuple(_OPEN_LOCK_DESCRIPTORS)
    _OPEN_LOCK_DESCRIPTORS.clear()
    for descriptor in descriptors:
        with suppress(OSError):
            os.close(descriptor)
    _THREAD_LOCK_STATE.locks.clear()
    _OPEN_LOCK_DESCRIPTORS_GUARD.release()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)


def _open_lock_file(path: Path) -> int:
    """Open one regular lock file without following a final-component symlink."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        msg = f"Lock path is not a regular file: {path}"
        raise OSError(msg)
    return descriptor


@contextmanager
def exclusive_file_lock(
    path: Path | str,
    *,
    blocking: bool,
) -> Iterator[Path]:
    """
    Hold an exclusive advisory lock for the duration of one context.

    Parameters
    ----------
    path : Path | str
        Persistent lock-file path. Its parent directory is created.
    blocking : bool
        Wait for the current owner when true; fail immediately when false.

    Yields
    ------
    pathlib.Path
        Canonical lock-file path.

    Raises
    ------
    FileLockUnavailableError
        If a nonblocking lock is held by another process or thread.

    """
    lock_path = Path(path).expanduser().resolve(strict=False)
    held = _THREAD_LOCK_STATE.locks.get(lock_path)
    if held is not None:
        held.depth += 1
        try:
            yield lock_path
        finally:
            held.depth -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _OPEN_LOCK_DESCRIPTORS_GUARD:
        descriptor = _open_lock_file(lock_path)
        _OPEN_LOCK_DESCRIPTORS.add(descriptor)
    operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, operation)
    except OSError as error:
        with _OPEN_LOCK_DESCRIPTORS_GUARD:
            _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
            os.close(descriptor)
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            msg = f"Exclusive file lock is already held: {lock_path}"
            raise FileLockUnavailableError(msg) from error
        raise

    held = _HeldLock(descriptor=descriptor)
    _THREAD_LOCK_STATE.locks[lock_path] = held
    try:
        yield lock_path
    finally:
        del _THREAD_LOCK_STATE.locks[lock_path]
        with _OPEN_LOCK_DESCRIPTORS_GUARD:
            _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
