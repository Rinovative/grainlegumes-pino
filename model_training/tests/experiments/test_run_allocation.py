# ruff: noqa: BLE001, S101, EM101, PLR2004, TC003, TRY003
"""Verify collision-safe run allocation, resume, and lifecycle status."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from pathlib import Path
from typing import Any

import pytest
from src import common, experiments


def _objective(identifier: str) -> dict[str, Any]:
    """Return one complete synthetic resolved objective."""
    return {
        "id": identifier,
        "kind": identifier,
        "space": "normalized",
        "fields": ["value"],
        "reduction": "element_mean",
        "direction": "minimize",
    }


def _minimal_config(output_root: Path) -> dict[str, Any]:
    """Return the minimum fresh-run metadata used by allocation tests."""
    return {
        "task": "synthetic_task",
        "run": {"name": "run", "seed": 7, "deterministic": True, "device": "cpu"},
        "evaluation": {"objective": _objective("objective")},
        "paths": {"output_root": str(output_root)},
    }


def test_fresh_collision_rejects_before_any_run_write(tmp_path: Path) -> None:
    """An existing run leaf and marker remain byte-identical on collision."""
    run_dir = tmp_path / "outputs" / "synthetic_task" / "runs" / "run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "marker.bin"
    marker.write_bytes(b"do-not-touch")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    with pytest.raises(FileExistsError, match="explicit --resume"):
        experiments.run.prepare_fresh_run(_minimal_config(tmp_path / "outputs"), run_dir=run_dir)

    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    assert after == before
    assert not (run_dir / "config.yaml").exists()
    assert not (run_dir / "summary.json").exists()


def test_initialization_failure_never_looks_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-allocation initialization failure records failed, never completed."""
    run_dir = tmp_path / "run"

    def fail_save(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected config failure")

    monkeypatch.setattr(experiments.run.config_loader, "save_yaml", fail_save)
    with pytest.raises(OSError, match="injected config failure"):
        experiments.run.prepare_fresh_run(_minimal_config(tmp_path), run_dir=run_dir)

    assert run_dir.is_dir()
    assert experiments.run.read_run_summary(run_dir)["status"] == "failed"
    assert not common.paths.is_current_run_dir(run_dir)


def test_optuna_trial_paths_are_study_and_trial_qualified(tmp_path: Path) -> None:
    """Study identity and trial number independently qualify each leaf."""
    first = common.paths.resolve_optuna_trial_dir("task", "study-a", 3, output_root=tmp_path)
    other_study = common.paths.resolve_optuna_trial_dir("task", "study-b", 3, output_root=tmp_path)
    other_trial = common.paths.resolve_optuna_trial_dir("task", "study-a", 4, output_root=tmp_path)

    assert len({first, other_study, other_trial}) == 3
    experiments.run.allocate_run_directory(first)
    with pytest.raises(FileExistsError):
        experiments.run.allocate_run_directory(first)


def test_run_status_transitions_are_atomic_and_explicit(tmp_path: Path) -> None:
    """Initializing, running, interrupted, resumed, and completed are retained."""
    run_dir = experiments.run.allocate_run_directory(tmp_path / "run")
    experiments.run.transition_run_status(run_dir, "initializing")
    experiments.run.transition_run_status(run_dir, "running")
    experiments.run.transition_run_status(run_dir, "interrupted")
    experiments.run.transition_run_status(run_dir, "running")
    summary = experiments.run.transition_run_status(run_dir, "completed")

    assert summary["status"] == "completed"
    assert [entry["status"] for entry in summary["status_history"]] == [
        "initializing",
        "running",
        "interrupted",
        "running",
        "completed",
    ]
    with pytest.raises(experiments.run.RunLifecycleError, match="Invalid run status transition"):
        experiments.run.transition_run_status(run_dir, "failed")


def _resume_config(*, epochs: int = 4) -> dict[str, Any]:
    """Return fixed semantic sections used by resume identity tests."""
    return {
        "task": "synthetic_task",
        "run": {"name": "run", "device": "cpu", "seed": 3, "deterministic": True},
        "paths": {"output_root": "/saved/output"},
        "training": {"epochs": epochs, "evaluation_interval": 2},
        "data": {"train_dataset": "id", "ood_datasets": ["ood"]},
        "model": {"kind": "tiny", "params": {"width": 4}},
        "loss": {"data": {"kind": "mse"}},
        "optimizer": {"kind": "adam", "lr": 0.001},
        "scheduler": {"kind": "plateau", "factor": 0.5},
        "evaluation": {"objective": _objective("mse")},
    }


def test_resume_allows_only_duration_increase_and_runtime_location_changes() -> None:
    """Terminal duration may increase while device/output location remain runtime-only."""
    saved = _resume_config(epochs=4)
    requested = _resume_config(epochs=7)
    requested["run"]["device"] = "cuda"
    requested["paths"]["output_root"] = "/new/output"

    assert experiments.run.validate_resume_config(requested, saved) == 7
    assert experiments.run.validate_resume_config(saved, saved) == 4
    with pytest.raises(ValueError, match="only retain or increase"):
        experiments.run.validate_resume_config(_resume_config(epochs=3), saved)


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        ("task", "other_task"),
        ("data", {"train_dataset": "other", "ood_datasets": ["ood"]}),
        ("model", {"kind": "tiny", "params": {"width": 8}}),
        ("loss", {"data": {"kind": "relative_l2"}}),
        ("optimizer", {"kind": "sgd", "lr": 0.001}),
        ("scheduler", {"kind": "plateau", "factor": 0.8}),
    ],
)
def test_resume_rejects_semantic_changes(section: str, replacement: Any) -> None:
    """Task/data/model/loss/optimizer/scheduler semantics remain immutable."""
    saved = _resume_config()
    requested = _resume_config(epochs=6)
    requested[section] = replacement

    with pytest.raises(ValueError, match="incompatible with the saved run"):
        experiments.run.validate_resume_config(requested, saved)


def _attempt_file_lock(lock_path: Path, outcomes: Any) -> None:
    """Try one fail-fast lock acquisition in a forked process."""
    try:
        with common.locking.exclusive_file_lock(lock_path, blocking=False):
            outcomes.put("acquired")
    except common.locking.FileLockUnavailableError:
        outcomes.put("blocked")


def _hold_run_writer_lease(
    run_dir: Path,
    acquired: Any,
    release: Any,
    outcomes: Any,
) -> None:
    """Hold one run lease until the parent releases the worker."""
    try:
        with experiments.run.run_writer_lease(run_dir):
            acquired.set()
            if not release.wait(timeout=10):
                outcomes.put("timed out")
                return
        outcomes.put("released")
    except Exception as error:
        outcomes.put(f"{type(error).__name__}: {error}")


def test_file_lock_blocks_another_thread_in_the_same_process(tmp_path: Path) -> None:
    """Separate descriptors in sibling threads cannot bypass exclusivity."""
    lock_path = tmp_path / "thread.lock"
    outcomes: queue.Queue[str] = queue.Queue()

    def contend() -> None:
        try:
            with common.locking.exclusive_file_lock(lock_path, blocking=False):
                outcomes.put("acquired")
        except common.locking.FileLockUnavailableError:
            outcomes.put("blocked")

    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        contender = threading.Thread(target=contend)
        contender.start()
        contender.join(timeout=5)
        assert not contender.is_alive()

    assert outcomes.get_nowait() == "blocked"


def test_forked_child_does_not_inherit_parent_lock_ownership(tmp_path: Path) -> None:
    """A forked child cannot inherit the parent thread's re-entrant ownership."""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("POSIX fork context is required for descriptor-inheritance coverage")
    context = mp.get_context("fork")
    outcomes = context.Queue()
    lock_path = tmp_path / "fork.lock"

    with common.locking.exclusive_file_lock(lock_path, blocking=False):
        process = context.Process(target=_attempt_file_lock, args=(lock_path, outcomes))
        process.start()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        assert process.exitcode == 0
        assert outcomes.get(timeout=5) == "blocked"


def test_resume_prevalidation_rejects_a_second_process_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run lease is acquired before any resume artifact prevalidation."""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("POSIX fork context is required for process lease coverage")
    context = mp.get_context("fork")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    acquired = context.Event()
    release = context.Event()
    outcomes = context.Queue()
    holder = context.Process(
        target=_hold_run_writer_lease,
        args=(run_dir, acquired, release, outcomes),
    )
    holder.start()
    try:
        assert acquired.wait(timeout=5)
        monkeypatch.setattr(experiments.run.config_loader, "load_and_resolve_config", lambda _path: {})
        with pytest.raises(experiments.run.RunLifecycleError, match="active writer lease"):
            experiments.run.run_experiment("unused.yaml", resume=run_dir)
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)

    assert holder.exitcode == 0
    assert outcomes.get(timeout=5) == "released"
