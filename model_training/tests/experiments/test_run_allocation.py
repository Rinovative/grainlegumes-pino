# ruff: noqa: BLE001, S101, EM101, PLR2004, SLF001, TRY003
"""
Protect collision-safe run allocation, writer leases, resume, and status transitions.

Temporary roots cover fresh-leaf refusal, atomic initialization failure, study
qualified paths, concurrent leases, immutable scientific config, and duration-only
resume extension. Checkpoint tensor/RNG restoration is covered by
``test_checkpoint_resume``; no model training is performed here.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import shlex
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from src import common, experiments
from src.experiments.cli import cli_train
from support import configs


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
    """
    Prepare a fresh run at an existing leaf containing an authoritative marker.

    Collision must fail before config/summary writes and preserve every byte,
    requiring callers to choose explicit resume rather than implicit overwrite.
    """
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


def _resolved_admission_config(output_root: Path) -> dict[str, Any]:
    """Return one complete maintained config redirected only to a temporary output root."""
    resolved = experiments.config.loader.load_and_resolve_config(configs.experiment_config_paths()[0])
    resolved["paths"]["output_root"] = str(output_root)
    return resolved


def _write_lightweight_admission_state(
    run_dir: Path,
    config: dict[str, Any],
    *,
    status: str,
    completed_epoch: int,
    include_best: bool = True,
) -> None:
    """Publish only metadata and placeholder state needed by read-only admission inspection."""
    run_dir.mkdir(parents=True)
    experiments.config.loader.save_yaml(config, common.paths.resolve_run_config_path(run_dir))
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {
            "schema_version": experiments.run.RUN_SUMMARY_SCHEMA_VERSION,
            "status": status,
            "completed_epoch": completed_epoch,
        },
    )
    common.paths.resolve_split_indices_path(run_dir).touch()
    common.paths.resolve_normalizer_path(run_dir).touch()
    common.paths.resolve_last_checkpoint_file(run_dir).touch()
    if include_best:
        common.paths.resolve_best_checkpoint_file(run_dir).touch()


def _admission_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    completed_epoch: int | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Create one lightweight existing run and inspect it without runtime allocation."""
    output_root = tmp_path / "outputs"
    training_root = tmp_path / "training"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=output_root,
    )
    _write_lightweight_admission_state(
        run_dir,
        config,
        status=status,
        completed_epoch=config["training"]["epochs"] if completed_epoch is None else completed_epoch,
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=configs.experiment_config_paths()[0],
    )
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    return report, run_dir, config


def test_compatible_interrupted_admission_prints_a_parser_valid_resume_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recommend explicit resume only for a compatible inactive interrupted run."""
    report, run_dir, _config = _admission_report(
        tmp_path,
        monkeypatch,
        status="interrupted",
        completed_epoch=3,
    )

    assert report["resume_compatibility"] == "compatible"
    assert report["active_lock"] is False
    assert report["last_checkpoint_available"] is True
    assert report["best_checkpoint_available"] is True
    cli_train._print_existing_run_admission(report)
    stderr = capsys.readouterr().err
    command_line = next(line.strip() for line in stderr.splitlines() if line.strip().startswith("./scripts/docker_job.sh"))
    arguments = shlex.split(command_line)
    assert arguments[:2] == ["./scripts/docker_job.sh", "train"]
    parsed = cli_train._build_parser().parse_args(arguments[2:])
    assert Path(parsed.resume) == run_dir
    assert parsed.config_path == report["config_path"]


def test_explicit_historical_run_reference_preserves_persisted_name_without_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit persisted old naming metadata read-only through its exact directory reference."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(tmp_path / "training"))
    requested = _resolved_admission_config(output_root)
    historical = copy.deepcopy(requested)
    historical["run"]["prefix"] = None
    historical["run"]["suffix"] = "historical_manual_identity"
    historical["run"]["name"] = "old_format_exact_leaf"
    run_dir = common.paths.resolve_run_output_dir(
        historical["task"],
        historical["run"]["name"],
        output_root=output_root,
    )
    _write_lightweight_admission_state(
        run_dir,
        historical,
        status="interrupted",
        completed_epoch=2,
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        requested,
        config_path=configs.experiment_config_paths()[0],
    )

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    proposed_dir = common.paths.resolve_run_output_dir(
        requested["task"],
        requested["run"]["name"],
        output_root=output_root,
    )
    assert report["resume_compatibility"] == "compatible"
    assert report["run_dir"] == str(run_dir.resolve())
    assert after == before
    assert not proposed_dir.exists()
    assert tuple(path.name for path in run_dir.parent.iterdir()) == (historical["run"]["name"],)


def test_completed_admission_is_separate_and_has_no_resume_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a completed target separately without recommending same-target resume."""
    report, _run_dir, config = _admission_report(
        tmp_path,
        monkeypatch,
        status="completed",
        completed_epoch=None,
    )
    assert report["resume_compatibility"] == "completed"
    assert report["completed_epoch"] == config["training"]["epochs"]

    cli_train._print_existing_run_admission(report)
    stderr = capsys.readouterr().err
    assert "Run already completed; no training was started." in stderr
    assert f"Completed epoch: {config['training']['epochs']} / {config['training']['epochs']}" in stderr
    assert "Resume explicitly with:" not in stderr


def test_incompatible_and_damaged_admission_fail_closed_without_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose a specific semantic incompatibility and suppress executable resume advice."""
    report, run_dir, config = _admission_report(
        tmp_path,
        monkeypatch,
        status="interrupted",
        completed_epoch=2,
    )
    saved = copy.deepcopy(config)
    saved["optimizer"]["lr"] = float(saved["optimizer"]["lr"]) * 2
    experiments.config.loader.save_yaml(saved, common.paths.resolve_run_config_path(run_dir))
    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=configs.experiment_config_paths()[0],
    )

    assert report["resume_compatibility"] == "incompatible"
    assert "Differing field(s): optimizer.lr" in report["reason"]
    cli_train._print_existing_run_admission(report)
    stderr = capsys.readouterr().err
    assert "Resume compatibility: incompatible" in stderr
    assert "Reason: Requested config is incompatible" in stderr
    assert "Resume explicitly with:" not in stderr


def test_damaged_completed_and_malformed_summary_admission_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat missing selected state and invalid lifecycle JSON as incompatible."""
    output_root = tmp_path / "outputs"
    training_root = tmp_path / "training"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(config["task"], config["run"]["name"], output_root=output_root)
    _write_lightweight_admission_state(
        run_dir,
        config,
        status="completed",
        completed_epoch=config["training"]["epochs"],
        include_best=False,
    )

    damaged = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=configs.experiment_config_paths()[0],
    )
    assert damaged["resume_compatibility"] == "incompatible"
    assert damaged["reason"] == "Completed run is missing best_checkpoint.pt."

    common.paths.resolve_run_summary_path(run_dir).write_text("{invalid", encoding="utf-8")
    malformed = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=configs.experiment_config_paths()[0],
    )
    assert malformed["resume_compatibility"] == "incompatible"
    assert "invalid JSON" in malformed["reason"]


def test_active_lock_admission_is_blocked_without_simultaneous_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report an active writer as blocked and never recommend simultaneous resume."""
    output_root = tmp_path / "outputs"
    training_root = tmp_path / "training"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(config["task"], config["run"]["name"], output_root=output_root)
    _write_lightweight_admission_state(run_dir, config, status="interrupted", completed_epoch=2)

    with common.locking.exclusive_file_lock(common.paths.resolve_run_lock_path(run_dir), blocking=False):
        report = experiments.run.inspect_existing_run_admission(
            run_dir,
            config,
            config_path=configs.experiment_config_paths()[0],
        )

    assert report["active_lock"] is True
    assert report["resume_compatibility"] == "blocked"
    cli_train._print_existing_run_admission(report)
    stderr = capsys.readouterr().err
    assert "Active lock: present" in stderr
    assert "Resume compatibility: blocked" in stderr
    assert "Resume explicitly with:" not in stderr


def test_fresh_existing_run_rejects_before_device_model_or_tracking_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify an existing leaf before concrete device or downstream runtime setup."""
    output_root = tmp_path / "outputs"
    training_root = tmp_path / "training"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    config_path = configs.experiment_config_paths()[0]
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(config["task"], config["run"]["name"], output_root=output_root)
    _write_lightweight_admission_state(run_dir, config, status="interrupted", completed_epoch=2)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def reject_device(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("device resolution must not run for an existing fresh destination")

    monkeypatch.setattr(experiments.run.learning.device, "resolve_device", reject_device)
    with pytest.raises(experiments.run.ExistingRunAdmissionError) as captured:
        experiments.run.run_experiment(config_path, output_root=output_root)

    assert captured.value.report["resume_compatibility"] == "compatible"
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_initialization_failure_never_looks_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject config-save failure immediately after run-directory allocation.

    The leaf must retain a failed status and remain non-loadable, preventing partial
    initialization from appearing complete merely because a directory exists.
    """
    run_dir = tmp_path / "run"

    def fail_save(*_args: Any, **_kwargs: Any) -> None:
        """Fail exactly where fresh-run initialization persists config."""
        raise OSError("injected config failure")

    monkeypatch.setattr(experiments.run.config_loader, "save_yaml", fail_save)
    with pytest.raises(OSError, match="injected config failure"):
        experiments.run.prepare_fresh_run(_minimal_config(tmp_path), run_dir=run_dir)

    assert run_dir.is_dir()
    assert experiments.run.read_run_summary(run_dir)["status"] == "failed"
    assert not common.paths.is_current_run_dir(run_dir)


@pytest.mark.parametrize("invalid_version", [True, 1.0, 0, 2])
def test_run_summary_requires_type_exact_schema_version(
    tmp_path: Path,
    invalid_version: object,
) -> None:
    """Reject every non-current integer and numeric lookalike before lifecycle use."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    common.serialization.atomic_write_json(
        common.paths.resolve_run_summary_path(run_dir),
        {"schema_version": invalid_version, "status": "completed"},
    )
    for filename in common.paths.CURRENT_RUN_REQUIRED_FILES:
        path = run_dir / filename
        if not path.exists():
            path.touch()
    assert not common.paths.is_current_run_dir(run_dir)

    with pytest.raises(experiments.run.RunLifecycleError, match="run summary schema"):
        experiments.run.read_run_summary(run_dir)


def test_optuna_trial_paths_are_study_and_trial_qualified(tmp_path: Path) -> None:
    """
    Resolve equal trial numbers across studies and unequal trials within one study.

    All leaves must be distinct and exact reallocation must fail, protecting Optuna
    output ownership from study/trial collisions.
    """
    first = common.paths.resolve_optuna_trial_dir("task", "study-a", "fno__s1__optuna_trial_003", output_root=tmp_path)
    other_study = common.paths.resolve_optuna_trial_dir("task", "study-b", "fno__s1__optuna_trial_003", output_root=tmp_path)
    other_trial = common.paths.resolve_optuna_trial_dir("task", "study-a", "fno__s1__optuna_trial_004", output_root=tmp_path)
    assert first.name == "fno__s1__optuna_trial_003"

    assert len({first, other_study, other_trial}) == 3
    experiments.run.allocate_run_directory(first)
    with pytest.raises(FileExistsError):
        experiments.run.allocate_run_directory(first)


def test_run_status_transitions_are_atomic_and_explicit(tmp_path: Path) -> None:
    """
    Drive one run through initialization, interruption, resume, and completion.

    Atomic history must retain every state in order and reject a terminal transition,
    preserving an auditable lifecycle rather than only the latest label.
    """
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


def test_runtime_sessions_append_requested_and_resolved_facts_without_rewriting_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Start with ``auto`` resolved to CPU, interrupt, then resume under explicit CPU.

    The second session must append requested/resolved facts without rewriting the
    first, preserving runtime history across portable resume.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / "run-sessions")
    experiments.run.transition_run_status(run_dir, "initializing")
    monkeypatch.setattr(experiments.run.learning.device.torch.cuda, "is_available", lambda: False)
    first_resolution = experiments.run.learning.device.resolve_device("auto")
    first_started = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            first_resolution,
            started_at=first_started,
        ),
    )
    first_session = experiments.run.read_run_summary(run_dir)["runtime_sessions"][0].copy()

    experiments.run.transition_run_status(run_dir, "interrupted")
    second_resolution = experiments.run.learning.device.resolve_device("cpu")
    second_started = datetime(2025, 1, 2, 5, 6, tzinfo=UTC)
    summary = experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            second_resolution,
            started_at=second_started,
        ),
    )

    assert summary["runtime_sessions"][0] == first_session
    assert summary["runtime_sessions"][0]["requested_policy"] == "auto"
    assert summary["runtime_sessions"][0]["resolved_device"] == "cpu"
    assert summary["runtime_sessions"][1]["requested_policy"] == "cpu"
    assert summary["runtime_sessions"][1]["resolved_device"] == "cpu"
    assert summary["runtime_sessions"][1]["started_at"] == second_started.isoformat()
    assert summary["runtime_device"] == second_resolution.as_dict()


def test_tracking_runtime_updates_are_atomic_and_session_scoped(
    tmp_path: Path,
) -> None:
    """
    Update tracking state inside one running session, then append a resume session.

    Atomic session mutation must preserve status history and earlier tracking facts,
    keeping observer state scoped to the runtime session that produced it.
    """
    run_dir = experiments.run.allocate_run_directory(tmp_path / "tracking-sessions")
    experiments.run.transition_run_status(run_dir, "initializing")
    resolution = experiments.run.learning.device.resolve_device("cpu")
    first_started = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    experiments.run.transition_run_status(
        run_dir,
        "running",
        updates=experiments.run.runtime_session_updates(
            run_dir,
            resolution,
            started_at=first_started,
            session_id="first-session",
            tracking_state={
                "requested_mode": "online",
                "status": "active",
            },
        ),
    )
    updated = experiments.run.update_runtime_session(
        run_dir,
        "first-session",
        {
            "wandb_run_id": "opaque-run-id",
            "last_logged_epoch": 2,
            "status": "failed",
            "failed_operation": "history",
        },
    )
    first_session = copy.deepcopy(updated["runtime_sessions"][0])
    history = copy.deepcopy(updated["status_history"])

    appended = experiments.run.append_runtime_session(
        run_dir,
        resolution,
        started_at=datetime(2025, 1, 2, 5, 6, tzinfo=UTC),
        session_id="resume-session",
        tracking_state={
            "requested_mode": "online",
            "wandb_run_id": "opaque-run-id",
            "session_kind": "resume",
            "status": "active",
        },
    )

    assert appended["status"] == "running"
    assert appended["status_history"] == history
    assert appended["runtime_sessions"][0] == first_session
    assert first_session["tracking"]["requested_mode"] == "online"
    assert first_session["tracking"]["wandb_run_id"] == "opaque-run-id"
    assert first_session["tracking"]["status"] == "failed"
    assert first_session["tracking"]["failed_operation"] == "history"
    assert appended["runtime_sessions"][1]["tracking"]["session_kind"] == "resume"


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
    """
    Increase terminal epochs while changing only device and output-root location.

    Resume must accept retained/increased duration and operational relocation but
    reject shortening, protecting completed progress from rollback.
    """
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
    """
    Replace each scientific continuation section while increasing allowed duration.

    Every parametrized section must fail resume compatibility, proving operational
    allowances cannot conceal task, data, model, loss, or optimization drift.
    """
    saved = _resume_config()
    requested = _resume_config(epochs=6)
    requested[section] = replacement

    with pytest.raises(ValueError, match="incompatible with the saved run"):
        experiments.run.validate_resume_config(requested, saved)


def _attempt_file_lock(lock_path: Path, outcomes: Any) -> None:
    """
    Attempt one nonblocking lock acquisition in a forked worker.

    Only a string outcome crosses the process boundary, allowing the parent to
    distinguish inherited ownership from correctly blocked acquisition.
    """
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
    """
    Hold one run-writer lease while the parent probes resume admission.

    Events make lease acquisition and release deterministic; the outcome queue
    reports timeout, release, or an unexpected worker exception.
    """
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
    """
    Contend for one nonblocking file lock from a sibling thread in the same process.

    The second descriptor must be blocked, protecting process-local re-entrancy from
    being mistaken for ownership by arbitrary threads.
    """
    lock_path = tmp_path / "thread.lock"
    outcomes: queue.Queue[str] = queue.Queue()

    def contend() -> None:
        """Attempt sibling-thread acquisition and report its boundary result."""
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
    """
    Fork a child while the parent owns one nonblocking file lock.

    The child must report blocked rather than inherit the parent's ownership token,
    preserving mutual exclusion across forked worker processes.
    """
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
    """
    Hold a run-writer lease in one process while a second process attempts resume.

    Resume must fail on the active writer before config or artifact admission,
    preventing concurrent mutation during prevalidation.
    """
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("POSIX fork context is required for process lease coverage")
    context = mp.get_context("fork")
    training_root = tmp_path / "training"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
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
        monkeypatch.setattr(
            experiments.run.config_loader,
            "load_yaml",
            lambda _path: {"run": {"device": "cpu"}, "training": {"mixed_precision": False}},
        )
        monkeypatch.setattr(experiments.run.config_loader, "resolve_config", lambda raw: raw)
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
    lock_path = common.paths.resolve_run_lock_path(run_dir)
    assert lock_path.parent == training_root / ".state/runs/locks"
    assert lock_path.is_file()
    assert lock_path.stat().st_size == 0
    assert not list(run_dir.rglob("*.lock"))
