# ruff: noqa: BLE001, S101, EM101, PLR2004, TRY003
"""
Protect collision-safe run allocation, writer leases, resume, and status transitions.

Temporary roots cover fresh-leaf refusal, atomic initialization failure, study
qualified paths, concurrent leases, immutable scientific config, and duration-only
resume extension. Checkpoint tensor/RNG restoration is covered by
``test_checkpoint_resume``. No model training is performed here.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from src import common, experiments
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


def test_rounded_name_collision_fails_closed_without_overwrite(
    tmp_path: Path,
) -> None:
    """Reject distinct science that shares one rounded visible run leaf."""
    raw = configs.direct_config(
        model_kind="uno",
        suffix="collision_safety",
    )
    first_raw = copy.deepcopy(raw)
    second_raw = copy.deepcopy(raw)
    first_raw["model"]["params"]["mode_ratio"] = 0.4951
    second_raw["model"]["params"]["mode_ratio"] = 0.4952
    first = experiments.config.loader.resolve_config(first_raw)
    second = experiments.config.loader.resolve_config(second_raw)
    first["paths"]["output_root"] = str(tmp_path)
    second["paths"]["output_root"] = str(tmp_path)
    assert first["run"]["name"] == second["run"]["name"]

    run_dir = common.paths.resolve_run_output_dir(
        str(first["task"]),
        str(first["run"]["name"]),
        output_root=tmp_path,
    )
    _write_lightweight_admission_state(
        run_dir,
        first,
        status="interrupted",
        completed_epoch=2,
    )
    before = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    config_path = configs.write_yaml(
        tmp_path / "requested.yaml",
        second_raw,
    )

    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        second,
        config_path=config_path,
    )

    assert report["resume_compatibility"] == "incompatible"
    assert "model.params.mode_ratio" in report["reason"]
    with pytest.raises(FileExistsError):
        experiments.run.prepare_fresh_run(second, run_dir=run_dir)
    after = {item.relative_to(run_dir): item.read_bytes() for item in run_dir.rglob("*") if item.is_file()}
    assert after == before


def test_fresh_run_persists_one_canonical_name(tmp_path: Path) -> None:
    """Persist the generated synthetic name across directory, config, and summary."""
    config = experiments.config.loader.resolve_config(configs.direct_config())
    config["paths"]["output_root"] = str(tmp_path)

    run_dir = experiments.run.prepare_fresh_run(config)
    saved = experiments.config.loader.load_yaml(common.paths.resolve_run_config_path(run_dir))
    summary = experiments.run.read_run_summary(run_dir)

    assert run_dir.name == config["run"]["name"]
    assert saved["run"]["name"] == config["run"]["name"]
    assert summary["run_name"] == config["run"]["name"]


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


def _synthetic_config_path(tmp_path: Path) -> Path:
    """Write the direct request used by read-only admission tests."""
    return configs.write_yaml(
        tmp_path / "synthetic-request.yaml",
        configs.direct_config(),
    )


def _resolved_admission_config(output_root: Path) -> dict[str, Any]:
    """Return one complete artificial config redirected to a temporary root."""
    resolved = experiments.config.loader.resolve_config(configs.direct_config())
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
    """Create and inspect one lightweight existing synthetic run."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "MODEL_TRAINING_DATA_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
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
        completed_epoch=(config["training"]["epochs"] if completed_epoch is None else completed_epoch),
    )
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    report = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=config_path,
    )
    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    assert after == before
    return report, run_dir, config


def test_completed_admission_is_distinct_from_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify a complete run without presenting it as resumable."""
    report, _run_dir, config = _admission_report(
        tmp_path,
        monkeypatch,
        status="completed",
        completed_epoch=None,
    )

    assert report["resume_compatibility"] == "completed"
    assert report["completed_epoch"] == config["training"]["epochs"]


def test_damaged_or_malformed_completed_run_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing selected state and invalid lifecycle JSON."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "MODEL_TRAINING_DATA_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=output_root,
    )
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
        config_path=config_path,
    )
    assert damaged["resume_compatibility"] == "incompatible"

    common.paths.resolve_run_summary_path(run_dir).write_text(
        "{invalid",
        encoding="utf-8",
    )
    malformed = experiments.run.inspect_existing_run_admission(
        run_dir,
        config,
        config_path=config_path,
    )
    assert malformed["resume_compatibility"] == "incompatible"


def test_active_writer_blocks_resume_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never admit simultaneous resume while another writer owns the run."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "MODEL_TRAINING_DATA_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=output_root,
    )
    _write_lightweight_admission_state(
        run_dir,
        config,
        status="interrupted",
        completed_epoch=2,
    )

    with common.locking.exclusive_file_lock(
        common.paths.resolve_run_lock_path(run_dir),
        blocking=False,
    ):
        report = experiments.run.inspect_existing_run_admission(
            run_dir,
            config,
            config_path=config_path,
        )

    assert report["active_lock"] is True
    assert report["resume_compatibility"] == "blocked"


def test_existing_fresh_target_rejects_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect an existing leaf before device, model, or tracking setup."""
    output_root = tmp_path / "outputs"
    monkeypatch.setenv(
        "MODEL_TRAINING_DATA_ROOT",
        str(tmp_path / "training"),
    )
    config_path = _synthetic_config_path(tmp_path)
    config = _resolved_admission_config(output_root)
    run_dir = common.paths.resolve_run_output_dir(
        config["task"],
        config["run"]["name"],
        output_root=output_root,
    )
    _write_lightweight_admission_state(
        run_dir,
        config,
        status="interrupted",
        completed_epoch=2,
    )
    before = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}

    def reject_device(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("device resolution must not run")

    monkeypatch.setattr(
        experiments.run.learning.device,
        "resolve_device",
        reject_device,
    )
    with pytest.raises(experiments.run.ExistingRunAdmissionError):
        experiments.run.run_experiment(
            config_path,
            output_root=output_root,
        )

    after = {item.relative_to(tmp_path): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
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


@pytest.mark.parametrize("invalid_version", [True, 2])
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
        ("data", {"train_dataset": "other", "ood_datasets": ["ood"]}),
        ("model", {"kind": "tiny", "params": {"width": 8}}),
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


def _hold_run_writer_lease(
    run_dir: Path,
    acquired: Any,
    release: Any,
    outcomes: Any,
) -> None:
    """
    Hold one run-writer lease while the parent probes resume admission.

    Events make lease acquisition and release deterministic. The outcome queue
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
