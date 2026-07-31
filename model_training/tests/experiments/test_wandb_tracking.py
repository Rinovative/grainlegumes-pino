# ruff: noqa: EM101, PLR2004, S101, S105, TRY003
"""
Protect optional W&B identity, telemetry, degradation, and bounded uploads.

An in-memory SDK fake covers disabled/offline/online initialization, exact resume
IDs, semantic config, monotonic epoch history, fixed monitor membership, curated
media allowlists, and mode-specific failures. Local run/checkpoint correctness is
owned by run and checkpoint tests; no network connection is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from src import experiments

tracking = experiments.tracking
_CONFIG_PATH = Path(__file__).parents[2] / "configs/experiments/steady_flow_fno.yaml"


class _FakeArtifact:
    """
    Record one explicit curated media bundle without SDK serialization.

    Arbitrary constructor metadata is retained verbatim; added files and tables
    are ordered records only. The fake performs no filesystem reads or upload.
    """

    def __init__(self, **metadata: Any) -> None:
        """Retain constructor metadata and initialize ordered add-call records."""
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []
        self.tables: list[tuple[Any, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        """Record one explicitly named media file without reading it."""
        self.files.append((path, name))

    def add(self, value: Any, name: str) -> None:
        """Record one named table-like object without SDK conversion."""
        self.tables.append((value, name))


class _FakeRun:
    """
    Record W&B run interactions and inject selected I/O failures locally.

    Summary mutation follows ordinary dictionary behavior; history, finish, file,
    and artifact calls are retained in order. No network, background process, or
    real W&B filesystem is involved.
    """

    def __init__(self) -> None:
        """Initialize empty call records with every injectable failure disabled."""
        self.summary: dict[str, Any] = {}
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.exit_codes: list[int] = []
        self.saved: list[tuple[str, str, str]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.fail_log = False
        self.fail_finish = False
        self.fail_save = False

    def log(self, data: dict[str, Any], *, step: int) -> None:
        """Record copied history at one explicit step or inject transport failure."""
        if self.fail_log:
            raise OSError("synthetic transport failure")
        self.logs.append((dict(data), step))

    def finish(self, exit_code: int = 0) -> None:
        """Record the terminal exit code or inject finish failure."""
        if self.fail_finish:
            raise OSError("synthetic finish failure")
        self.exit_codes.append(exit_code)

    def save(self, path: str, *, base_path: str, policy: str) -> None:
        """Record one allowlisted file-upload request or inject failure."""
        if self.fail_save:
            raise OSError("synthetic upload failure")
        self.saved.append((path, base_path, policy))

    def log_artifact(self, artifact: _FakeArtifact, *, aliases: list[str]) -> None:
        """Record one curated artifact and its explicit aliases."""
        self.artifacts.append((artifact, aliases))


class _FakeWandb:
    """
    Create isolated fake runs and capture SDK construction settings.

    Parameters
    ----------
    init_error : BaseException | None, optional
        Error raised after initialization settings are recorded, used to exercise
        redaction and failure-state publication.

    """

    def __init__(self, *, init_error: BaseException | None = None) -> None:
        """Initialize isolated SDK records and an optional post-capture init error."""
        self.initializations: list[dict[str, Any]] = []
        self.runs: list[_FakeRun] = []
        self.init_error = init_error
        self.created_artifacts: list[_FakeArtifact] = []
        self.created_tables: list[dict[str, Any]] = []

    def init(self, **settings: Any) -> _FakeRun:
        """Record settings and return a new isolated run unless failure is injected."""
        self.initializations.append(settings)
        if self.init_error is not None:
            raise self.init_error
        run = _FakeRun()
        self.runs.append(run)
        return run

    def Artifact(self, **metadata: Any) -> _FakeArtifact:  # noqa: N802
        """Construct and retain one SDK-shaped fake artifact."""
        artifact = _FakeArtifact(**metadata)
        self.created_artifacts.append(artifact)
        return artifact

    def Table(self, *, columns: list[str], data: list[list[object]]) -> dict[str, Any]:  # noqa: N802
        """Construct and retain one SDK-shaped table payload."""
        table = {"columns": columns, "data": data}
        self.created_tables.append(table)
        return table


def _resolved_config(
    *,
    enabled: bool,
    mode: str = "online",
    epochs: int | None = None,
) -> dict[str, Any]:
    """Resolve one experiment with an explicit optional-tracking selection."""
    raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
    raw["tracking"] = {
        "wandb": {
            "enabled": enabled,
            "project": "airflow-tests",
            "entity": "research-team",
            "group": "steady-flow",
            "tags": ["synthetic", "cpu"],
            "mode": mode,
        }
    }
    if epochs is not None:
        raw["training"]["epochs"] = epochs
    return experiments.config.loader.resolve_config(raw)


def _patch_wandb(
    monkeypatch: pytest.MonkeyPatch,
    fake_wandb: _FakeWandb,
) -> None:
    """Route only the lazy W&B import to a local fake."""
    original = tracking.importlib.import_module
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else original(name),
    )


def _state_recorder() -> tuple[dict[str, Any], Any]:
    """Return one merged local tracking-state recorder."""
    state: dict[str, Any] = {}

    def update(values: dict[str, Any]) -> None:
        """Merge one production state-updater payload into the observable record."""
        state.update(values)

    return state, update


def test_disabled_wandb_has_no_sdk_or_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Initialize disabled tracking while making any lazy SDK import fail the test.

    The no-op session must log/finish harmlessly without IDs, local tracking state,
    imports, or filesystem writes, preserving disabled mode as zero-side-effect.
    """

    def fail_import(_name: str) -> None:
        """Fail if disabled tracking crosses the lazy SDK-import boundary."""
        pytest.fail("disabled W&B must not import the SDK")

    config = _resolved_config(enabled=False)
    monkeypatch.setattr(tracking.importlib, "import_module", fail_import)
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        state_updater=update,
    )

    assert session.enabled is False
    assert session.run_id is None
    assert tracking.epoch_callback(session) is None
    session.log_epoch(1, {"train/loss_total": 2.0})
    session.finish(status="completed", result={"best_metric": 1.0})
    assert state == {}
    assert list(tmp_path.iterdir()) == []


def test_fresh_ids_are_opaque_isolated_and_online_resume_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Start equal-display-name runs in distinct roots, then resume the first online.

    Fresh IDs must be opaque and isolated; resume must reuse exactly the persisted
    ID with strict SDK mode and reject duplicate epoch history.
    """
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(enabled=True)
    first_state, first_update = _state_recorder()
    second_state, second_update = _state_recorder()

    first = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "one",
        state_updater=first_update,
    )
    second = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "two",
        state_updater=second_update,
    )
    resumed_state, resumed_update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "one",
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_epoch=5,
        state_updater=resumed_update,
    )

    assert first.run_id != second.run_id
    assert len(str(first.run_id)) == 32
    assert first_state["wandb_run_id"] == first.run_id
    assert second_state["wandb_run_id"] == second.run_id
    assert resumed.run_id == first.run_id
    assert resumed_state["session_kind"] == "resume"
    assert [settings["resume"] for settings in fake.initializations] == [
        "never",
        "never",
        "must",
    ]
    resumed.log_epoch(6, {"train/loss_total": 1.0, "global_step": 7.0})
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_epoch(6, {"train/loss_total": 1.0})


def test_offline_mode_needs_no_key_and_documents_same_id_resume_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Start and resume offline tracking without an API key under the pinned SDK behavior.

    The persisted ID must remain stable while SDK ``resume`` stays unset and local
    metadata explicitly records the same-ID-segment fallback rather than claiming strict resume.
    """
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(enabled=True, mode="offline")
    fresh_state, fresh_update = _state_recorder()
    fresh = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        state_updater=fresh_update,
    )
    resumed_state, resumed_update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=fresh.run_id,
        state_updater=resumed_update,
    )

    assert fresh.run_id == resumed.run_id
    assert [settings["mode"] for settings in fake.initializations] == [
        "offline",
        "offline",
    ]
    assert [settings["resume"] for settings in fake.initializations] == [None, None]
    assert "same_persisted_id_segment" in resumed_state["offline_resume_fallback"]
    assert fresh_state["status"] == "offline"
    assert "api_key" not in str(fake.initializations).lower()


def test_semantic_config_is_compact_complete_and_path_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Build semantic tracking config from resolved task, split, model, and device evidence.

    Required compact identities and parameter counts must be present while tensors,
    credentials, effective paths, and the host home remain absent.
    """
    secret = "super-secret-test-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = _resolved_config(enabled=True)
    sample_ids = ["case_0001", "case_0002", "case_0003"]
    split = {
        "eval_indices": torch.tensor([2, 0]),
        "metadata": {
            "datasets": {
                "train": {
                    "dataset_id": "train-data",
                    "fingerprint": "a" * 64,
                    "sample_count": 3,
                    "spatial_shape": [8, 8],
                    "task_contract_digest": config["task_contract"]["digest"],
                    "sample_ids": sample_ids,
                },
                "ood": {
                    "dataset_id": "ood-data",
                    "fingerprint": "b" * 64,
                    "sample_count": 3,
                    "spatial_shape": [8, 8],
                    "task_contract_digest": config["task_contract"]["digest"],
                    "sample_ids": sample_ids,
                },
            },
            "membership_digests": {
                "train": "c" * 64,
                "eval": "d" * 64,
                "ood": "e" * 64,
            },
        },
    }
    model = torch.nn.Linear(2, 3)
    payload = tracking.build_semantic_config(
        config,
        split_indices=split,
        normalizer_sha256="f" * 64,
        checkpoint_identity={"effective_config_digest": "1" * 64},
        model=model,
        device_metadata={
            "requested_policy": "cpu",
            "resolved_device": "cpu",
            "device_type": "cpu",
            "pytorch_version": torch.__version__,
        },
    )

    assert payload["task"]["id"] == "steady_flow"
    assert payload["task"]["schema_version"] == 1
    assert payload["model"]["parameter_counts"] == {"total": 9, "trainable": 9}
    assert payload["objective"]["id"] == "normalized_macro_rmse"
    assert payload["training_loss"]["kind"] == "relative_h1"
    assert payload["physics"]["continuity"] == "div_eps_velocity"
    assert payload["split_membership_digests"]["eval"] == "d" * 64
    assert "paths" not in payload["effective_config"]
    serialized = str(payload)
    assert secret not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert str(Path.home()) not in serialized


def test_epoch_history_and_terminal_summary_mirror_local_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Log one completed epoch and finish from an authoritative local run summary.

    History must normalize train/eval/objective namespaces at the real epoch, and
    terminal summary/files must mirror bounded local identities exactly once.
    """
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(enabled=True)
    (tmp_path / "config.yaml").write_text("task: steady_flow\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        semantic_config={
            "task": {"id": "steady_flow", "contract_digest": "t" * 64},
            "datasets": {
                "train": {"dataset_id": "steady_train", "fingerprint": "d" * 64},
                "ood": {"dataset_id": "steady_ood", "fingerprint": "o" * 64},
            },
            "split_membership_digests": {"eval": "e" * 64, "ood": "q" * 64},
            "normalizer": {"identity": "saved_run_normalizer.pt", "sha256": "b" * 64},
            "physics": {"continuity": "div_eps_velocity"},
        },
        state_updater=update,
    )
    callback = tracking.epoch_callback(session)
    assert callback is not None
    callback(
        5,
        {
            "train/loss_total": 0.75,
            "train/loss_data": 0.7,
            "train/learning_rate": 0.125,
            "train/epoch_duration_seconds": 1.5,
            "global_step": 12.0,
            "normalized_macro_rmse": 0.5,
            "normalized_relative_h1": 0.6,
            "physical_rmse_p": 2.0,
        },
    )
    local_summary = {
        "task": "steady_flow",
        "effective_config_digest": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "split_indices_sha256": "c" * 64,
        "runtime_device": {
            "requested_policy": "cpu",
            "resolved_device": "cpu",
        },
    }
    session.finish(
        status="completed",
        result={
            "best_epoch": 5,
            "best_metric": 0.5,
            "completed_epoch": 5,
            "global_step": 12,
        },
        local_summary=local_summary,
    )

    run = fake.runs[0]
    assert len(run.logs) == 1
    logged, step = run.logs[0]
    assert step == 5
    assert logged["epoch"] == 5
    assert logged["global_step"] == 12
    assert logged["train/loss_total"] == 0.75
    assert logged["train/loss_data"] == 0.7
    assert logged["train/learning_rate"] == 0.125
    assert logged["train/epoch_duration_seconds"] == 1.5
    assert logged["eval/normalized_macro_rmse"] == 0.5
    assert logged["eval/normalized_relative_h1"] == 0.6
    assert logged["eval/physical_rmse_p"] == 2.0
    assert logged["eval/objective_value"] == 0.5
    assert logged["objective/value"] == 0.5
    assert "normalized_macro_rmse" not in logged
    assert run.summary["run/status"] == "completed"
    assert run.summary["objective/best_value"] == 0.5
    assert run.summary["objective/best_epoch"] == 5
    assert run.summary["training/completed_epoch"] == 5
    assert run.summary["training/global_step"] == 12
    assert run.summary["device/requested"] == "cpu"
    assert run.summary["device/resolved"] == "cpu"
    assert run.summary["task/id"] == "steady_flow"
    assert run.summary["task/contract_digest"] == "t" * 64
    assert run.summary["config/digest"] == "a" * 64
    assert run.summary["split/artifact_sha256"] == "c" * 64
    assert run.summary["normalizer/sha256"] == "b" * 64
    assert run.summary["data/identities"]["train"]["dataset_id"] == "steady_train"
    assert run.summary["split/membership_digests"]["eval"] == "e" * 64
    assert run.summary["normalizer/identity"] == "saved_run_normalizer.pt"
    assert run.summary["physics/continuity"] == "div_eps_velocity"
    assert run.summary["tracking/status"] == "finished"
    assert run.summary["tracking/mode"] == "online"
    assert run.summary["tracking/run_id"] == session.run_id
    assert run.summary["run/local_role"] == f"steady_flow/runs/{config['run']['name']}"
    assert {Path(saved[0]).name for saved in run.saved} == {
        "config.yaml",
        "summary.json",
    }
    saved_by_name = {Path(path).name: (path, base_path, policy) for path, base_path, policy in run.saved}
    assert Path(saved_by_name["config.yaml"][0]).relative_to(saved_by_name["config.yaml"][1]).parts == (
        tmp_path.name,
        "config.yaml",
    )
    assert saved_by_name["summary.json"][1] == str(tmp_path)
    assert state["last_logged_epoch"] == 5
    assert state["status"] == "finished"
    assert run.exit_codes == [0]


def test_monitor_membership_is_a_fixed_persisted_eval_prefix() -> None:
    """
    Build monitor membership twice from a three-case saved eval split capped at two.

    Both results must be identical and bind ordered indices, sample IDs, saved digest,
    and bound, preventing observer sampling from drifting across epochs or resume.
    """
    config = _resolved_config(enabled=True)
    config["tracking"]["wandb"]["monitor"]["max_cases"] = 2
    split = {
        "eval_indices": torch.tensor([4, 1, 3]),
        "metadata": {
            "datasets": {
                "train": {
                    "fingerprint": "f" * 64,
                    "sample_ids": [f"case-{index}" for index in range(5)],
                }
            },
            "membership_digests": {"eval": "e" * 64},
        },
    }

    first = tracking.build_monitor_membership(config, split)
    second = tracking.build_monitor_membership(config, split)

    assert first == second
    assert first is not None
    assert first["source_indices"] == [4, 1]
    assert first["sample_ids"] == ["case-4", "case-1"]
    assert first["saved_eval_membership_digest"] == "e" * 64
    assert first["max_cases"] == 2
    assert isinstance(first["membership_digest"], str)


def test_bounded_monitor_logs_both_continuities_only_at_declared_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Invoke one fixed-membership monitor at its interval and terminal epoch.

    Exactly the four shared physics diagnostics must be added at most once per
    eligible epoch, preserving bounded observer cost and dual continuity.
    """
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(enabled=True, epochs=3)
    config["tracking"]["wandb"]["monitor"]["interval"] = 2
    calls = 0

    def monitor() -> dict[str, float]:
        """Count evaluations and return the complete bounded physics payload."""
        nonlocal calls
        calls += 1
        return {
            "monitor/momentum_residual_mse": 1.0,
            "monitor/div_velocity_mse": 2.0,
            "monitor/div_eps_velocity_mse": 3.0,
            "monitor/pressure_boundary_mse": 4.0,
        }

    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        monitor_evaluator=monitor,
    )
    for epoch in (1, 2, 3):
        session.log_epoch(
            epoch,
            {
                "train/loss_total": float(epoch),
                "global_step": float(epoch),
            },
        )

    assert calls == 2
    assert "monitor/div_velocity_mse" not in fake.runs[0].logs[0][0]
    assert fake.runs[0].logs[1][0]["monitor/div_velocity_mse"] == 2.0
    assert fake.runs[0].logs[1][0]["monitor/div_eps_velocity_mse"] == 3.0
    assert fake.runs[0].logs[2][0]["monitor/pressure_boundary_mse"] == 4.0


def test_online_history_failure_degrades_once_but_offline_failure_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject the same history-write failure into online and offline fake sessions.

    Online mode must degrade once and suppress later writes without harming local
    completion; offline mode must raise because the requested local record was lost.
    """
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    online_state, online_update = _state_recorder()
    online = tracking.initialize_wandb(
        _resolved_config(enabled=True),
        run_dir=tmp_path / "online",
        state_updater=online_update,
    )
    fake.runs[0].fail_log = True
    online.log_epoch(1, {"train/loss_total": 1.0})
    online.log_epoch(2, {"train/loss_total": 0.5})
    assert online.degraded is True
    assert online_state["status"] == "degraded"
    assert online_state["degraded_operation"] == "history"
    online.finish(status="completed", result={"best_metric": 0.5})
    assert fake.runs[0].exit_codes == [0]

    offline_state, offline_update = _state_recorder()
    offline = tracking.initialize_wandb(
        _resolved_config(enabled=True, mode="offline"),
        run_dir=tmp_path / "offline",
        state_updater=offline_update,
    )
    fake.runs[1].fail_log = True
    with pytest.raises(tracking.TrackingIOError, match="offline W&B history"):
        offline.log_epoch(1, {"train/loss_total": 1.0})
    assert offline_state["status"] == "failed"
    assert offline.degraded is False


def test_online_degradation_does_not_change_local_objective_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Combine an authoritative objective consumer with a failing online W&B callback.

    The objective consumer must still receive every epoch exactly once, proving
    observer degradation cannot alter scheduling, checkpointing, or Optuna evidence.
    """
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(
        _resolved_config(enabled=True),
        run_dir=tmp_path,
    )
    fake.runs[0].fail_log = True
    consumed: list[tuple[int, float]] = []

    def consume_objective(epoch: int, values: dict[str, float]) -> None:
        """Record the authoritative objective path independently of W&B."""
        consumed.append((epoch, values["normalized_macro_rmse"]))

    callback = tracking.combine_epoch_callbacks(
        consume_objective,
        tracking.epoch_callback(session),
    )
    assert callback is not None
    callback(
        5,
        {
            "train/loss_total": 1.0,
            "normalized_macro_rmse": 0.25,
        },
    )

    assert consumed == [(5, 0.25)]
    assert session.degraded is True
    assert fake.runs[0].logs == []


def test_initialization_error_is_actionable_and_never_exposes_the_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject an online SDK initialization error containing the active API key.

    The public error and persisted failure state must redact the secret and classify
    failure before start, preventing authentication diagnostics from leaking credentials.
    """
    secret = "top-secret-wandb-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    fake = _FakeWandb(
        init_error=RuntimeError(f"authentication failed WANDB_API_KEY={secret}"),
    )
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()

    with pytest.raises(tracking.TrackingInitializationError) as captured:
        tracking.initialize_wandb(
            _resolved_config(enabled=True),
            run_dir=tmp_path,
            state_updater=update,
        )

    assert secret not in str(captured.value)
    assert secret not in str(state)
    assert state["status"] == "failed_before_start"
    assert state["failed_operation"] == "initialization"


def test_file_and_post_artifact_uploads_are_explicitly_allowlisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercise file and post-artifact upload allowlists with valid and forbidden kinds.

    Only explicitly enabled best/config/provenance and curated external media/table
    keys may be recorded; last checkpoints, data, globs, and cache-resident plots must fail.
    """
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(enabled=True)
    session = tracking.initialize_wandb(config, run_dir=tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("task: steady_flow\n", encoding="utf-8")
    session.upload_files({"config": config_path})
    best = tmp_path / "best_checkpoint.pt"
    best.write_bytes(b"checkpoint")
    with pytest.raises(tracking.TrackingUploadError, match="disabled"):
        session.upload_files({"best_checkpoint": best})
    config["tracking"]["wandb"]["upload"]["best_checkpoint"] = True
    session.upload_files({"best_checkpoint": best})
    last = tmp_path / "last_checkpoint.pt"
    last.write_bytes(b"resume only")
    for forbidden_kind in (
        "last_checkpoint",
        "split_indices",
        "normalizer",
        "raw_dataset",
        "merged_dataset",
        "per_case_npz",
    ):
        with pytest.raises(tracking.TrackingUploadError, match="Unsupported"):
            session.upload_files({forbidden_kind: last})

    artifact_root = tmp_path / "analysis" / "id"
    artifact_root.mkdir(parents=True)
    provenance = artifact_root / "artifact_provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    session.upload_files({"artifact_provenance": provenance})
    cached_plot = artifact_root / "pareto.png"
    cached_plot.write_bytes(b"rendered")
    with pytest.raises(tracking.TrackingUploadError, match="outside the immutable artifact cache"):
        session.upload_post_artifact(
            artifact_root=artifact_root,
            media_files={"accuracy_physics_pareto": cached_plot},
        )

    rendered_root = tmp_path / "rendered"
    rendered_root.mkdir()
    plot = rendered_root / "pareto.png"
    plot.write_bytes(b"rendered")
    table = {"columns": ["label", "score"], "data": [["run", 0.25]]}
    session.upload_post_artifact(
        artifact_root=artifact_root,
        media_files={"accuracy_physics_pareto": plot},
        tables={"run_summary_table": table},
    )

    run = fake.runs[0]
    assert {Path(item[0]).name for item in run.saved} == {
        "config.yaml",
        "best_checkpoint.pt",
        "artifact_provenance.json",
    }
    assert len(run.artifacts) == 1
    bundle, aliases = run.artifacts[0]
    assert aliases == ["latest"]
    assert bundle.files[0][1] == "accuracy_physics_pareto.png"
    assert fake.created_tables == [table]
    assert bundle.tables == [(table, "run_summary_table")]
    with pytest.raises(tracking.TrackingUploadError, match="Unsupported"):
        session.upload_post_artifact(
            artifact_root=artifact_root,
            media_files={"raw_dataset": plot},
        )


@pytest.mark.parametrize(
    ("wandb_settings", "match"),
    [
        ({"enabled": 1}, r"tracking\.wandb\.enabled"),
        ({"mode": "disabled"}, r"tracking\.wandb\.mode"),
        ({"tags": ["duplicate", "duplicate"]}, r"tracking\.wandb\.tags must be unique"),
        ({"monitor": {"interval": 0}}, r"tracking\.wandb\.monitor\.interval"),
        ({"training_images": {"max_snapshots": False}}, r"tracking\.wandb\.training_images\.max_snapshots"),
        ({"upload": {"best_checkpoint": 1}}, r"tracking\.wandb\.upload\.best_checkpoint"),
    ],
)
def test_wandb_config_is_strict(
    wandb_settings: dict[str, Any],
    match: str,
) -> None:
    """
    Vary W&B booleans, mode, duplicate tags, monitor bounds, and upload scalar types.

    Every malformed family must fail at its semantic config path, preventing YAML
    coercion or SDK-only values from entering tracking lifecycle behavior.
    """
    raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
    raw["tracking"] = {"wandb": wandb_settings}

    with pytest.raises(ValueError, match=match):
        experiments.config.loader.resolve_config(raw)
