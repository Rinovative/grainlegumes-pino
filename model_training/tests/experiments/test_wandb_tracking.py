# ruff: noqa: EM101, PLR2004, S101, S105, SLF001, TRY003
"""Exercise W&B lifecycle, provenance, failure, and upload behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from src import experiments
from support import configs

tracking = experiments.tracking


class _FakeArtifact:
    def __init__(self, **metadata: Any) -> None:
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []
        self.tables: list[tuple[Any, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))

    def add(self, value: Any, name: str) -> None:
        self.tables.append((value, name))


class _FakeRun:
    def __init__(self, *, tags: tuple[str, ...] = ()) -> None:
        self.summary: dict[str, Any] = {}
        self.tags = tags
        self.metric_definitions: list[tuple[str, dict[str, Any]]] = []
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.exit_codes: list[int] = []
        self.saved: list[tuple[str, str, str]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.fail_log = False
        self.fail_finish = False

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.metric_definitions.append((name, kwargs))

    def log(self, data: dict[str, Any], *, step: int) -> None:
        if self.fail_log:
            raise OSError("synthetic transport failure")
        self.logs.append((dict(data), step))

    def finish(self, exit_code: int = 0) -> None:
        if self.fail_finish:
            raise OSError("synthetic finish failure")
        self.exit_codes.append(exit_code)

    def save(self, path: str, *, base_path: str, policy: str) -> None:
        self.saved.append((path, base_path, policy))

    def log_artifact(self, artifact: _FakeArtifact, *, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))


class _FakeWandb:
    def __init__(
        self,
        *,
        init_error: BaseException | None = None,
        resumed_tags: tuple[str, ...] | None = None,
    ) -> None:
        self.initializations: list[dict[str, Any]] = []
        self.runs: list[_FakeRun] = []
        self.init_error = init_error
        self.resumed_tags = resumed_tags
        self.created_artifacts: list[_FakeArtifact] = []
        self.created_tables: list[dict[str, Any]] = []

    def init(self, **settings: Any) -> _FakeRun:
        self.initializations.append(settings)
        if self.init_error is not None:
            raise self.init_error
        tags = tuple(settings.get("tags") or ())
        if settings.get("resume") == "must" and self.resumed_tags is not None:
            tags = self.resumed_tags
        run = _FakeRun(tags=tags)
        self.runs.append(run)
        return run

    def Artifact(self, **metadata: Any) -> _FakeArtifact:  # noqa: N802
        artifact = _FakeArtifact(**metadata)
        self.created_artifacts.append(artifact)
        return artifact

    def Table(  # noqa: N802
        self,
        *,
        columns: list[str],
        data: list[list[object]],
    ) -> dict[str, Any]:
        table = {"columns": columns, "data": data}
        self.created_tables.append(table)
        return table


@pytest.fixture(autouse=True)
def _fake_online_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "test-only-wandb-key")


def _resolved_config(
    *,
    mode: str = "online",
    workflow: str = "train",
    study: str | None = None,
    epochs: int = 3,
    upload: bool = False,
    physics_enabled: bool = False,
) -> dict[str, Any]:
    raw = configs.direct_config(
        physics_enabled=physics_enabled,
        wandb_mode=mode,
        workflow=workflow,
    )
    settings = raw["tracking"]["wandb"]
    settings["upload"]["evaluation_artifacts"] = upload
    if study is not None:
        settings["study"] = study
    raw["training"]["epochs"] = epochs
    return experiments.config.loader.resolve_config(raw)


def _patch_wandb(monkeypatch: pytest.MonkeyPatch, fake: _FakeWandb) -> None:
    original = tracking.importlib.import_module
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda name: fake if name == "wandb" else original(name),
    )


def _state_recorder() -> tuple[dict[str, Any], Any]:
    state: dict[str, Any] = {}

    def update(values: dict[str, Any]) -> None:
        state.update(values)

    return state, update


def _split_evidence(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "eval_indices": torch.tensor([2, 0]),
        "metadata": {
            "datasets": {
                role: {
                    "dataset_id": f"artificial-{role}",
                    "fingerprint": marker * 64,
                    "sample_count": 3,
                    "spatial_shape": [8, 8],
                    "data_contract_digest": config["task_contract"]["data_contract_digest"],
                    "sample_ids": ["sample-a", "sample-b", "sample-c"],
                }
                for role, marker in (("train", "a"), ("ood", "b"))
            },
            "membership_digests": {
                "train": "c" * 64,
                "eval": "d" * 64,
                "ood": "e" * 64,
            },
            "n_train_full": 3,
            "n_train": 1,
            "n_eval": 2,
            "n_ood_full": 3,
            "n_ood": 1,
            "train_ratio": 0.8,
            "ood_fraction": 0.2,
            "split_seed": 9,
        },
    }


@pytest.mark.parametrize(
    ("role", "expected"),
    [(None, "optuna"), ("production", "optuna-production"), ("smoke", "optuna-smoke")],
)
def test_runtime_tags_deduplicate_and_qualify_explicit_optuna_roles(
    role: str | None,
    expected: str,
) -> None:
    """Qualify only explicit Optuna roles while retaining stable user tags."""
    settings = {
        "workflow": "optuna_trial",
        "tags": ["fno", "reviewed", "optuna", "reviewed"],
    }
    tags = tracking._runtime_wandb_tags(settings, {"tuning": {"study_role": role}})
    assert tags == ["fno", "reviewed", expected]


def test_disabled_tracking_has_no_sdk_or_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave SDK and disk untouched when tracking is disabled."""
    monkeypatch.setattr(
        tracking.importlib,
        "import_module",
        lambda _name: pytest.fail("disabled tracking imported the SDK"),
    )
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(mode="disabled"),
        run_dir=tmp_path,
        state_updater=update,
    )
    session.log_epoch(1, {"train/loss_total": 2.0})
    session.finish(status="completed")
    assert not session.enabled
    assert state == {}
    assert list(tmp_path.iterdir()) == []


def test_online_authentication_is_checked_before_sdk_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject absent online credentials before importing the SDK."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    with pytest.raises(tracking.TrackingInitializationError, match="authentication"):
        tracking.initialize_wandb(
            _resolved_config(),
            run_dir=tmp_path,
            state_updater=update,
        )
    assert fake.initializations == []
    assert state["status"] == "failed_before_start"
    assert state["failed_operation"] == "authentication"


def test_fresh_runs_are_isolated_and_resume_reuses_persisted_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate fresh identities and resume only the persisted run."""
    manual_tags = ("reviewed-manually",)
    fake = _FakeWandb(resumed_tags=manual_tags)
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    first_state, first_update = _state_recorder()
    first = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "first",
        state_updater=first_update,
    )
    second = tracking.initialize_wandb(config, run_dir=tmp_path / "second")
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "first",
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_epoch=2,
    )
    assert first.run_id != second.run_id
    assert resumed.run_id == first.run_id == first_state["wandb_run_id"]
    assert [item["resume"] for item in fake.initializations] == ["never", "never", "must"]
    assert set(config["tracking"]["wandb"]["tags"]) <= set(fake.runs[2].tags)
    assert set(manual_tags) <= set(fake.runs[2].tags)
    resumed.log_epoch(3, {"train/loss_total": 1.0})
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_epoch(3, {"train/loss_total": 0.5})


def test_offline_mode_needs_no_key_and_preserves_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep offline resume local and bound to its persisted identity."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(mode="offline")
    fresh = tracking.initialize_wandb(config, run_dir=tmp_path)
    state, update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=fresh.run_id,
        state_updater=update,
    )
    assert resumed.run_id == fresh.run_id
    assert all(item["mode"] == "offline" for item in fake.initializations)
    assert all(item["resume"] is None for item in fake.initializations)
    assert state["offline_resume_fallback"]
    assert "api_key" not in str(fake.initializations).lower()


@pytest.mark.parametrize("physics_enabled", [False, True])
def test_semantic_config_is_path_free_and_preserves_active_science(
    physics_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish scientific provenance without runtime paths or secrets."""
    secret = "super-secret-test-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = _resolved_config(physics_enabled=physics_enabled)
    payload = tracking.build_semantic_config(
        config,
        split_indices=_split_evidence(config),
        split_indices_sha256="2" * 64,
        normalizer_sha256="f" * 64,
        checkpoint_identity={"effective_config_digest": "1" * 64},
        model=torch.nn.Linear(2, 3),
        device_metadata={
            "requested_policy": "cpu",
            "resolved_device": "cpu",
            "device_type": "cpu",
            "pytorch_version": torch.__version__,
        },
        duration_contract=experiments.run.RUN_DURATION_CONTRACT,
    )
    assert payload["model"]["parameter_counts"] == {"total": 9, "trainable": 9}
    assert payload["loss"]["physics"]["enabled"] is physics_enabled
    if physics_enabled:
        assert payload["loss"]["physics"]["continuity"] == config["loss"]["physics"]["continuity"]
        assert payload["loss"]["physics"]["residual_weight"] == config["loss"]["physics"]["residual_weight"]
    else:
        assert payload["loss"]["physics"] == {"enabled": False}
    assert payload["data"]["datasets"]["id"]["dataset_id"] == "artificial-train"
    assert payload["data"]["split"]["artifact_sha256"] == "2" * 64
    assert payload["provenance"]["config_digest"] == "1" * 64
    assert {"effective_config", "paths", "tracking"}.isdisjoint(payload)
    serialized = str(payload)
    assert secret not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert str(Path.home()) not in serialized


def test_model_parameter_counts_ignore_device_and_dtype() -> None:
    """Count parameter state independently of placement and precision."""
    model = torch.nn.Linear(2, 3)
    model.bias.requires_grad_(False)
    expected = {"total": 9, "trainable": 6}
    assert tracking.model_parameter_counts(model) == expected
    model.to(device=torch.device("meta"), dtype=torch.float16)
    assert tracking.model_parameter_counts(model) == expected


def test_epoch_history_forwards_supported_metrics_and_selected_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map supported epoch values and retain the selected result."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(epochs=2)
    objective_id = str(config["evaluation"]["objective"]["id"])
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        semantic_config={
            "model": {
                "variant": "fno",
                "parameter_counts": {"total": 10, "trainable": 9},
            },
            "runtime": {"device": {"resolved_device": "cpu"}},
        },
        state_updater=update,
    )
    session.log_epoch(
        1,
        {
            "train/loss_total": 0.75,
            f"id/{objective_id}": 0.5,
            "physics/id/unsupported": 99.0,
        },
    )
    logged, step = fake.runs[0].logs[0]
    assert step == 1
    assert logged["epoch"] == 1
    assert logged["Overview/train_loss_total"] == 0.75
    assert logged[f"Overview/ID/{objective_id}"] == 0.5
    assert 99.0 not in logged.values()
    session.finish(
        status="completed",
        result={
            "completed_epoch": 1,
            "selected_epoch": 1,
            "selected_metrics": {f"selected/id/{objective_id}": 0.5},
        },
    )
    assert fake.runs[0].summary["selected/epoch"] == 1
    assert fake.runs[0].summary[f"selected/id/{objective_id}"] == 0.5
    assert fake.runs[0].summary["tracking/status"] == "finished"
    assert state["last_logged_epoch"] == 1
    assert fake.runs[0].exit_codes == [0]


def test_pi_history_emits_only_configured_continuity_contribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit only the continuity loss selected by the configuration."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(physics_enabled=True)
    session = tracking.initialize_wandb(config, run_dir=tmp_path)
    session.log_epoch(
        1,
        {
            "physics/train/loss_momentum": 1.0,
            "physics/train/loss_boundary": 2.0,
            "physics/train/loss_continuity_div_velocity": 3.0,
            "physics/train/loss_continuity_div_eps_velocity": 4.0,
        },
    )
    logged = fake.runs[0].logs[0][0]
    configured = config["loss"]["physics"]["continuity"]
    other = "div_eps_velocity" if configured == "div_velocity" else "div_velocity"
    assert f"Physics/Training/loss_continuity_{configured}" in logged
    assert f"Physics/Training/loss_continuity_{other}" not in logged


def test_optuna_trial_metadata_reaches_history_and_terminal_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carry trial provenance through history and completion."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    study = "artificial-study"
    config = _resolved_config(workflow="optuna_trial", study=study)
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        semantic_config={
            "tuning": {
                "study_name": study,
                "study_role": "production",
                "trial_number": 7,
                "training_seed": 17,
                "sampler_seed": 23,
                "sampled_parameters": {"model.hidden_channels": 12},
            }
        },
    )
    session.log_epoch(1, {"optuna/objective": 0.72})
    session.finish(status="completed", result={"best_metric": 0.61})
    run = fake.runs[0]
    assert run.logs[0][0]["Optuna/objective"] == 0.72
    assert fake.initializations[0]["group"] == study
    assert run.summary["tuning/trial_number"] == 7
    assert run.summary["tuning/sampled_parameters"] == {"model.hidden_channels": 12}
    assert run.summary["Optuna/objective"] == 0.61
    assert run.summary["Optuna/state"] == "completed"


def test_completed_epoch_physics_is_forwarded_without_epoch_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward supplied physics diagnostics only for completed epochs."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    physics = {"physics/id/momentum_residual_mse": 1.0}
    with pytest.raises(tracking.TrackingError, match="epoch >= 1"):
        session.log_epoch(0, physics)
    session.log_epoch(1, {"train/loss_total": 8.0})
    session.log_epoch(2, physics)
    assert "Physics/ID/momentum_residual_mse" not in fake.runs[0].logs[0][0]
    assert fake.runs[0].logs[1][0]["Physics/ID/momentum_residual_mse"] == 1.0


def test_monitor_membership_is_repeatable_and_bounded() -> None:
    """Reuse a deterministic bounded diagnostic membership."""
    config = _resolved_config()
    config["tracking"]["wandb"]["monitor"]["max_cases"] = 2
    split = _split_evidence(config)
    split["eval_indices"] = torch.tensor([2, 0, 1])
    first = tracking.build_monitor_membership(config, split)
    assert first == tracking.build_monitor_membership(config, split)
    assert first is not None
    assert first["source_indices"] == [2, 0]
    assert first["sample_ids"] == ["sample-c", "sample-a"]


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_requested_history_failures_are_fail_closed(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat requested history loss as a tracking failure."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(mode=mode),
        run_dir=tmp_path,
        state_updater=update,
    )
    fake.runs[0].fail_log = True
    with pytest.raises(tracking.TrackingIOError, match=rf"{mode} W&B history"):
        session.log_epoch(1, {"train/loss_total": 1.0})
    assert state["status"] == "failed"
    assert state["failed_operation"] == "history"


def test_callback_order_preserves_authoritative_consumer_before_observer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the authoritative consumer before a failing observer."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    objective_id = str(config["evaluation"]["objective"]["id"])
    session = tracking.initialize_wandb(config, run_dir=tmp_path)
    fake.runs[0].fail_log = True
    consumed: list[tuple[int, dict[str, float]]] = []

    def consume(epoch: int, values: dict[str, float]) -> None:
        consumed.append((epoch, values))

    callback = tracking.combine_epoch_callbacks(consume, tracking.epoch_callback(session))
    assert callback is not None
    payload = {f"id/{objective_id}": 0.25}
    with pytest.raises(tracking.TrackingIOError):
        callback(5, payload)
    assert consumed == [(5, payload)]
    assert consumed[0][1] is payload
    assert fake.runs[0].logs == []


def test_initialization_and_finish_errors_are_redacted_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redact credentials while recording lifecycle failures."""
    secret = "top-secret-wandb-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    fake = _FakeWandb(init_error=RuntimeError(f"api_key={secret}"))
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    with pytest.raises(tracking.TrackingInitializationError) as captured:
        tracking.initialize_wandb(
            _resolved_config(),
            run_dir=tmp_path,
            state_updater=update,
        )
    assert secret not in str(captured.value)
    assert secret not in str(state)

    finish_fake = _FakeWandb()
    _patch_wandb(monkeypatch, finish_fake)
    finish_state, finish_update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(),
        run_dir=tmp_path,
        state_updater=finish_update,
    )
    finish_fake.runs[0].fail_finish = True
    with pytest.raises(tracking.TrackingIOError, match="finish"):
        session.finish(status="completed")
    assert finish_state["status"] == "failed"
    assert finish_state["failed_operation"] == "finish"


def test_artifact_upload_enforces_enablement_allowlist_and_root_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restrict uploads to enabled, bounded run artifacts."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    analysis_root = tmp_path / "analysis" / "id"
    analysis_root.mkdir(parents=True)
    provenance = analysis_root / "artifact_provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")

    disabled = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    with pytest.raises(tracking.TrackingUploadError, match="disabled"):
        disabled.upload_files({"artifact_provenance": provenance})

    session = tracking.initialize_wandb(_resolved_config(upload=True), run_dir=tmp_path)
    with pytest.raises(tracking.TrackingUploadError, match="Unsupported"):
        session.upload_files({"arbitrary": provenance})
    outside = tmp_path / "outside" / "artifact_provenance.json"
    outside.parent.mkdir()
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(tracking.TrackingUploadError, match="analysis root"):
        session.upload_files({"artifact_provenance": outside})
    session.upload_files({"artifact_provenance": provenance})
    assert Path(fake.runs[1].saved[0][0]) == provenance

    media: dict[str, Path] = {}
    for name in tracking.POST_ARTIFACT_MEDIA_KEYS - {"run_summary_table"}:
        rendered = tmp_path / "rendered" / f"{name}.png"
        rendered.parent.mkdir(exist_ok=True)
        rendered.write_bytes(b"rendered")
        media[name] = rendered
    session.upload_post_artifact(
        artifact_root=analysis_root,
        media_files=media,
        tables={"run_summary_table": {"columns": ["score"], "data": [[0.25]]}},
    )
    assert len(fake.runs[1].artifacts) == 1


def test_persisted_identity_requires_one_run_id_and_uses_latest_epoch() -> None:
    """Recover one run identity and its latest logged epoch."""
    summary = {
        "runtime_sessions": [
            {"tracking": {"wandb_run_id": "same", "last_logged_epoch": 2}},
            {"tracking": {"wandb_run_id": "same", "last_logged_epoch": 5}},
        ]
    }
    assert tracking.persisted_wandb_identity(summary) == ("same", 5)
    summary["runtime_sessions"].append({"tracking": {"wandb_run_id": "different"}})
    with pytest.raises(tracking.TrackingInitializationError, match="found 2"):
        tracking.persisted_wandb_identity(summary)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"mode": "invalid"}, r"tracking\.wandb\.mode"),
        ({"monitor": {"interval": 0}}, r"tracking\.wandb\.monitor\.interval"),
        (
            {"upload": {"evaluation_artifacts": 1}},
            r"tracking\.wandb\.upload\.evaluation_artifacts",
        ),
        ({"workflow": "optuna_trial"}, r"tracking\.wandb\.study is required"),
        (
            {"workflow": "train", "study": "not-allowed"},
            r"tracking\.wandb\.study is valid only",
        ),
    ],
)
def test_wandb_config_rejects_invalid_values(
    settings: dict[str, Any],
    match: str,
) -> None:
    """Reject malformed tracking settings through the public resolver."""
    raw = configs.direct_config()
    raw["tracking"] = {"wandb": settings}
    with pytest.raises(ValueError, match=match):
        experiments.config.loader.resolve_config(raw)
