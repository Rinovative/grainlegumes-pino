# ruff: noqa: EM101, PLR2004, S101, S105, TRY003
"""Protect the canonical W&B identity, metric, failure, and upload contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from src import experiments, learning
from support import configs

tracking = experiments.tracking
_EXPERIMENT_CONFIGS = configs.experiment_config_paths()
_CONFIG_PATH = configs.acceptance_config_path()
_PI_FNO_CONFIG_PATH = configs.experiment_config_path(
    model_kind="fno",
    physics_enabled=True,
)
_GPU_CONFIG_PATH = configs.acceptance_config_path()


class _FakeArtifact:
    """Record one explicit curated artifact without SDK serialization."""

    def __init__(self, **metadata: Any) -> None:
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []
        self.tables: list[tuple[Any, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))

    def add(self, value: Any, name: str) -> None:
        self.tables.append((value, name))


class _FakeRun:
    """Record the exact W&B run surface and inject bounded failures."""

    def __init__(self, *, tags: tuple[str, ...] = ()) -> None:
        self.summary: dict[str, Any] = {}
        self.tags: tuple[str, ...] = tags
        self.metric_definitions: list[tuple[str, dict[str, Any]]] = []
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.exit_codes: list[int] = []
        self.saved: list[tuple[str, str, str]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.fail_log = False
        self.fail_finish = False
        self.fail_save = False

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
        if self.fail_save:
            raise OSError("synthetic upload failure")
        self.saved.append((path, base_path, policy))

    def log_artifact(
        self,
        artifact: _FakeArtifact,
        *,
        aliases: list[str],
    ) -> None:
        self.artifacts.append((artifact, aliases))


class _FakeWandb:
    """Create isolated fake runs and capture initialization settings."""

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
        configured_tags = tuple(settings.get("tags") or ())
        tags = self.resumed_tags if settings.get("resume") == "must" and self.resumed_tags is not None else configured_tags
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
def _noninteractive_test_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give explicitly fake online sessions a non-empty test-only key."""
    monkeypatch.setenv("WANDB_API_KEY", "test-only-wandb-key")


def _resolved_config(
    *,
    mode: str = "online",
    workflow: str = "train",
    study: str | None = None,
    epochs: int | None = None,
    upload: bool = False,
    config_path: Path = _CONFIG_PATH,
) -> dict[str, Any]:
    """Resolve one experiment through the public tracking schema."""
    raw = experiments.config.loader.load_yaml(config_path)
    wandb: dict[str, Any] = {
        "mode": mode,
        "workflow": workflow,
        "upload": {"evaluation_artifacts": upload},
    }
    if study is not None:
        wandb["study"] = study
    raw["tracking"] = {"wandb": wandb}
    if epochs is not None:
        raw["training"]["epochs"] = epochs
    return experiments.config.loader.resolve_config(raw)


def _expected_variant(config: dict[str, Any]) -> str:
    """Derive the public model taxonomy from resolved model and loss semantics."""
    model_kind = config["model"]["kind"]
    if config["loss"]["physics"]["enabled"]:
        return f"pi-{model_kind}"
    return str(model_kind)


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
    """Return a merged local tracking-state recorder."""
    state: dict[str, Any] = {}

    def update(values: dict[str, Any]) -> None:
        state.update(values)

    return state, update


def test_airflow_and_drying_projects_are_explicitly_separate() -> None:
    """Reserve distinct comparison universes while this repository owns airflow."""
    projects = experiments.config.defaults.WANDB_REPOSITORY_PROJECTS
    assert projects == {
        "airflow": "grainlegumes-pino-airflow",
        "drying": "grainlegumes-pino-drying",
    }
    assert projects["airflow"] != projects["drying"]
    assert _resolved_config()["tracking"]["wandb"]["project"] == projects["airflow"]


def test_normal_recipes_explicitly_select_semantic_taxonomy_and_tracking() -> None:
    """Validate every discovered recipe without freezing filenames or run names."""
    for config_path in _EXPERIMENT_CONFIGS:
        raw = experiments.config.loader.load_yaml(config_path)
        assert raw["tracking"]["wandb"]["mode"] == "online"
        assert raw["tracking"]["wandb"]["workflow"] == "train"
        config = experiments.config.loader.resolve_config(raw)
        variant = _expected_variant(config)
        settings = config["tracking"]["wandb"]

        assert settings["mode"] == "online"
        assert settings["workflow"] == "train"
        assert settings["study"] is None
        assert settings["project"] == "grainlegumes-pino-airflow"
        assert settings["entity"] == "Rinovative-Hub"
        assert settings["tags"] == [variant]
        assert settings["monitor"]["enabled"] is True
        assert settings["monitor"]["interval"] == config["training"]["evaluation_interval"]
        assert settings["monitor"]["interval"] == config["training"]["ood_evaluation_interval"]
        assert settings["monitor"]["max_cases"] >= 1
        assert settings["upload"]["evaluation_artifacts"] is False
        assert not any(tag.startswith("arch:") for tag in settings["tags"])
        assert "final" not in settings["tags"]
        assert config["run"]["name"]
        assert config["model"]["params"]


def test_gpu_smoke_tracking_override_is_reachable_and_consumed() -> None:
    """Preserve acceptance workflow ownership without freezing its mutable budget."""
    raw = experiments.config.loader.load_yaml(_GPU_CONFIG_PATH)
    assert raw["tracking"]["wandb"]["mode"] == "online"
    assert raw["tracking"]["wandb"]["workflow"] == "gpu_smoke"
    config = experiments.config.loader.resolve_config(raw)
    settings = config["tracking"]["wandb"]
    cadence = config["training"]["evaluation_interval"]

    assert settings["mode"] == "online"
    assert settings["workflow"] == "gpu_smoke"
    assert settings["study"] is None
    assert settings["project"] == "grainlegumes-pino-airflow"
    assert settings["entity"] == "Rinovative-Hub"
    assert settings["tags"] == []
    assert settings["monitor"]["enabled"] is True
    assert settings["monitor"]["interval"] == cadence
    assert settings["monitor"]["max_cases"] >= 1
    assert settings["upload"]["evaluation_artifacts"] is False
    assert cadence <= config["training"]["epochs"]
    assert config["training"]["ood_evaluation_interval"] == cadence
    assert experiments.run.initial_tracking_state(config)["workflow"] == "gpu_smoke"


def test_workflow_taxonomies_and_derived_organization_are_strict() -> None:
    """Keep trial tags semantic, validation tags empty, and organization derived."""
    for config_path in _EXPERIMENT_CONFIGS:
        base = experiments.config.loader.load_and_resolve_config(config_path)
        variant = _expected_variant(base)
        study = f"{variant}-study"
        settings = _resolved_config(
            workflow="optuna_trial",
            study=study,
            config_path=config_path,
        )["tracking"]["wandb"]
        assert settings["mode"] == "online"
        assert settings["workflow"] == "optuna_trial"
        assert settings["study"] == study
        assert settings["project"] == "grainlegumes-pino-airflow"
        assert settings["entity"] == "Rinovative-Hub"
        assert settings["tags"] == [variant, "optuna"]
        assert settings["monitor"]["enabled"] is True
        assert settings["monitor"]["interval"] == base["training"]["evaluation_interval"]
        assert settings["monitor"]["max_cases"] >= 1
        assert settings["upload"]["evaluation_artifacts"] is False

    for workflow in ("gpu_smoke", "cpu_acceptance", "tracking_validation"):
        settings = _resolved_config(workflow=workflow)["tracking"]["wandb"]
        assert settings["tags"] == []

    for key in ("project", "entity", "tags"):
        raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
        raw["tracking"] = {"wandb": {key: "override"}}
        with pytest.raises(ValueError, match=rf"tracking\.wandb.*{key}"):
            experiments.config.loader.resolve_config(raw)


def test_disabled_wandb_has_no_sdk_or_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return before SDK import, identity allocation, or local writes."""

    def fail_import(_name: str) -> None:
        pytest.fail("disabled W&B must not import the SDK")

    config = _resolved_config(mode="disabled")
    monkeypatch.setattr(tracking.importlib, "import_module", fail_import)
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        state_updater=update,
    )
    assert not session.enabled
    assert session.run_id is None
    assert tracking.epoch_callback(session) is None
    session.log_epoch(1, {"train/loss_total": 2.0})
    session.finish(status="completed", result={"best_metric": 1.0})
    assert state == {}
    assert list(tmp_path.iterdir()) == []


def test_online_authentication_is_checked_before_sdk_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject blank online authentication without printing or importing a key."""
    monkeypatch.delenv("WANDB_API_KEY")
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    with pytest.raises(
        tracking.TrackingInitializationError,
        match="non-interactive environment authentication",
    ):
        tracking.initialize_wandb(
            _resolved_config(),
            run_dir=tmp_path,
            state_updater=update,
        )
    assert fake.initializations == []
    assert state["status"] == "failed_before_start"
    assert state["failed_operation"] == "authentication"


def test_online_initialization_has_no_workspace_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Initialize a fake online run without workspace calls, URLs, or warnings."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    captured = capsys.readouterr()
    assert session.enabled
    assert len(fake.initializations) == 1
    assert captured.out == ""
    assert captured.err == ""
    assert list(recwarn) == []


def test_fresh_ids_are_opaque_and_online_resume_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep equal display names isolated while exact resume reuses one ID."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    semantic_config = {
        "model": {
            "variant": "fno",
            "parameter_counts": {"total": 9, "trainable": 6},
        }
    }
    first_state, first_update = _state_recorder()
    second_state, second_update = _state_recorder()
    first = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "one",
        semantic_config=semantic_config,
        state_updater=first_update,
    )
    second = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "two",
        semantic_config=semantic_config,
        state_updater=second_update,
    )
    resumed_state, resumed_update = _state_recorder()
    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path / "one",
        resume=True,
        persisted_run_id=first.run_id,
        previous_last_logged_epoch=5,
        semantic_config=semantic_config,
        state_updater=resumed_update,
    )
    assert first.run_id != second.run_id
    assert len(str(first.run_id)) == 32
    assert first_state["wandb_run_id"] == first.run_id
    assert second_state["wandb_run_id"] == second.run_id
    assert resumed.run_id == first.run_id
    assert resumed_state["session_kind"] == "resume"
    assert [item["resume"] for item in fake.initializations] == [
        "never",
        "never",
        "must",
    ]
    for key in ("project", "entity", "name", "config"):
        assert fake.initializations[0][key] == fake.initializations[2][key]
    assert fake.initializations[0]["tags"] == ["fno"]
    assert fake.initializations[2]["tags"] is None
    assert fake.initializations[2]["config"]["model"]["parameter_counts"] == {
        "total": 9,
        "trainable": 6,
    }
    resumed.log_epoch(6, {"train/loss_total": 1.0, "global_step": 7.0})
    with pytest.raises(tracking.TrackingError, match="cannot rewrite"):
        resumed.log_epoch(6, {"train/loss_total": 1.0})


def test_online_resume_preserves_manual_tags_and_restores_required_base_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Union repository tags with remote UI tags without adding ``final`` itself."""
    fake = _FakeWandb(resumed_tags=("final", "reviewed-manually"))
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config()
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id="persisted-run-id",
    )
    assert fake.initializations[0]["tags"] is None
    assert fake.runs[0].tags == ("final", "reviewed-manually", "fno")
    assert config["tracking"]["wandb"]["tags"] == ["fno"]
    assert "final" not in str(config)
    assert session.run_id == "persisted-run-id"


def test_real_sdk_offline_smoke_is_local_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialize, log, and finish one real offline SDK run without credentials."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    monkeypatch.setenv("WANDB_DISABLE_GIT", "true")
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "data"))
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(mode="offline", epochs=1),
        run_dir=tmp_path,
        semantic_config={"runtime": {"device": {"resolved_device": "cpu"}}},
        state_updater=update,
    )
    session.log_epoch(1, {"train/loss_total": 1.0, "train/loss_data": 0.9})
    session.finish(status="completed", result={"completed_epoch": 1})
    assert session.enabled
    assert state["requested_mode"] == "offline"
    assert state["last_logged_epoch"] == 1
    assert state["status"] == "finished"
    assert list((tmp_path / "wandb").glob("offline-run-*"))


def test_offline_mode_needs_no_key_and_keeps_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the persisted ID for an explicit offline resume segment."""
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
    assert fresh.run_id == resumed.run_id
    assert [item["mode"] for item in fake.initializations] == [
        "offline",
        "offline",
    ]
    assert [item["resume"] for item in fake.initializations] == [None, None]
    assert "same_persisted_id_segment" in state["offline_resume_fallback"]
    assert "api_key" not in str(fake.initializations).lower()


def _split_evidence(config: dict[str, Any]) -> dict[str, Any]:
    sample_ids = ["case_0001", "case_0002", "case_0003"]
    return {
        "schema_version": 1,
        "eval_indices": torch.tensor([2, 0]),
        "metadata": {
            "datasets": {
                role: {
                    "dataset_id": f"{role}-data",
                    "fingerprint": marker * 64,
                    "sample_count": 3,
                    "spatial_shape": [8, 8],
                    "task_contract_digest": config["task_contract"]["digest"],
                    "sample_ids": sample_ids,
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


def test_semantic_config_is_complete_path_free_and_nonduplicative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish one nested scientific config without paths, tensors, or secrets."""
    secret = "super-secret-test-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = _resolved_config()
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
    assert payload["task"]["id"] == "steady_flow"
    assert payload["task"]["contract"]["schema_version"] == 1
    assert payload["model"]["variant"] == "fno"
    assert payload["model"]["parameter_counts"] == {"total": 9, "trainable": 9}
    assert payload["runtime"]["duration_contract"] == experiments.run.RUN_DURATION_CONTRACT
    assert payload["evaluation"]["objective"]["id"] == "normalized_macro_rmse"
    assert payload["evaluation"]["roles"] == {
        "selection": "id",
        "diagnostic": ["ood", "physics"],
        "event_model": "completed_epoch_interval_or_terminal",
        "id_interval_epochs": config["training"]["evaluation_interval"],
        "ood_interval_epochs": config["training"]["ood_evaluation_interval"],
        "physics_interval_epochs": config["tracking"]["wandb"]["monitor"]["interval"],
        "epoch_zero_evaluation": False,
    }
    assert payload["loss"]["data"]["kind"] == "relative_h1"
    assert payload["loss"]["physics"] == {"enabled": False}
    assert payload["diagnostics"]["physics_monitor"] == {
        "enabled": True,
        "role": "id",
        "membership": "bounded_saved_evaluation_prefix",
        "interval_epochs": config["tracking"]["wandb"]["monitor"]["interval"],
        "max_cases": config["tracking"]["wandb"]["monitor"]["max_cases"],
        "physics_kind": "steady_2d_brinkman",
        "equation_set": "steady_two_dimensional_brinkman",
        "continuity_forms": ["div_velocity", "div_eps_velocity"],
        "boundary": "pressure_inlet_zero_pressure_outlet",
        "derivatives": {"kind": "spectral", "extension": "reflect"},
        "interior_crop": 2,
        "metric_ids": [
            "momentum_residual_mse",
            "continuity_div_velocity_mse",
            "continuity_div_eps_velocity_mse",
            "pressure_boundary_mse",
        ],
    }
    assert payload["data"]["datasets"]["id"]["dataset_id"] == "train-data"
    assert payload["data"]["split"]["membership_digests"]["eval"] == "d" * 64
    assert payload["data"]["split"]["artifact_sha256"] == "2" * 64
    assert payload["provenance"]["config_digest"] == "1" * 64
    assert payload["provenance"]["schema_versions"]["tracking_integration"] == 3
    assert "effective_config" not in payload
    assert "paths" not in payload
    assert "tracking" not in payload
    serialized = str(payload)
    assert secret not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert str(Path.home()) not in serialized


def test_pi_semantic_config_retains_active_optimization_physics() -> None:
    """Retain full PI settings only when physics participates in optimization."""
    config = _resolved_config(config_path=_PI_FNO_CONFIG_PATH)
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
    assert payload["loss"]["physics"]["enabled"] is True
    assert payload["loss"]["physics"]["residual_weight"] == config["loss"]["physics"]["residual_weight"]
    assert payload["loss"]["physics"]["boundary_weight"] == config["loss"]["physics"]["boundary_weight"]
    assert payload["diagnostics"]["physics_monitor"]["enabled"] is True


def test_model_parameter_counts_use_parameter_state_not_runtime_placement() -> None:
    """Count a small module across trainability, device, and dtype transitions."""
    model = torch.nn.Linear(2, 3)
    model.bias.requires_grad_(False)
    cpu_counts = tracking.model_parameter_counts(model)

    assert cpu_counts == {"total": 9, "trainable": 6}
    assert all(type(value) is int for value in cpu_counts.values())
    model.to(device=torch.device("meta"), dtype=torch.float16)
    assert tracking.model_parameter_counts(model) == cpu_counts


def test_epoch_history_uses_exact_namespaces_and_selected_terminal_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the complete automatic hierarchy while keeping selected values summary-only."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(epochs=5)
    state, update = _state_recorder()
    monitor_values = {
        "physics/id/momentum_residual_mse": 0.04,
        "physics/id/continuity_div_velocity_mse": 0.03,
        "physics/id/continuity_div_eps_velocity_mse": 0.02,
        "physics/id/pressure_boundary_mse": 0.01,
    }
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        semantic_config={
            "task": {"id": "steady_flow", "contract_digest": "t" * 64},
            "data": {
                "datasets": {
                    "id": {"dataset_id": "steady_train", "fingerprint": "d" * 64},
                    "ood": {"dataset_id": "steady_ood", "fingerprint": "o" * 64},
                },
                "split": {"artifact_sha256": "c" * 64},
                "normalization": {"artifact_sha256": "b" * 64},
            },
            "model": {"variant": "fno", "parameter_counts": {"total": 10, "trainable": 9}},
            "provenance": {"config_digest": "a" * 64},
            "runtime": {"device": {"resolved_device": "cpu"}},
        },
        state_updater=update,
    )
    source_accuracy = {
        f"{role}/{metric_id}": float(index)
        for index, (role, metric_id) in enumerate(
            ((role, metric_id) for role in ("id", "ood") for metric_id in tracking.ACCURACY_HISTORY_METRIC_IDS),
            start=1,
        )
    }
    callback = tracking.epoch_callback(session)
    assert callback is not None
    callback(
        5,
        {
            "train/loss_total": 0.75,
            "train/loss_data": 0.7,
            "optimization/learning_rate": 0.125,
            "system/epoch_duration_seconds": 1.5,
            "system/train_duration_seconds": 0.75,
            "system/train_samples_per_second": 8.0,
            "global_step": 12.0,
            "id/normalized_macro_rmse": 0.7,
            "ood/normalized_macro_rmse": 1.0,
            "generalization/objective_gap": 0.3,
            "id/normalized_rmse": 91.0,
            "ood/normalized_rmse": 92.0,
            "objective/value": 99.0,
            **source_accuracy,
            **monitor_values,
        },
    )
    selected_science_metrics = {
        "selected/id/normalized_macro_rmse": 0.5,
        "selected/ood/normalized_macro_rmse": 0.8,
        "selected/id/normalized_rmse": 0.51,
        "selected/ood/normalized_rmse": 0.81,
        "selected/id/normalized_rmse_p": 0.52,
        "selected/ood/normalized_rmse_p": 0.82,
        "selected/id/normalized_rmse_u": 0.53,
        "selected/ood/normalized_rmse_u": 0.83,
        "selected/id/normalized_rmse_v": 0.54,
        "selected/ood/normalized_rmse_v": 0.84,
        "selected/id/normalized_relative_l2": 0.55,
        "selected/ood/normalized_relative_l2": 0.85,
        "selected/id/normalized_relative_h1": 0.56,
        "selected/ood/normalized_relative_h1": 0.86,
        "selected/id/physical_rmse_p": 0.57,
        "selected/ood/physical_rmse_p": 0.87,
        "selected/id/physical_rmse_u": 0.58,
        "selected/ood/physical_rmse_u": 0.88,
        "selected/id/physical_rmse_v": 0.59,
        "selected/ood/physical_rmse_v": 0.89,
    }
    selected_metrics = {
        **selected_science_metrics,
        "selected/generalization/objective_gap": 0.3,
        "selected/physics/momentum_residual_mse": 0.04,
        "selected/physics/continuity_div_velocity_mse": 0.03,
        "selected/physics/continuity_div_eps_velocity_mse": 0.02,
        "selected/physics/pressure_boundary_mse": 0.01,
        "selected/training/residual_weight": 0.00008,
        "selected/training/boundary_weight": 0.0004,
    }
    session.finish(
        status="completed",
        result={
            "completed_epoch": 5,
            "global_step": 12,
            "selected_epoch": 4,
            "selected_metrics": selected_metrics,
            "terminal_epoch": 5,
            "terminal_metrics": {
                "train/loss_total": 0.75,
                "train/loss_data": 0.7,
            },
        },
        local_summary={
            "effective_config_digest": "a" * 64,
            "normalizer_sha256": "b" * 64,
            "split_indices_sha256": "c" * 64,
            "best_checkpoint_sha256": "9" * 64,
            "elapsed_seconds": 4.25,
            "runtime_sessions": [
                {"tracking": {"session_kind": "fresh"}},
                {"tracking": {"session_kind": "resume"}},
            ],
        },
    )
    run = fake.runs[0]
    logged, step = run.logs[0]
    assert step == 5
    expected_logged: dict[str, float | int] = {
        "epoch": 5,
        "Overview/ID/normalized_macro_rmse": 0.7,
        "Overview/OOD/normalized_macro_rmse": 1.0,
        "Overview/generalization_gap": 0.3,
        "Overview/train_loss_total": 0.75,
        "Overview/train_loss_data": 0.7,
        "Overview/learning_rate": 0.125,
        "Diagnostics/epoch_duration_seconds": 1.5,
        "Diagnostics/train_duration_seconds": 0.75,
        "Diagnostics/train_samples_per_second": 8.0,
    }
    expected_logged.update(
        {
            f"Accuracy/{role.upper()}/{metric_id}": value
            for (role, metric_id), value in (
                (source_key.split("/", maxsplit=1), source_value) for source_key, source_value in source_accuracy.items()
            )
        }
    )
    expected_logged.update({f"Physics/ID/{key.removeprefix('physics/id/')}": value for key, value in monitor_values.items()})
    assert logged == expected_logged
    assert not any(key.startswith(("id/", "ood/", "train/", "physics/", "system/", "optimization/", "generalization/")) for key in logged)
    assert "Accuracy/ID/normalized_rmse" not in logged
    assert "Accuracy/OOD/normalized_rmse" not in logged
    assert not any(key.startswith("selected/") for key in logged)

    expected_definitions = [
        "epoch",
        "Overview/ID/normalized_macro_rmse",
        "Overview/OOD/normalized_macro_rmse",
        "Overview/generalization_gap",
        "Overview/train_loss_total",
        "Overview/train_loss_data",
        "Overview/learning_rate",
        *(f"Accuracy/ID/{metric_id}" for metric_id in tracking.ACCURACY_HISTORY_METRIC_IDS),
        *(f"Accuracy/OOD/{metric_id}" for metric_id in tracking.ACCURACY_HISTORY_METRIC_IDS),
        "Physics/ID/momentum_residual_mse",
        "Physics/ID/continuity_div_velocity_mse",
        "Physics/ID/continuity_div_eps_velocity_mse",
        "Physics/ID/pressure_boundary_mse",
        "Diagnostics/epoch_duration_seconds",
        "Diagnostics/train_duration_seconds",
        "Diagnostics/train_samples_per_second",
    ]
    assert [name for name, _kwargs in run.metric_definitions] == expected_definitions
    assert run.metric_definitions[0][1] == {"hidden": True, "summary": "none"}
    assert all(kwargs == {"step_metric": "epoch", "step_sync": False, "summary": "none"} for _name, kwargs in run.metric_definitions[1:])
    assert tracking.AUTOMATIC_HISTORY_TOP_LEVEL_PREFIXES == (
        "Overview",
        "Accuracy",
        "Physics",
        "Diagnostics",
        "Optuna",
    )
    assert all(definition.owner for definition in session.history_metric_definitions)
    assert all(definition.computation_cost for definition in session.history_metric_definitions)
    assert all(definition.scientific_question for definition in session.history_metric_definitions)
    accuracy_names = [name for name in expected_definitions if name.startswith("Accuracy/")]
    assert max(index for index, name in enumerate(accuracy_names) if name.startswith("Accuracy/ID/")) < min(
        index for index, name in enumerate(accuracy_names) if name.startswith("Accuracy/OOD/")
    )
    assert not any(name.startswith("Physics/Training/") for name in expected_definitions)
    assert "Diagnostics/cuda_peak_memory_allocated_bytes" not in expected_definitions

    initialization = fake.initializations[0]
    assert initialization["name"] == config["run"]["name"] == session.run_name
    assert "_disable_stats" not in initialization["settings"]
    assert initialization["tags"] == ["fno"]
    assert run.saved == []
    assert run.summary["selected/epoch"] == 4
    assert run.summary["terminal/epoch"] == 5
    assert run.summary["selected/id/normalized_macro_rmse"] == 0.5
    assert run.summary["selected/ood/normalized_macro_rmse"] == 0.8
    for key, value in selected_science_metrics.items():
        assert run.summary[key] == value
    assert run.summary["selected/physics/continuity_div_velocity_mse"] == 0.03
    assert run.summary["terminal/train/loss_total"] == 0.75
    assert run.summary["model/variant"] == "fno"
    assert run.summary["model/parameters_total"] == 10
    assert run.summary["model/parameters_trainable"] == 9
    assert initialization["config"]["model"]["parameter_counts"]["total"] == run.summary["model/parameters_total"]
    assert initialization["config"]["model"]["parameter_counts"]["trainable"] == run.summary["model/parameters_trainable"]
    assert run.summary["run/duration_seconds"] == 4.25
    assert run.summary["run/completed_epoch"] == 5
    assert run.summary["run/global_step"] == 12
    assert run.summary["run/resume_count"] == 1
    assert run.summary["selected/checkpoint_sha256_short"] == "9" * 16
    assert run.summary["tracking/status"] == "finished"
    assert state["last_logged_epoch"] == 5
    assert state["status"] == "finished"
    assert run.exit_codes == [0]


def test_pi_history_uses_only_the_configured_continuity_contribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit PI components and weights without fabricating the other continuity loss."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(config_path=_PI_FNO_CONFIG_PATH)
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        semantic_config={"runtime": {"device": {"resolved_device": "cpu"}}},
    )
    session.log_epoch(
        1,
        {
            "physics/train/loss_momentum": 1.0,
            "physics/train/loss_boundary": 2.0,
            "physics/train/loss_continuity_div_velocity": 3.0,
            "physics/train/loss_continuity_div_eps_velocity": 4.0,
            "physics/train/residual_weight": 5.0,
            "physics/train/boundary_weight": 6.0,
        },
    )

    continuity = config["loss"]["physics"]["continuity"]
    continuity_metric = f"Physics/Training/loss_continuity_{continuity}"
    continuity_value = 3.0 if continuity == "div_velocity" else 4.0
    training_definitions = [name for name, _kwargs in fake.runs[0].metric_definitions if name.startswith("Physics/Training/")]
    assert training_definitions == [
        "Physics/Training/loss_momentum",
        "Physics/Training/loss_boundary",
        continuity_metric,
        "Physics/Training/residual_weight",
        "Physics/Training/boundary_weight",
    ]
    assert fake.runs[0].logs == [
        (
            {
                "epoch": 1,
                "Physics/Training/loss_momentum": 1.0,
                "Physics/Training/loss_boundary": 2.0,
                continuity_metric: continuity_value,
                "Physics/Training/residual_weight": 5.0,
                "Physics/Training/boundary_weight": 6.0,
            },
            1,
        )
    ]


def test_cuda_peak_diagnostic_is_registered_and_emitted_only_for_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the resolved runtime device, not hardware policy guesses, for CUDA history."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(
        workflow="gpu_smoke",
        config_path=_GPU_CONFIG_PATH,
    )
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        semantic_config={"runtime": {"device": {"resolved_device": "cuda:0"}}},
    )
    session.log_epoch(
        1,
        {
            "system/epoch_duration_seconds": 2.0,
            "system/train_duration_seconds": 1.25,
            "system/train_samples_per_second": 16.0,
            "system/cuda_peak_memory_allocated_bytes": 4096.0,
        },
    )
    assert "Diagnostics/cuda_peak_memory_allocated_bytes" in {name for name, _kwargs in fake.runs[0].metric_definitions}
    assert fake.runs[0].logs == [
        (
            {
                "epoch": 1,
                "Diagnostics/epoch_duration_seconds": 2.0,
                "Diagnostics/train_duration_seconds": 1.25,
                "Diagnostics/train_samples_per_second": 16.0,
                "Diagnostics/cuda_peak_memory_allocated_bytes": 4096.0,
            },
            1,
        )
    ]


def test_optuna_history_and_terminal_summary_preserve_trial_comparison_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror exact Optuna reports and retain selected objective plus seed metadata."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    training_seed = 17
    sampler_seed = 23
    session = tracking.initialize_wandb(
        _resolved_config(workflow="optuna_trial", study="steady_flow_fno_search"),
        run_dir=tmp_path,
        semantic_config={
            "tuning": {
                "study_name": "steady_flow_fno_search",
                "study_role": "production",
                "trial_number": 7,
                "training_seed": training_seed,
                "sampler_seed": sampler_seed,
                "search_signature": "search-signature",
                "sampled_parameters": {"model.hidden_channels": 48},
            }
        },
    )
    session.log_epoch(
        5,
        {
            "id/normalized_macro_rmse": 0.72,
            "optuna/objective": 0.72,
            "optuna/best_objective_so_far": 0.72,
        },
    )
    session.finish(
        status="completed",
        result={
            "best_metric": 0.61,
            "selected_epoch": 2,
            "selected_metrics": {
                "selected/id/normalized_macro_rmse": 0.61,
                "selected/ood/normalized_macro_rmse": 0.79,
            },
        },
        local_summary={"elapsed_seconds": 12.5, "best_metric": 0.61},
    )

    run = fake.runs[0]
    assert run.logs == [
        (
            {
                "epoch": 5,
                "Overview/ID/normalized_macro_rmse": 0.72,
                "Optuna/objective": 0.72,
                "Optuna/best_objective_so_far": 0.72,
            },
            5,
        )
    ]
    assert [name for name, _settings in run.metric_definitions if name.startswith("Optuna/")] == [
        "Optuna/objective",
        "Optuna/best_objective_so_far",
    ]
    initialization = fake.initializations[0]
    assert initialization["group"] == "steady_flow_fno_search"
    assert initialization["job_type"] == "optuna_trial"
    assert initialization["tags"] == ["fno", "optuna-production"]
    assert run.summary["tuning/study_name"] == "steady_flow_fno_search"
    assert run.summary["tuning/study_role"] == "production"
    assert run.summary["tuning/trial_number"] == 7
    assert run.summary["tuning/training_seed"] == training_seed
    assert run.summary["tuning/sampler_seed"] == sampler_seed
    assert run.summary["tuning/search_signature"] == "search-signature"
    assert run.summary["tuning/sampled_parameters"] == {"model.hidden_channels": 48}
    assert run.summary["tuning/final_state"] == "completed"
    assert run.summary["Optuna/trial_number"] == 7
    assert run.summary["Optuna/state"] == "completed"
    assert run.summary["Optuna/pruned"] is False
    assert run.summary["Optuna/objective"] == 0.61
    assert run.summary["Optuna/trial_duration_seconds"] == 12.5
    assert run.summary["selected/id/normalized_macro_rmse"] == 0.61
    assert run.summary["selected/ood/normalized_macro_rmse"] == 0.79
    assert run.summary["selected/epoch"] == 2


def test_tracking_consumes_authoritative_completed_epoch_physics_without_epoch_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map supplied physics metrics without owning their evaluation or cadence."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(epochs=3)
    physics = {
        "physics/id/momentum_residual_mse": 1.0,
        "physics/id/continuity_div_velocity_mse": 2.0,
        "physics/id/continuity_div_eps_velocity_mse": 3.0,
        "physics/id/pressure_boundary_mse": 4.0,
    }
    session = tracking.initialize_wandb(config, run_dir=tmp_path)
    with pytest.raises(tracking.TrackingError, match="epoch >= 1"):
        session.log_epoch(0, physics)
    session.log_epoch(1, {"train/loss_total": 8.0})
    session.log_epoch(2, {"train/loss_total": 7.0, **physics})

    assert [step for _payload, step in fake.runs[0].logs] == [1, 2]
    assert "Physics/ID/momentum_residual_mse" not in fake.runs[0].logs[0][0]
    assert fake.runs[0].logs[1][0]["Physics/ID/momentum_residual_mse"] == 1.0

    resumed = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        resume=True,
        persisted_run_id=session.run_id,
        previous_last_logged_epoch=2,
    )
    resumed.log_epoch(3, {"train/loss_total": 6.0})
    assert fake.runs[1].logs == [({"epoch": 3, "Overview/train_loss_total": 6.0}, 3)]


def test_monitor_membership_is_fixed_and_bounded() -> None:
    """Reuse one deterministic saved ID prefix for training-owned diagnostics."""
    config = _resolved_config(epochs=3)
    config["tracking"]["wandb"]["monitor"]["max_cases"] = 2
    split = _split_evidence(config)
    split["eval_indices"] = torch.tensor([2, 0, 1])
    first = tracking.build_monitor_membership(config, split)
    second = tracking.build_monitor_membership(config, split)
    assert first == second
    assert first is not None
    assert first["source_indices"] == [2, 0]
    assert first["sample_ids"] == ["case_0003", "case_0001"]


def test_gpu_smoke_declares_reachable_dense_completed_epoch_evaluations() -> None:
    """Keep acceptance evaluation dense, terminal-inclusive, and free of epoch zero."""
    config = experiments.config.loader.load_and_resolve_config(_GPU_CONFIG_PATH)
    target_epoch = config["training"]["epochs"]
    cadence = config["training"]["evaluation_interval"]
    assert config["training"]["ood_evaluation_interval"] == cadence
    assert config["tracking"]["wandb"]["monitor"]["interval"] == cadence
    events = learning.training.events.completed_epoch_events(interval=cadence, target_epoch=target_epoch)
    assert events == tuple(range(1, target_epoch + 1))
    assert 0 not in events
    assert events[-1] == target_epoch


def test_unmapped_source_metric_is_ignored_without_observer_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore unsupported source keys while logging the declared metric surface."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(epochs=1), run_dir=tmp_path)
    session.log_epoch(
        1,
        {
            "train/loss_total": 1.0,
            "physics/id/unsupported": 2.0,
        },
    )
    assert fake.runs[0].logs == [({"epoch": 1, "Overview/train_loss_total": 1.0}, 1)]


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_requested_history_failures_are_fail_closed(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface the same write loss in online and offline modes."""
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
    fake.runs[0].fail_log = False
    session.finish(status="failed", error="history failure")
    assert fake.runs[0].summary["tracking/status"] == "failed"
    assert state["status"] == "failed"


def test_combined_console_and_wandb_consumers_receive_one_authoritative_payload() -> None:
    """Forward the exact same computed object without recomputation or mutation."""
    payload = {
        "train/loss_total": 0.5,
        "id/normalized_macro_rmse": 0.25,
        "physics/id/momentum_residual_mse": 4.0,
    }
    observed: list[tuple[int, dict[str, float]]] = []

    def consume(epoch: int, values: dict[str, float]) -> None:
        observed.append((epoch, values))

    callback = tracking.combine_epoch_callbacks(consume, consume)
    assert callback is not None
    callback(5, payload)
    assert observed == [(5, payload), (5, payload)]
    assert observed[0][1] is payload
    assert observed[1][1] is payload


def test_authoritative_consumer_runs_before_observer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve callback order while still failing the requested observer."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    fake.runs[0].fail_log = True
    consumed: list[tuple[int, float]] = []

    def consume(epoch: int, values: dict[str, float]) -> None:
        consumed.append((epoch, values["id/normalized_macro_rmse"]))

    callback = tracking.combine_epoch_callbacks(
        consume,
        tracking.epoch_callback(session),
    )
    assert callback is not None
    with pytest.raises(tracking.TrackingIOError):
        callback(
            5,
            {
                "train/loss_total": 1.0,
                "id/normalized_macro_rmse": 0.25,
            },
        )
    assert consumed == [(5, 0.25)]
    assert fake.runs[0].logs == []


def test_initialization_and_finish_errors_are_redacted_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never expose the active API key in initialization or finish diagnostics."""
    secret = "top-secret-wandb-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    fake = _FakeWandb(
        init_error=RuntimeError(f"authentication failed WANDB_API_KEY={secret}"),
    )
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
    assert state["status"] == "failed_before_start"

    finish_fake = _FakeWandb()
    _patch_wandb(monkeypatch, finish_fake)
    finish_state, finish_update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(),
        run_dir=tmp_path,
        state_updater=finish_update,
    )
    finish_fake.runs[0].fail_finish = True
    with pytest.raises(tracking.TrackingIOError, match="online W&B finish"):
        session.finish(status="completed")
    assert finish_state["status"] == "failed"
    assert finish_state["failed_operation"] == "finish"


def _curated_files(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    files: dict[str, Path] = {}
    for name in (
        "accuracy_physics_pareto",
        "dual_continuity_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    ):
        path = root / f"{name}.png"
        path.write_bytes(b"rendered")
        files[name] = path
    return files


def test_evaluation_artifact_upload_is_explicit_complete_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit only two provenance files and the complete five-item media bundle."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    disabled = tracking.initialize_wandb(_resolved_config(), run_dir=tmp_path)
    analysis_root = tmp_path / "analysis" / "id"
    analysis_root.mkdir(parents=True)
    provenance = analysis_root / "artifact_provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    with pytest.raises(tracking.TrackingUploadError, match="disabled"):
        disabled.upload_files({"artifact_provenance": provenance})

    session = tracking.initialize_wandb(
        _resolved_config(upload=True),
        run_dir=tmp_path,
    )
    with pytest.raises(tracking.TrackingUploadError, match="Unsupported"):
        session.upload_files({"config": tmp_path / "config.yaml"})
    session.upload_files({"artifact_provenance": provenance})

    media = _curated_files(tmp_path / "rendered")
    table = {"columns": ["label", "score"], "data": [["run", 0.25]]}
    incomplete = dict(media)
    incomplete.pop("spectral_fidelity")
    with pytest.raises(tracking.TrackingUploadError, match="missing required"):
        session.upload_post_artifact(
            artifact_root=analysis_root,
            media_files=incomplete,
            tables={"run_summary_table": table},
        )
    cached = dict(media)
    cached["accuracy_physics_pareto"] = analysis_root / "cached.png"
    cached["accuracy_physics_pareto"].write_bytes(b"rendered")
    with pytest.raises(tracking.TrackingUploadError, match="outside the immutable"):
        session.upload_post_artifact(
            artifact_root=analysis_root,
            media_files=cached,
            tables={"run_summary_table": table},
        )
    session.upload_post_artifact(
        artifact_root=analysis_root,
        media_files=media,
        tables={"run_summary_table": table},
    )
    run = fake.runs[1]
    assert [Path(item[0]).name for item in run.saved] == ["artifact_provenance.json"]
    assert len(run.artifacts) == 1
    bundle, aliases = run.artifacts[0]
    assert aliases == ["latest"]
    assert {name for _path, name in bundle.files} == {f"{name}.png" for name in media}
    assert bundle.tables == [(table, "run_summary_table")]


def test_persisted_identity_requires_exactly_one_run_id() -> None:
    """Recover one opaque ID and maximum epoch, rejecting ambiguous summaries."""
    summary = {
        "runtime_sessions": [
            {"tracking": {"wandb_run_id": "abc", "last_logged_epoch": 2}},
            {"tracking": {"wandb_run_id": "abc", "last_logged_epoch": 5}},
        ]
    }
    assert tracking.persisted_wandb_identity(summary) == ("abc", 5)
    summary["runtime_sessions"].append({"tracking": {"wandb_run_id": "different"}})
    with pytest.raises(tracking.TrackingInitializationError, match="found 2"):
        tracking.persisted_wandb_identity(summary)


def test_wandb_config_is_strict() -> None:
    """Reject malformed values and invalid workflow context."""
    invalid_settings = (
        ({"mode": "invalid"}, r"tracking\.wandb\.mode"),
        ({"monitor": {"interval": 0}}, r"tracking\.wandb\.monitor\.interval"),
        (
            {"upload": {"evaluation_artifacts": 1}},
            r"tracking\.wandb\.upload\.evaluation_artifacts",
        ),
        ({"workflow": "optuna_trial"}, r"tracking\.wandb\.study is required"),
        (
            {"workflow": "train", "study": "not_allowed"},
            r"tracking\.wandb\.study is valid only",
        ),
    )
    for wandb_settings, match in invalid_settings:
        raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
        raw["tracking"] = {"wandb": wandb_settings}
        with pytest.raises(ValueError, match=match):
            experiments.config.loader.resolve_config(raw)
