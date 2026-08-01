# ruff: noqa: EM101, PLR2004, S101, S105, TRY003
"""Protect the canonical W&B identity, metric, failure, and upload contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from src import experiments, learning

tracking = experiments.tracking
_CONFIG_PATH = Path(__file__).parents[2] / "configs/experiments/steady_flow_fno.yaml"
_PI_FNO_CONFIG_PATH = Path(__file__).parents[2] / "configs/experiments/steady_flow_pifno.yaml"
_UNO_CONFIG_PATH = Path(__file__).parents[2] / "configs/experiments/steady_flow_uno.yaml"
_PI_UNO_CONFIG_PATH = Path(__file__).parents[2] / "configs/experiments/steady_flow_piuno.yaml"
_GPU_CONFIG_PATH = Path(__file__).parents[2] / "configs/acceptance/steady_flow_fno_gpu_smoke.yaml"


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


@pytest.mark.parametrize(
    ("config_path", "variant", "run_name"),
    [
        (_CONFIG_PATH, "fno", "steady_flow__fno_m128x160_h64_l3__s9"),
        (_PI_FNO_CONFIG_PATH, "pi-fno", "steady_flow__pi-fno_m128x160_h64_l3__s9"),
        (_UNO_CONFIG_PATH, "uno", "steady_flow__uno_m64x64_h32_l5_r0p5__s9"),
        (_PI_UNO_CONFIG_PATH, "pi-uno", "steady_flow__pi-uno_m64x64_h32_l5_r0p5__s9"),
    ],
)
def test_normal_taxonomy_uses_only_variant_tag_and_structured_architecture(
    config_path: Path,
    variant: str,
    run_name: str,
) -> None:
    """Keep one base tag while retaining compact names and full model config."""
    config = _resolved_config(config_path=config_path)
    settings = config["tracking"]["wandb"]
    assert settings["mode"] == "online"
    assert settings["project"] == "grainlegumes-pino-airflow"
    assert settings["entity"] == "Rinovative-Hub"
    assert settings["tags"] == [variant]
    assert "group" not in settings
    assert "job_type" not in settings
    assert "campaign" not in settings
    assert not any(tag.startswith("arch:") for tag in settings["tags"])
    assert "final" not in settings["tags"]
    assert config["run"]["name"] == run_name
    assert config["model"]["params"]


def test_optuna_and_acceptance_taxonomy_is_minimal() -> None:
    """Use variant-plus-optuna for trials and no tags for validation runs."""
    tuned = _resolved_config(workflow="optuna_trial", study="steady_flow_fno_search")["tracking"]["wandb"]
    assert tuned["tags"] == ["fno", "optuna"]
    assert tuned["study"] == "steady_flow_fno_search"
    assert "group" not in tuned
    assert "job_type" not in tuned

    gpu_raw = experiments.config.loader.load_yaml(_GPU_CONFIG_PATH)
    gpu = experiments.config.loader.resolve_config(gpu_raw)["tracking"]["wandb"]
    assert gpu["tags"] == []
    assert "group" not in gpu
    assert "job_type" not in gpu


@pytest.mark.parametrize(
    ("config_path", "variant", "study"),
    [
        (_CONFIG_PATH, "fno", "steady_flow_fno_search"),
        (_PI_FNO_CONFIG_PATH, "pi-fno", "steady_flow_pifno_search"),
        (_UNO_CONFIG_PATH, "uno", "steady_flow_uno_search"),
        (_PI_UNO_CONFIG_PATH, "pi-uno", "steady_flow_piuno_search"),
    ],
)
def test_all_optuna_variants_use_variant_plus_optuna(
    config_path: Path,
    variant: str,
    study: str,
) -> None:
    """Keep all trial tags exact and free of architecture or study slugs."""
    settings = _resolved_config(
        workflow="optuna_trial",
        study=study,
        config_path=config_path,
    )["tracking"]["wandb"]
    assert settings["tags"] == [variant, "optuna"]
    assert "group" not in settings
    assert "job_type" not in settings


@pytest.mark.parametrize("workflow", ["gpu_smoke", "cpu_acceptance", "tracking_validation"])
def test_all_acceptance_workflows_have_no_tags(workflow: str) -> None:
    """Keep bounded validation runs intentionally untagged."""
    settings = _resolved_config(workflow=workflow)["tracking"]["wandb"]
    assert settings["tags"] == []
    assert "group" not in settings
    assert "job_type" not in settings


def test_retired_campaign_is_rejected() -> None:
    """Reject the removed campaign/group taxonomy at the user boundary."""
    raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
    raw["tracking"] = {"wandb": {"campaign": "final"}}
    with pytest.raises(ValueError, match=r"unknown key.*campaign"):
        experiments.config.loader.resolve_config(raw)


@pytest.mark.parametrize("key", ["project", "entity", "group", "job_type", "campaign", "tags"])
def test_user_cannot_override_derived_organization(key: str) -> None:
    """Reject every W&B organization field at the public user-schema boundary."""
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
    assert not hasattr(session, "workspace_url")


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
            "parameter_counts": {"total": 127_445_475, "trainable": 127_445_475},
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
    assert "group" not in fake.initializations[0]
    assert "job_type" not in fake.initializations[0]
    assert fake.initializations[2]["config"]["model"]["parameter_counts"] == {
        "total": 127_445_475,
        "trainable": 127_445_475,
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
        "diagnostic": "ood",
        "cadence": "same_evaluation_interval",
    }
    assert payload["loss"]["data"]["kind"] == "relative_h1"
    assert payload["data"]["datasets"]["id"]["dataset_id"] == "train-data"
    assert payload["data"]["split"]["membership_digests"]["eval"] == "d" * 64
    assert payload["data"]["split"]["artifact_sha256"] == "2" * 64
    assert payload["provenance"]["config_digest"] == "1" * 64
    assert "effective_config" not in payload
    assert "paths" not in payload
    assert "tracking" not in payload
    serialized = str(payload)
    assert "job_type" not in serialized
    assert "group" not in serialized
    assert secret not in serialized
    assert "WANDB_API_KEY" not in serialized
    assert str(Path.home()) not in serialized


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        (_CONFIG_PATH, 127_445_475),
        (_PI_FNO_CONFIG_PATH, 127_445_475),
        (_UNO_CONFIG_PATH, 8_758_195),
        (_PI_UNO_CONFIG_PATH, 8_758_195),
    ],
)
def test_instantiated_model_parameter_counts_are_exact_and_runtime_independent(
    config_path: Path,
    expected: int,
) -> None:
    """Count real FNO/UNO parameters without formulas, device, or dtype dependence."""
    config = experiments.config.loader.load_and_resolve_config(config_path)
    model = learning.models.factory.build_model(config, device=torch.device("cpu"))
    cpu_counts = tracking.model_parameter_counts(model)
    assert cpu_counts == {"total": expected, "trainable": expected}
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
        monitor_evaluator=lambda: monitor_values,
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
            "global_step": 12.0,
            "id/normalized_macro_rmse": 0.7,
            "ood/normalized_macro_rmse": 1.0,
            "generalization/objective_gap": 0.3,
            "id/normalized_rmse": 91.0,
            "ood/normalized_rmse": 92.0,
            "objective/value": 99.0,
            **source_accuracy,
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
    ]
    assert [name for name, _kwargs in run.metric_definitions] == expected_definitions
    assert run.metric_definitions[0][1] == {"hidden": True, "summary": "none"}
    assert all(kwargs == {"step_metric": "epoch", "step_sync": False, "summary": "none"} for _name, kwargs in run.metric_definitions[1:])
    assert tracking.AUTOMATIC_HISTORY_TOP_LEVEL_PREFIXES == (
        "Overview",
        "Accuracy",
        "Physics",
        "Diagnostics",
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
    assert "_disable_stats" not in initialization["settings"]
    assert initialization["tags"] == ["fno"]
    assert "group" not in initialization
    assert "job_type" not in initialization
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
    assert not any(key.startswith("final/") for key in run.summary)
    assert "objective/best_value" not in run.summary
    assert run.summary["tracking/status"] == "finished"
    assert "device/requested" not in run.summary
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

    training_definitions = [name for name, _kwargs in fake.runs[0].metric_definitions if name.startswith("Physics/Training/")]
    assert training_definitions == [
        "Physics/Training/loss_momentum",
        "Physics/Training/loss_boundary",
        "Physics/Training/loss_continuity_div_eps_velocity",
        "Physics/Training/residual_weight",
        "Physics/Training/boundary_weight",
    ]
    assert fake.runs[0].logs == [
        (
            {
                "epoch": 1,
                "Physics/Training/loss_momentum": 1.0,
                "Physics/Training/loss_boundary": 2.0,
                "Physics/Training/loss_continuity_div_eps_velocity": 4.0,
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
            "system/cuda_peak_memory_allocated_bytes": 4096.0,
        },
    )
    assert "Diagnostics/cuda_peak_memory_allocated_bytes" in {name for name, _kwargs in fake.runs[0].metric_definitions}
    assert fake.runs[0].logs == [
        (
            {
                "epoch": 1,
                "Diagnostics/epoch_duration_seconds": 2.0,
                "Diagnostics/cuda_peak_memory_allocated_bytes": 4096.0,
            },
            1,
        )
    ]


def test_optuna_history_and_terminal_summary_preserve_trial_comparison_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain each trial trajectory and its selected ID objective for Optuna views."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    session = tracking.initialize_wandb(
        _resolved_config(workflow="optuna_trial", study="steady_flow_fno_search"),
        run_dir=tmp_path,
        semantic_config={
            "tuning": {
                "study_name": "steady_flow_fno_search",
                "trial_number": 7,
                "search_signature": "search-signature",
                "sampled_parameters": {"model.hidden_channels": 48},
            }
        },
    )
    session.log_epoch(3, {"id/normalized_macro_rmse": 0.72})
    session.finish(
        status="completed",
        result={
            "selected_epoch": 2,
            "selected_metrics": {
                "selected/id/normalized_macro_rmse": 0.61,
                "selected/ood/normalized_macro_rmse": 0.79,
            },
        },
    )

    run = fake.runs[0]
    assert run.logs == [({"epoch": 3, "Overview/ID/normalized_macro_rmse": 0.72}, 3)]
    assert run.summary["tuning/study_name"] == "steady_flow_fno_search"
    assert run.summary["tuning/trial_number"] == 7
    assert run.summary["tuning/search_signature"] == "search-signature"
    assert run.summary["tuning/sampled_parameters"] == {"model.hidden_channels": 48}
    assert run.summary["tuning/final_state"] == "completed"
    assert run.summary["selected/id/normalized_macro_rmse"] == 0.61
    assert run.summary["selected/ood/normalized_macro_rmse"] == 0.79
    assert run.summary["selected/epoch"] == 2


def test_monitor_membership_and_metrics_are_fixed_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one saved ID prefix and log four monitor metrics only at cadence."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    config = _resolved_config(epochs=3)
    config["tracking"]["wandb"]["monitor"]["max_cases"] = 2
    config["tracking"]["wandb"]["monitor"]["interval"] = 2
    split = _split_evidence(config)
    split["eval_indices"] = torch.tensor([2, 0, 1])
    first = tracking.build_monitor_membership(config, split)
    second = tracking.build_monitor_membership(config, split)
    assert first == second
    assert first is not None
    assert first["source_indices"] == [2, 0]
    assert first["sample_ids"] == ["case_0003", "case_0001"]

    calls = 0

    def monitor() -> dict[str, float]:
        nonlocal calls
        calls += 1
        return {
            "physics/id/momentum_residual_mse": 1.0,
            "physics/id/continuity_div_velocity_mse": 2.0,
            "physics/id/continuity_div_eps_velocity_mse": 3.0,
            "physics/id/pressure_boundary_mse": 4.0,
        }

    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
        monitor_evaluator=monitor,
    )
    for epoch in (1, 2, 3):
        session.log_epoch(epoch, {"train/loss_total": float(epoch)})
    assert calls == 2
    assert "Physics/ID/momentum_residual_mse" not in fake.runs[0].logs[0][0]
    assert fake.runs[0].logs[1][0]["Physics/ID/continuity_div_velocity_mse"] == 2.0
    assert fake.runs[0].logs[2][0]["Physics/ID/pressure_boundary_mse"] == 4.0


def test_scientific_monitor_failure_is_recorded_but_not_reclassified_as_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the scientific owner and original cause across the W&B callback."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()

    def monitor() -> dict[str, float]:
        cause = ValueError("x-coordinate spacing is materially nonuniform")
        message = "Bounded physics-monitor scientific evaluation failed"
        raise learning.training.loop.PhysicsMonitorEvaluationError(message) from cause

    session = tracking.initialize_wandb(
        _resolved_config(epochs=1),
        run_dir=tmp_path,
        state_updater=update,
        monitor_evaluator=monitor,
    )
    with pytest.raises(learning.training.loop.PhysicsMonitorEvaluationError) as captured:
        session.log_epoch(1, {"train/loss_total": 1.0})
    assert isinstance(captured.value.__cause__, ValueError)
    assert not isinstance(captured.value, tracking.TrackingIOError)
    assert state["status"] == "failed"
    assert state["failed_operation"] == "physics_monitor"
    assert state["failure_owner"] == "scientific_evaluation"
    assert fake.runs[0].logs == []


def test_monitor_payload_contract_failure_is_callback_not_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify unsupported monitor keys as callback orchestration."""
    fake = _FakeWandb()
    _patch_wandb(monkeypatch, fake)
    state, update = _state_recorder()
    session = tracking.initialize_wandb(
        _resolved_config(epochs=1),
        run_dir=tmp_path,
        state_updater=update,
        monitor_evaluator=lambda: {"physics/id/unsupported": 1.0},
    )
    with pytest.raises(tracking.TrackingCallbackError, match="unsupported key"):
        session.log_epoch(1, {"train/loss_total": 1.0})
    assert state["failure_owner"] == "callback_orchestration"
    assert state["failed_operation"] == "physics_monitor"


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


@pytest.mark.parametrize(
    ("wandb_settings", "match"),
    [
        ({"enabled": True}, r"tracking\.wandb.*enabled"),
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
    ],
)
def test_wandb_config_is_strict(
    wandb_settings: dict[str, Any],
    match: str,
) -> None:
    """Reject retired fields, malformed values, and invalid workflow context."""
    raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
    raw["tracking"] = {"wandb": wandb_settings}
    with pytest.raises(ValueError, match=match):
        experiments.config.loader.resolve_config(raw)
