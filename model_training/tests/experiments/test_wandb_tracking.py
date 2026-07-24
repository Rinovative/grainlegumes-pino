# ruff: noqa: S101
"""Verify optional W&B configuration, logging, and lifecycle isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from src import experiments
from torch.optim.sgd import SGD

tracking = experiments.tracking
_CONFIG_PATH = Path(__file__).parents[2] / "configs/experiments/steady_flow_fno.yaml"
_EXPECTED_LOG_EPOCH = 5


class _FakeRun:
    """Record SDK interactions without network access."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logs: list[tuple[dict[str, Any], int]] = []
        self.exit_codes: list[int] = []

    def log(self, data: dict[str, Any], *, step: int) -> None:
        self.logs.append((dict(data), step))

    def finish(self, exit_code: int = 0) -> None:
        self.exit_codes.append(exit_code)


class _FakeWandb:
    """Create isolated fake runs and capture initialization settings."""

    def __init__(self) -> None:
        self.initializations: list[dict[str, Any]] = []
        self.runs: list[_FakeRun] = []

    def init(self, **settings: Any) -> _FakeRun:
        self.initializations.append(settings)
        run = _FakeRun()
        self.runs.append(run)
        return run


def _resolved_config(*, enabled: bool) -> dict[str, Any]:
    """Resolve one experiment with an explicit optional-tracking selection."""
    raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
    raw["tracking"] = {
        "wandb": {
            "enabled": enabled,
            "project": "airflow-tests",
            "entity": "research-team",
            "group": "steady-flow",
            "tags": ["synthetic", "cpu"],
            "mode": "online",
        }
    }
    return experiments.config.loader.resolve_config(raw)


def test_disabled_wandb_never_imports_the_sdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default disabled path is a dependency- and network-free no-op."""

    def fail_import(_name: str) -> None:
        pytest.fail("disabled W&B must not import the SDK")

    config = _resolved_config(enabled=False)
    optimizer = SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.2)
    monkeypatch.setattr(tracking.importlib, "import_module", fail_import)
    session = tracking.initialize_wandb(
        config,
        run_dir=tmp_path,
    )

    assert session.enabled is False
    assert (
        tracking.epoch_callback(
            session,
            optimizer,
        )
        is None
    )
    session.log_epoch(1, {"train_loss": 2.0}, learning_rate=0.2)
    session.finish(status="completed", result={"best_metric": 1.0})


def test_enabled_wandb_uses_standard_sdk_lifecycle_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled runs initialize, log semantic values, and finish on success/failure."""
    fake_wandb = _FakeWandb()
    config = _resolved_config(enabled=True)
    optimizer = SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.125)
    monkeypatch.setattr(tracking.importlib, "import_module", lambda name: fake_wandb if name == "wandb" else None)
    session = tracking.initialize_wandb(config, run_dir=tmp_path)
    callback = tracking.epoch_callback(session, optimizer)

    assert callback is not None
    callback(
        5,
        {
            "train_loss": 0.75,
            "normalized_relative_h1": 0.5,
            "physical_rmse_p": 2.0,
        },
    )
    session.finish(
        status="completed",
        result={
            "best_epoch": 5,
            "best_metric": 0.5,
            "completed_epoch": 5,
            "global_step": 12,
        },
    )
    session.finish(status="failed", error="must be idempotent")

    settings = fake_wandb.initializations[0]
    assert settings["project"] == "airflow-tests"
    assert settings["entity"] == "research-team"
    assert settings["group"] == "steady-flow"
    assert settings["tags"] == ["synthetic", "cpu"]
    assert settings["mode"] == "online"
    assert settings["name"] == config["run"]["name"]
    run_identity = f"{config['task']}\0{config['run']['name']}"
    assert settings["id"] == hashlib.sha256(run_identity.encode()).hexdigest()[:32]
    assert settings["resume"] == "allow"
    assert settings["job_type"] == "training"
    assert settings["dir"] == str(tmp_path)
    assert settings["config"] == config
    assert "api_key" not in settings
    run = fake_wandb.runs[0]
    assert len(run.logs) == 1
    logged_values, logged_step = run.logs[0]
    assert logged_step == _EXPECTED_LOG_EPOCH
    required_log_values = {
        "epoch": 5,
        "optimizer/learning_rate": 0.125,
        "train/loss": 0.75,
        "evaluation/normalized_relative_h1": 0.5,
        "evaluation/physical_rmse_p": 2.0,
        "objective/value": 0.5,
    }
    for key, expected in required_log_values.items():
        assert logged_values[key] == expected
    required_summary = {
        "status": "completed",
        "objective/id": "normalized_relative_h1",
        "best_epoch": 5,
        "best_metric": 0.5,
        "completed_epoch": 5,
        "global_step": 12,
    }
    for key, expected in required_summary.items():
        assert run.summary[key] == expected
    assert run.exit_codes == [0]

    failed_session = tracking.initialize_wandb(config, run_dir=tmp_path)
    failed_session.finish(status="failed", error="synthetic failure")
    assert fake_wandb.runs[1].summary["status"] == "failed"
    assert fake_wandb.runs[1].summary["error"] == "synthetic failure"
    assert fake_wandb.runs[1].exit_codes == [1]


@pytest.mark.parametrize(
    ("wandb_settings", "match"),
    [
        ({"enabled": 1}, r"tracking\.wandb\.enabled"),
        ({"mode": "disabled"}, r"tracking\.wandb\.mode"),
        ({"tags": ["duplicate", "duplicate"]}, r"tracking\.wandb\.tags must be unique"),
    ],
)
def test_wandb_config_is_strict(
    wandb_settings: dict[str, Any],
    match: str,
) -> None:
    """Tracking settings reject coercion, SDK-only modes, and duplicate tags."""
    raw = experiments.config.loader.load_yaml(_CONFIG_PATH)
    raw["tracking"] = {"wandb": wandb_settings}

    with pytest.raises(ValueError, match=match):
        experiments.config.loader.resolve_config(raw)
