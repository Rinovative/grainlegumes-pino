# ruff: noqa: S101
"""Protect persistent, W&B-independent experiment console diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src import experiments
from support import configs

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _config(*, physics_enabled: bool = False) -> dict[str, Any]:
    """Resolve one dynamically discovered production request for reporter tests."""
    path = configs.experiment_config_path(
        model_kind="fno",
        physics_enabled=physics_enabled,
    )
    config = experiments.config.loader.load_and_resolve_config(path)
    config["tracking"]["wandb"]["mode"] = "disabled"
    return config


def _epoch_payload() -> dict[str, float]:
    """Return one authoritative payload containing every due event type."""
    return {
        "train/loss_total": 0.75,
        "train/loss_data": 0.7,
        "optimization/learning_rate": 0.001,
        "optimization/scheduler_old_learning_rate": 0.001,
        "optimization/scheduler_new_learning_rate": 0.0005,
        "system/train_duration_seconds": 1.25,
        "system/train_samples_per_second": 16.0,
        "system/epoch_duration_seconds": 2.5,
        "system/session_elapsed_seconds": 12.5,
        "system/estimated_remaining_seconds": 20.0,
        "system/id_evaluation_duration_seconds": 0.4,
        "system/id_evaluation_case_count": 8.0,
        "system/ood_evaluation_duration_seconds": 0.5,
        "system/ood_evaluation_case_count": 4.0,
        "system/physics_monitor_duration_seconds": 0.6,
        "system/physics_monitor_case_count": 4.0,
        "id/normalized_macro_rmse": 0.25,
        "id/normalized_rmse_p": 0.2,
        "id/normalized_rmse_u": 0.3,
        "id/normalized_rmse_v": 0.4,
        "id/normalized_relative_h1": 0.5,
        "ood/normalized_macro_rmse": 0.75,
        "ood/normalized_rmse_p": 0.7,
        "ood/normalized_rmse_u": 0.8,
        "ood/normalized_rmse_v": 0.9,
        "ood/normalized_relative_h1": 1.0,
        "physics/id/momentum_residual_mse": 10.0,
        "physics/id/continuity_div_velocity_mse": 20.0,
        "physics/id/continuity_div_eps_velocity_mse": 30.0,
        "physics/id/pressure_boundary_mse": 40.0,
        "checkpoint/new_best": 1.0,
        "checkpoint/last_published": 1.0,
    }


def test_disabled_wandb_startup_and_due_epoch_are_complete_line_oriented(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Show every lifecycle phase without SDK state, TTY rewriting, or secrets."""
    secret = "never-print-this-" + "api-key"
    monkeypatch.setenv("WANDB_API_KEY", secret)
    config = _config()
    reporter = experiments.console.ConsoleReporter(config=config, run_dir=tmp_path)
    reporter.startup(resolved_device="cpu")
    reporter.epoch(5, _epoch_payload())

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert all(line.startswith("event=") for line in lines)
    assert all("\r" not in line for line in lines)
    assert sum(line.startswith("event=startup ") for line in lines) == 1
    assert sum(line.startswith("event=loss_composition ") for line in lines) == 1
    assert sum(line.startswith("event=training_epoch ") for line in lines) == 1
    assert sum(line.startswith("event=id_evaluation ") for line in lines) == 1
    assert sum(line.startswith("event=ood_evaluation ") for line in lines) == 1
    assert sum(line.startswith("event=physics_monitor ") for line in lines) == 1
    assert sum(line.startswith("event=best_checkpoint ") for line in lines) == 1
    assert sum(line.startswith("event=scheduler_update ") for line in lines) == 1
    output = captured.out
    assert f"run_name={config['run']['name']}" in output
    assert output.index(f"run_name={config['run']['name']}") < output.index(" project=")
    assert "architecture=fno_m128x160_h64_l3" in output
    assert "wandb_mode=disabled" in output
    assert "id_interval=5 id_first=5" in output
    assert "ood_interval=5 ood_first=5" in output
    assert "physics_interval=5 physics_first=5" in output
    assert "formula=1.0*relative_h1[normalized]" in output
    assert "epoch=5/600" in output
    assert "samples_per_second=16" in output
    assert "dataset=lhs_var120_seed4001" in output
    assert "objective=0.25" in output
    assert "momentum_residual_mse=10" in output
    assert "last_checkpoint=published" in output
    assert "new_learning_rate=0.0005" in output
    assert "batch=" not in output
    assert secret not in output
    assert captured.err == ""


def test_pi_loss_composition_contains_only_active_terms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explain the selected PI continuity and scheduled residual/boundary terms."""
    config = _config(physics_enabled=True)
    reporter = experiments.console.ConsoleReporter(config=config, run_dir=tmp_path)
    reporter.startup(resolved_device="cpu")
    output = capsys.readouterr().out
    physics = config["loss"]["physics"]
    assert "pi_enabled=True" in output
    assert f"continuity={physics['continuity']}" in output
    assert "momentum_and_" + physics["continuity"] in output
    assert "residual_weight(epoch" in output
    assert "boundary_weight(epoch" in output
    inactive = "div_velocity" if physics["continuity"] == "div_eps_velocity" else "div_eps_velocity"
    assert "momentum_and_" + inactive not in output


def test_final_and_failure_output_identify_selected_checkpoint_phase_and_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Separate reloaded-best results and retain a redacted complete failure stack."""
    config = _config()
    reporter = experiments.console.ConsoleReporter(config=config, run_dir=tmp_path)
    result = {
        "completed_epoch": 600,
        "best_epoch": 575,
        "best_metric": 0.2,
        "best_checkpoint_path": str(tmp_path / "best_checkpoint.pt"),
        "last_checkpoint_path": str(tmp_path / "last_checkpoint.pt"),
        "selected_metrics": {
            "selected/id/normalized_macro_rmse": 0.19,
            "selected/ood/normalized_macro_rmse": 0.31,
            "selected/physics/momentum_residual_mse": 4.0,
        },
    }
    reporter.final(result, total_wall_seconds=123.0)
    success = capsys.readouterr().out
    assert "event=completed" in success
    assert "selected_best_epoch=575" in success
    assert "selected_source=reloaded_best_checkpoint.pt" in success
    assert f"best_checkpoint={tmp_path / 'best_checkpoint.pt'}" in success
    assert "final_id_objective=0.19" in success
    assert "final_ood_objective=0.31" in success

    try:
        error = RuntimeError("api_key=do-not-leak")
        error.__dict__["training_phase"] = "physics_monitor"
        error.__dict__["completed_epoch"] = 15
        raise error
    except RuntimeError as captured_error:
        reporter.failure(captured_error, status="failed")
    failure = capsys.readouterr().err
    assert "event=failure" in failure
    assert "phase=physics_monitor" in failure
    assert "epoch=15" in failure
    assert "exception_type=RuntimeError" in failure
    assert "Traceback (most recent call last):" in failure
    assert "<redacted>" in failure
    assert "do-not-leak" not in failure


def test_optuna_trial_events_expose_observation_and_pruning_decision(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep trial identity, sampled values, objective, step, and decision visible."""
    experiments.console.optuna_trial_event(
        "observed",
        study="steady_flow_fno_search",
        trial=42,
        run_name="fno_m48x48_h64_l4__s9__optuna_trial_042",
        sampled={"hidden_channels": 64, "lr": 0.001},
        objective_id="normalized_macro_rmse",
        objective=0.25,
        step=7,
        pruning="continue",
    )
    output = capsys.readouterr().out
    assert "status=observed" in output
    assert "study=steady_flow_fno_search" in output
    assert "trial=42" in output
    assert "run_name=fno_m48x48_h64_l4__s9__optuna_trial_042" in output
    assert "sampled=hidden_channels:64,lr:0.001" in output
    assert "objective=0.25" in output
    assert "step=7" in output
    assert "pruning=continue" in output
