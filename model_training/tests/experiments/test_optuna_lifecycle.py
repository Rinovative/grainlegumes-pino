# ruff: noqa: S101
"""Protect Optuna persistence, resume, pruning, and terminal-state behavior."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
import torch
from src import common, domain, experiments
from support import configs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

optuna_runtime = experiments.tuning.optuna
_EXPECTED_GLOBAL_STEP = 7
_EXPECTED_RESUMED_TRIAL_COUNT = 4
_TASK = domain.tasks.registry.get_task("steady_flow")
_OBJECTIVE_ID = next(metric.id for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")


class _Trial:
    """Implement the small trial surface needed by reporter and failure tests."""

    def __init__(self, *, number: int = 0, prune: bool = False) -> None:
        self.number = number
        self.prune = prune
        self.attrs: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> Any:
        del name
        return choices[0]

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        del name, high, log, step
        return low

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        del name, high, log, step
        return low

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def set_user_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def should_prune(self) -> bool:
        return self.prune


def _load(tmp_path: Path) -> optuna_runtime.OptunaStudyConfig:
    """Load one artificial study with external tracking disabled."""
    path = configs.write_yaml(
        tmp_path / "synthetic-lifecycle.yaml",
        configs.optuna_config(),
    )
    loaded = optuna_runtime.load_optuna_study_config(path)
    base = copy.deepcopy(loaded.base_config)
    base["tracking"]["wandb"]["mode"] = "disabled"
    return replace(loaded, base_config=base)


def test_signature_excludes_runtime_location_but_covers_science(
    tmp_path: Path,
) -> None:
    """Separate study continuation identity from invocation-only settings."""
    loaded = _load(tmp_path)
    baseline = optuna_runtime.build_study_signature(loaded)["digest"]

    operational_base = copy.deepcopy(loaded.base_config)
    operational_base["run"]["device"] = "cuda"
    operational_base["paths"]["output_root"] = str(tmp_path / "elsewhere")
    operational_study = copy.deepcopy(loaded.study)
    operational_study.update(
        {
            "name": "renamed_display",
            "n_trials": 99,
            "storage": "sqlite:///relocated.db",
        }
    )
    operational = replace(
        loaded,
        base_config=operational_base,
        study=operational_study,
    )
    assert optuna_runtime.build_study_signature(operational)["digest"] == baseline

    scientific_base = copy.deepcopy(loaded.base_config)
    scientific_base["optimizer"]["lr"] = 2.0e-3
    scientific = replace(loaded, base_config=scientific_base)
    assert optuna_runtime.build_study_signature(scientific)["digest"] != baseline

    changed_space = list(loaded.search_space)
    changed_space[0] = replace(
        changed_space[0],
        values=(*changed_space[0].values, 16),
    )
    changed = replace(loaded, search_space=tuple(changed_space))
    assert optuna_runtime.build_study_signature(changed)["digest"] != baseline


def test_reporter_uses_completed_epochs_and_prunes_immediately() -> None:
    """Reject missing/discontinuous evidence and stop on the pruning epoch."""
    missing = optuna_runtime.OptunaEpochReporter(
        trial=_Trial(),
        objective_id=_OBJECTIVE_ID,
        direction="minimize",
    )
    with pytest.raises(KeyError, match="Held-out Optuna objective"):
        missing(1, {"train/loss_total": 0.1})

    discontinuous = optuna_runtime.OptunaEpochReporter(
        trial=_Trial(),
        objective_id=_OBJECTIVE_ID,
        direction="minimize",
    )
    with pytest.raises(ValueError, match="Expected 1, received 2"):
        discontinuous(2, {f"id/{_OBJECTIVE_ID}": 0.5})

    trial = _Trial(prune=True)
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=trial,
        objective_id=_OBJECTIVE_ID,
        direction="minimize",
    )
    with pytest.raises(optuna.TrialPruned, match="completed epoch 1"):
        reporter(
            1,
            {
                f"id/{_OBJECTIVE_ID}": 0.5,
                "global_step": float(_EXPECTED_GLOBAL_STEP),
            },
        )
    assert trial.reports == [(0.5, 1)]
    assert trial.attrs["last_reported_epoch"] == 1
    assert trial.attrs["last_global_step"] == _EXPECTED_GLOBAL_STEP


def test_existing_unbound_study_fails_before_trial_allocation(
    tmp_path: Path,
) -> None:
    """Reject a pre-existing SQLite study without semantic metadata."""
    loaded = optuna_runtime.with_runtime_overrides(
        _load(tmp_path),
        device="cpu",
        output_root=tmp_path,
    )
    study_name = loaded.study["name"]
    study_dir = tmp_path / "steady_flow" / "studies" / study_name
    study_dir.mkdir(parents=True)
    storage = f"sqlite:///{study_dir / (study_name + '.db')}"
    optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
    )

    with pytest.raises(
        ValueError,
        match="missing required semantic metadata",
    ):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)
    assert not list(study_dir.glob("trial_*"))


def test_sqlite_study_persists_states_and_resumes_new_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve completed history while adding failed and pruned trial states."""
    loaded = _load(tmp_path)
    base = copy.deepcopy(loaded.base_config)
    base["training"]["epochs"] = 10
    base["training"]["evaluation_interval"] = 5
    base["training"]["ood_evaluation_interval"] = 5
    base["tracking"]["wandb"]["monitor"]["interval"] = 5
    settings = copy.deepcopy(loaded.study)
    settings["name"] = "synthetic_lifecycle"
    settings["pruner"] = {
        "kind": "median",
        "n_startup_trials": 0,
        "n_warmup_steps": 5,
        "interval_steps": 5,
    }
    loaded = optuna_runtime.with_runtime_overrides(
        replace(loaded, base_config=base, study=settings),
        device="cpu",
        output_root=tmp_path,
    )

    def synthetic_factory(
        _config: optuna_runtime.OptunaStudyConfig,
    ) -> Callable[[Any], float]:
        def objective(trial: Any) -> float:
            if trial.number == 1:
                message = "recoverable synthetic failure"
                raise optuna_runtime.RecoverableTrialError(message)
            reporter = optuna_runtime.OptunaEpochReporter(
                trial=trial,
                objective_id=_OBJECTIVE_ID,
                direction="minimize",
                evaluation_interval=5,
                target_epoch=10,
                pruner_config=settings["pruner"],
            )
            values = (0.10, 0.09) if trial.number == 0 else (1.0, 0.9)
            for epoch, value in zip((5, 10), values, strict=True):
                reporter(
                    epoch,
                    {
                        f"id/{_OBJECTIVE_ID}": value,
                        "global_step": float(epoch),
                    },
                )
            assert reporter.best_value is not None
            return reporter.best_value

        return objective

    monkeypatch.setattr(
        optuna_runtime,
        "create_objective",
        synthetic_factory,
    )
    first = optuna_runtime.run_optuna_study(loaded, n_trials=2)
    first_evidence = (
        first.trials[0].state,
        first.trials[0].value,
        dict(first.trials[0].intermediate_values),
    )
    resumed = optuna_runtime.run_optuna_study(loaded, n_trials=1)

    assert [trial.number for trial in resumed.trials] == [0, 1, 2]
    assert [trial.state for trial in resumed.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]
    assert first_evidence == (
        resumed.trials[0].state,
        resumed.trials[0].value,
        dict(resumed.trials[0].intermediate_values),
    )
    assert resumed.trials[0].intermediate_values == {5: 0.10, 10: 0.09}
    assert resumed.trials[2].intermediate_values == {5: 1.0}

    study_dir = tmp_path / "steady_flow/studies/synthetic_lifecycle"
    summary = json.loads((study_dir / "study_summary.json").read_text(encoding="utf-8"))
    assert [trial["number"] for trial in summary["trials"]] == [0, 1, 2]
    assert summary["training_seed"] == loaded.base_config["run"]["seed"]
    assert summary["sampler"]["seed"] == loaded.study["seed"]

    def unexpected_factory(
        _config: optuna_runtime.OptunaStudyConfig,
    ) -> Callable[[Any], float]:
        def objective(_trial: Any) -> float:
            message = "unexpected synthetic bug"
            raise RuntimeError(message)

        return objective

    monkeypatch.setattr(
        optuna_runtime,
        "create_objective",
        unexpected_factory,
    )
    with pytest.raises(RuntimeError, match="unexpected synthetic bug"):
        optuna_runtime.run_optuna_study(loaded, n_trials=1)

    storage = f"sqlite:///{study_dir / 'synthetic_lifecycle.db'}"
    reopened = optuna.load_study(
        study_name="synthetic_lifecycle",
        storage=storage,
    )
    assert [trial.number for trial in reopened.trials] == [0, 1, 2, 3]
    assert reopened.trials[-1].state == optuna.trial.TrialState.FAIL

    drifted = copy.deepcopy(loaded.base_config)
    drifted["optimizer"]["lr"] = 2.0e-3
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(
            replace(loaded, base_config=drifted),
            n_trials=1,
        )
    assert (
        len(
            optuna.load_study(
                study_name="synthetic_lifecycle",
                storage=storage,
            ).trials
        )
        == _EXPECTED_RESUMED_TRIAL_COUNT
    )


def _install_running_failure_harness(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace expensive boundaries while retaining real trial lifecycle state."""
    processor = SimpleNamespace(
        state_dict=dict,
        in_normalizer=None,
        out_normalizer=SimpleNamespace(std=torch.ones(3)),
        to=lambda _device: None,
    )

    def configure_reproducibility(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> dict[str, int]:
        del device
        return {"model_init": 1}

    def seed_process(_seed: int, *, device: torch.device) -> None:
        del device

    monkeypatch.setattr(
        experiments.run,
        "configure_reproducibility",
        configure_reproducibility,
    )
    monkeypatch.setattr(experiments.run, "seed_process", seed_process)
    monkeypatch.setattr(
        experiments.config.loader,
        "create_dataloaders_from_config",
        lambda *_args, **_kwargs: {
            "data_processor": processor,
            "split_indices": {},
            "train": object(),
            "eval": object(),
            "ood": object(),
        },
    )

    def build_model(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> torch.nn.Module:
        del device
        return torch.nn.Linear(1, 1)

    def build_training_loss(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> object:
        del device
        return SimpleNamespace()

    def build_evaluation_metrics(
        _config: dict[str, Any],
        *,
        device: torch.device,
        output_standard_deviations: torch.Tensor,
    ) -> dict[str, Any]:
        del device
        assert torch.equal(output_standard_deviations, torch.ones(3))
        return {}

    monkeypatch.setattr(
        optuna_runtime.learning.models.factory,
        "build_model",
        build_model,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.losses.factory,
        "build_training_loss",
        build_training_loss,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.metrics.metrics,
        "build_evaluation_metrics",
        build_evaluation_metrics,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.optim,
        "build_optimizer",
        lambda _model, _config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.optim,
        "build_scheduler",
        lambda _optimizer, _config: None,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.checkpoint,
        "build_checkpoint_identity",
        lambda *_args, **_kwargs: {"effective_config_digest": "synthetic"},
    )
    monkeypatch.setattr(
        optuna_runtime.experiments.tracking,
        "build_monitor_membership",
        lambda *_args: None,
    )

    def fail_training(**_kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(
        optuna_runtime.learning.training.loop,
        "train_loop",
        fail_training,
    )


@pytest.mark.parametrize(
    ("error_factory", "expected_status", "expected_error"),
    [
        pytest.param(
            lambda: torch.cuda.OutOfMemoryError("synthetic CUDA OOM"),
            "oom_pruned",
            optuna.TrialPruned,
            id="oom",
        ),
        pytest.param(
            lambda: FloatingPointError("synthetic non-finite loss"),
            "nonfinite_pruned",
            optuna.TrialPruned,
            id="nonfinite",
        ),
        pytest.param(
            lambda: optuna_runtime.RecoverableTrialError("synthetic recoverable failure"),
            "recoverable_failed",
            optuna_runtime.RecoverableTrialError,
            id="recoverable",
        ),
        pytest.param(
            lambda: RuntimeError("synthetic programming failure"),
            "failed",
            RuntimeError,
            id="failed",
        ),
        pytest.param(
            lambda: KeyboardInterrupt("synthetic interrupt"),
            "interrupted",
            KeyboardInterrupt,
            id="interrupted",
        ),
    ],
)
def test_running_trial_persists_distinct_terminal_states(
    error_factory: Callable[[], BaseException],
    expected_status: str,
    expected_error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify representative terminal outcomes and avoid CUDA cleanup on CPU."""
    loaded = optuna_runtime.with_runtime_overrides(
        _load(tmp_path),
        device="cpu",
        output_root=tmp_path,
    )
    error = error_factory()
    _install_running_failure_harness(error, monkeypatch)

    def cuda_query_forbidden() -> bool:
        pytest.fail("CPU cleanup must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_query_forbidden)
    trial = _Trial()
    with pytest.raises(expected_error):
        optuna_runtime.run_trial(loaded, trial)

    run_dir = common.paths.resolve_optuna_trial_dir(
        "steady_flow",
        loaded.study["name"],
        trial.attrs["run_name"],
        output_root=tmp_path,
    )
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == expected_status
    assert summary["failure_context"]["error_type"] == type(error).__name__
    assert summary["sampled_parameters"] == trial.attrs["overrides"]


def test_public_run_statuses_distinguish_core_terminal_outcomes() -> None:
    """Keep completed, pruned, failed, and interrupted persistence distinct."""
    assert {"completed", "pruned", "failed", "interrupted"} <= (experiments.run.RUN_STATUSES)
