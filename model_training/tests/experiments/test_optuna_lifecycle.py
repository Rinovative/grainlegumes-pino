# ruff: noqa: S101
"""
Exercise continuous pruning, semantic study signatures, reopening, and outcomes.

A tiny CPU SQLite study proves actual completed-epoch steps and fresh additional
trials; stubs classify pruning, non-finite, OOM, recoverable, interrupt, and bug
paths while guarding CPU cleanup from CUDA calls. General YAML/search parsing is
covered by ``test_optuna_contract``; no production study is run.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
import torch
from src import common, experiments

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

optuna_runtime = experiments.tuning.optuna
search_space = experiments.tuning.search_space
_CONFIG_ROOT = Path(__file__).parents[2] / "configs" / "optuna"
_EXPECTED_OBJECTIVE = {
    "id": "normalized_macro_rmse",
    "kind": "macro_rmse",
    "space": "normalized",
    "fields": ["p", "u", "v"],
    "reduction": "field_macro_element_mean",
    "direction": "minimize",
}
_MAX_CATEGORICAL_VALUES = 3
_EXPECTED_GLOBAL_STEP = 7
_EXPECTED_RESUMED_TRIAL_COUNT = 4
_EXPECTED_RECIPES = {
    "steady_flow_fno_search.yaml": ("fno", False),
    "steady_flow_pifno_search.yaml": ("fno", True),
    "steady_flow_piuno_search.yaml": ("uno", True),
    "steady_flow_uno_search.yaml": ("uno", False),
}


class _Trial:
    """
    Implement only the Optuna trial surface needed by lifecycle policy tests.

    Suggestions choose the first or lower-bound value; reports and attributes remain
    in memory, and one toggle decides pruning. The fake performs no persistence,
    distribution validation, or sampler logic.
    """

    def __init__(self, *, number: int = 0, prune: bool = False) -> None:
        """Initialize deterministic identity, pruning policy, and empty records."""
        self.number = number
        self.prune = prune
        self.attrs: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        """Return the first categorical choice without emulating a sampler."""
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
        """Return the float lower bound after accepting Optuna-shaped arguments."""
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
        """Return the integer lower bound after accepting Optuna-shaped arguments."""
        del name, high, log, step
        return low

    def report(self, value: float, step: int) -> None:
        """Retain one objective report at its completed-epoch step."""
        self.reports.append((value, step))

    def set_user_attr(self, key: str, value: Any) -> None:
        """Retain one user attribute exactly as the reporter publishes it."""
        self.attrs[key] = value

    def should_prune(self) -> bool:
        """Return the test-selected pruning decision without policy evaluation."""
        return self.prune


def _load(name: str = "steady_flow_fno_search.yaml") -> optuna_runtime.OptunaStudyConfig:
    """Load one maintained study recipe through the public semantic loader."""
    return optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / name)


def test_exact_four_recipes_are_safe_complete_and_small() -> None:
    """
    Enumerate the maintained recipes and inspect model, physics, cadence, and searches.

    Exactly four small studies must exist, each base value must remain admissible,
    and continuity may vary only for physics-informed recipes.
    """
    paths = sorted(_CONFIG_ROOT.glob("*.yaml"))
    assert {path.name for path in paths} == set(_EXPECTED_RECIPES)

    for path in paths:
        expected_model, expected_physics = _EXPECTED_RECIPES[path.name]
        config = optuna_runtime.load_optuna_study_config(path)
        raw = experiments.config.loader.load_yaml(path)
        base = config.base_config
        assert base["model"]["kind"] == expected_model
        assert base["loss"]["physics"]["enabled"] is expected_physics
        assert base["evaluation"]["objective"] == _EXPECTED_OBJECTIVE
        assert base["training"]["evaluation_interval"] == 1
        assert base["data"]["num_workers"] == 0
        assert base["data"]["persistent_workers"] is False
        assert raw["study"]["pruner"] == {
            "kind": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 20,
            "interval_steps": 1,
        }
        assert "report_epochs" not in raw["study"]
        categorical_sizes = [
            len(parameter.values)
            for parameter in config.search_space
            if parameter.kind == "categorical" and parameter.path != "loss.physics.continuity"
        ]
        assert max(categorical_sizes) <= _MAX_CATEGORICAL_VALUES
        continuity = [parameter for parameter in config.search_space if parameter.path == "loss.physics.continuity"]
        if expected_physics:
            assert len(continuity) == 1
            assert continuity[0].values == ("div_velocity", "div_eps_velocity")
        else:
            assert continuity == []


def test_signature_excludes_invocation_and_tracking_but_covers_science(tmp_path: Path) -> None:
    """
    Compare one baseline signature with operational and scientific configuration drift.

    Device, paths, tracking, display name, count, and storage must remain reusable,
    while duration, optimization, split, seed, pruner, or search changes alter identity.
    """
    config = _load()
    baseline = optuna_runtime.build_study_signature(config)

    operational_base = copy.deepcopy(config.base_config)
    operational_base["run"]["device"] = "cpu"
    operational_base["paths"]["output_root"] = str(tmp_path / "elsewhere")
    operational_base["tracking"]["wandb"]["enabled"] = True
    operational_base["tracking"]["wandb"]["project"] = "different-observer"
    operational_study = copy.deepcopy(config.study)
    operational_study.update({"name": "display_name_changed", "n_trials": 97, "storage": "sqlite:///elsewhere.db"})
    operational = replace(config, base_config=operational_base, study=operational_study)
    assert optuna_runtime.build_study_signature(operational)["digest"] == baseline["digest"]

    scientific_variants: list[optuna_runtime.OptunaStudyConfig] = []
    duration = copy.deepcopy(config.base_config)
    duration["training"]["epochs"] += 1
    scientific_variants.append(replace(config, base_config=duration))
    optimizer = copy.deepcopy(config.base_config)
    optimizer["optimizer"]["lr"] = 5.0e-4
    scientific_variants.append(replace(config, base_config=optimizer))
    split = copy.deepcopy(config.base_config)
    split["data"]["train_ratio"] = 0.75
    scientific_variants.append(replace(config, base_config=split))
    seeded = copy.deepcopy(config.study)
    seeded["seed"] += 1
    scientific_variants.append(replace(config, study=seeded))
    pruned = copy.deepcopy(config.study)
    pruned["pruner"]["n_warmup_steps"] += 1
    scientific_variants.append(replace(config, study=pruned))
    parameters = list(config.search_space)
    parameters[0] = replace(parameters[0], values=(*parameters[0].values, 144))
    scientific_variants.append(replace(config, search_space=tuple(parameters)))

    assert all(optuna_runtime.build_study_signature(variant)["digest"] != baseline["digest"] for variant in scientific_variants)
    assert baseline["payload"]["task"]["contract_digest"] == config.base_config["task_contract"]["digest"]
    assert baseline["payload"]["reporting"]["step"] == "actual_completed_epoch"
    lifecycle = baseline["payload"]["trial_lifecycle"]
    assert lifecycle["resume_policy"] == "new_trials_only"
    assert lifecycle["trial_count_policy"] == "additional_fresh_trials_per_invocation"


def test_reporter_requires_held_out_metric_continuity_and_prunes_immediately() -> None:
    """
    Exercise missing objective, skipped epoch, and immediate-prune reporter paths.

    Only held-out metrics at consecutive completed epochs may enter the study, and a
    prune decision must stop at that epoch while retaining global-step evidence.
    """
    missing = optuna_runtime.OptunaEpochReporter(trial=_Trial(), objective_id="normalized_macro_rmse", direction="minimize")
    with pytest.raises(KeyError, match="Held-out Optuna objective"):
        missing(1, {"train/loss_total": 0.1})

    discontinuous = optuna_runtime.OptunaEpochReporter(trial=_Trial(), objective_id="normalized_macro_rmse", direction="minimize")
    with pytest.raises(ValueError, match="expected 1, got 2"):
        discontinuous(2, {"normalized_macro_rmse": 0.5})

    trial = _Trial(prune=True)
    reporter = optuna_runtime.OptunaEpochReporter(trial=trial, objective_id="normalized_macro_rmse", direction="minimize")
    with pytest.raises(optuna.TrialPruned, match="completed epoch 1"):
        reporter(1, {"normalized_macro_rmse": 0.5, "global_step": 7.0})
    assert trial.reports == [(0.5, 1)]
    assert trial.attrs["last_reported_epoch"] == 1
    assert trial.attrs["last_global_step"] == _EXPECTED_GLOBAL_STEP


def test_search_policy_rejects_unapproved_kinds_defaults_and_model_values() -> None:
    """
    Try fixed kinds, ranges excluding defaults, and model/physics-invalid values.

    Every candidate must fail the resolved-config allowlist or task/model contract,
    keeping searches scientifically anchored and structurally constructible.
    """
    fno = _load()
    fixed = search_space.SearchSpaceParameter(path="optimizer.lr", name="lr", kind="fixed", value=fno.base_config["optimizer"]["lr"])
    with pytest.raises(ValueError, match="does not support kind"):
        search_space.validate_search_space_paths(fno.base_config, (fixed,))

    misses_default = search_space.SearchSpaceParameter(path="optimizer.lr", name="lr", kind="float", low=1.0e-3, high=2.0e-3, log=True)
    with pytest.raises(ValueError, match="must contain its resolved base value"):
        search_space.validate_search_space_paths(fno.base_config, (misses_default,))

    physics_on_supervised = search_space.SearchSpaceParameter(
        path="loss.physics.continuity",
        name="continuity",
        kind="categorical",
        values=("div_eps_velocity",),
    )
    with pytest.raises(ValueError, match="not approved"):
        search_space.validate_search_space_paths(fno.base_config, (physics_on_supervised,))

    pi_fno = _load("steady_flow_pifno_search.yaml")
    bad_continuity = replace(
        physics_on_supervised,
        values=("div_eps_velocity", "unsupported_continuity"),
    )
    with pytest.raises(ValueError, match="unsupported by the task contract"):
        search_space.validate_search_space_paths(pi_fno.base_config, (bad_continuity,))

    uno = _load("steady_flow_uno_search.yaml")
    bad_depth = search_space.SearchSpaceParameter(
        path="model.params.n_layers",
        name="n_layers",
        kind="categorical",
        values=(5, 6),
    )
    with pytest.raises(ValueError, match="structurally supported values 5 or 7"):
        search_space.validate_search_space_paths(uno.base_config, (bad_depth,))


def test_existing_zero_trial_study_without_metadata_fails_closed(tmp_path: Path) -> None:
    """
    Create a zero-trial SQLite study without the required semantic metadata.

    Reopening must fail closed before allocating any trial leaf, preventing an empty
    unbound database from being silently adopted under the maintained scientific identity.
    """
    config = optuna_runtime.with_runtime_overrides(_load(), device="cpu", output_root=tmp_path)
    study_name = config.study["name"]
    study_dir = tmp_path / "steady_flow" / "studies" / study_name
    study_dir.mkdir(parents=True)
    storage = f"sqlite:///{study_dir / (study_name + '.db')}"
    optuna.create_study(study_name=study_name, direction="minimize", storage=storage)

    with pytest.raises(ValueError, match="missing required semantic metadata"):
        optuna_runtime.run_optuna_study(config, n_trials=1)
    assert not list(study_dir.glob("trial_*"))


def test_tiny_cpu_study_uses_actual_steps_prunes_and_resumes_new_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run and reopen a tiny CPU SQLite study with complete, failed, and pruned trials.

    Reports must use actual completed epochs and reopening must preserve history
    while allocating only additional trial numbers, protecting the no-run-resume policy.
    """
    config = optuna_runtime.with_runtime_overrides(_load(), device="cpu", output_root=tmp_path)
    settings = copy.deepcopy(config.study)
    settings["name"] = "tiny_cpu_lifecycle"
    settings["pruner"] = {
        "kind": "median",
        "n_startup_trials": 0,
        "n_warmup_steps": 0,
        "interval_steps": 1,
    }
    config = replace(config, study=settings)

    def synthetic_factory(_config: optuna_runtime.OptunaStudyConfig) -> Callable[[Any], float]:
        """Return a deterministic objective spanning complete, failed, and pruned trials."""

        def objective(trial: Any) -> float:
            """Emit trial-number-specific failure or completed-epoch observations."""
            if trial.number == 1:
                message = "explicitly recoverable synthetic failure"
                raise optuna_runtime.RecoverableTrialError(message)
            reporter = optuna_runtime.OptunaEpochReporter(
                trial=trial,
                objective_id="normalized_macro_rmse",
                direction="minimize",
            )
            values = (0.10, 0.09) if trial.number == 0 else (1.0, 0.9)
            for epoch, value in enumerate(values, start=1):
                reporter(epoch, {"normalized_macro_rmse": value, "global_step": float(epoch)})
            assert reporter.best_value is not None
            return reporter.best_value

        return objective

    monkeypatch.setattr(optuna_runtime, "create_objective", synthetic_factory)
    first = optuna_runtime.run_optuna_study(config, n_trials=2)
    first_snapshot = (
        first.trials[0].state,
        first.trials[0].value,
        dict(first.trials[0].intermediate_values),
    )
    resumed = optuna_runtime.run_optuna_study(config, n_trials=1)

    assert [trial.number for trial in resumed.trials] == [0, 1, 2]
    assert [trial.state for trial in resumed.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]
    assert first_snapshot == (
        resumed.trials[0].state,
        resumed.trials[0].value,
        dict(resumed.trials[0].intermediate_values),
    )
    assert resumed.trials[0].intermediate_values == {1: 0.10, 2: 0.09}
    assert resumed.trials[2].intermediate_values == {1: 1.0}
    assert resumed.user_attrs["semantic_signature"] == optuna_runtime.build_study_signature(config)["digest"]
    study_dir = tmp_path / "steady_flow" / "studies" / settings["name"]
    assert (study_dir / f"{settings['name']}.db").is_file()
    assert not list(study_dir.glob("trial_*"))

    def unexpected_factory(_config: optuna_runtime.OptunaStudyConfig) -> Callable[[Any], float]:
        """Return an objective that exposes one ordinary unexpected bug unchanged."""

        def objective(_trial: Any) -> float:
            """Raise the synthetic bug before producing an objective value."""
            message = "ordinary unexpected synthetic bug"
            raise RuntimeError(message)

        return objective

    monkeypatch.setattr(optuna_runtime, "create_objective", unexpected_factory)
    with pytest.raises(RuntimeError, match="ordinary unexpected synthetic bug"):
        optuna_runtime.run_optuna_study(config, n_trials=2)
    storage = f"sqlite:///{study_dir / (settings['name'] + '.db')}"
    reopened = optuna.load_study(study_name=settings["name"], storage=storage)
    assert [trial.number for trial in reopened.trials] == [0, 1, 2, 3]
    assert reopened.trials[-1].state == optuna.trial.TrialState.FAIL

    drifted_base = copy.deepcopy(config.base_config)
    drifted_base["optimizer"]["lr"] = 5.0e-4
    with pytest.raises(ValueError, match="semantic signature mismatch"):
        optuna_runtime.run_optuna_study(replace(config, base_config=drifted_base), n_trials=1)
    assert len(optuna.load_study(study_name=settings["name"], storage=storage).trials) == _EXPECTED_RESUMED_TRIAL_COUNT


def _install_running_failure_harness(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Replace expensive runtime construction while retaining trial lifecycle behavior.

    The harness builds only inert CPU objects, keeps allocation/status/summary code
    real, and injects ``error`` at the training boundary for taxonomy assertions.
    """
    processor = SimpleNamespace(state_dict=dict, in_normalizer=None, out_normalizer=None)

    def configure_reproducibility(_config: dict[str, Any], *, device: torch.device) -> dict[str, int]:
        """Return the minimum deterministic subseed mapping without changing RNGs."""
        del device
        return {"model_init": 1}

    def seed_process(_seed: int, *, device: torch.device) -> None:
        """Accept production seeding arguments without mutating process RNG state."""
        del device

    monkeypatch.setattr(experiments.run, "configure_reproducibility", configure_reproducibility)
    monkeypatch.setattr(experiments.run, "seed_process", seed_process)
    monkeypatch.setattr(
        experiments.config.loader,
        "create_dataloaders_from_config",
        lambda *_args, **_kwargs: {
            "data_processor": processor,
            "split_indices": {},
            "train": object(),
            "eval": object(),
        },
    )

    def build_model(_config: dict[str, Any], *, device: torch.device) -> torch.nn.Module:
        """Return a tiny parameterized CPU module in place of the configured model."""
        del device
        return torch.nn.Linear(1, 1)

    def build_training_loss(_config: dict[str, Any], *, device: torch.device) -> object:
        """Return an inert loss-shaped value without constructing scientific losses."""
        del device
        return SimpleNamespace()

    def build_eval_metrics(_config: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
        """Return no metrics because injected training fails before evaluation."""
        del device
        return {}

    monkeypatch.setattr(optuna_runtime.learning.models.factory, "build_model", build_model)
    monkeypatch.setattr(optuna_runtime.learning.losses.factory, "build_training_loss", build_training_loss)
    monkeypatch.setattr(optuna_runtime.learning.losses.factory, "build_eval_metrics", build_eval_metrics)
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
    monkeypatch.setattr(optuna_runtime.experiments.tracking, "build_monitor_membership", lambda *_args: None)

    def fail_training(**_kwargs: Any) -> Any:
        """Inject the caller-selected failure after the run enters running state."""
        raise error

    monkeypatch.setattr(optuna_runtime.learning.training.loop, "train_loop", fail_training)


def test_pruned_wandb_objective_is_mirrored_only_after_local_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Trigger pruning through the real reporter while recording local and W&B events.

    The authoritative pruned summary must publish before the observer mirrors that
    epoch, preventing remote telemetry from preceding durable local classification.
    """
    config = optuna_runtime.with_runtime_overrides(_load(), device="cpu", output_root=tmp_path)
    _install_running_failure_harness(RuntimeError("unused"), monkeypatch)
    events: list[str] = []

    def prune_training(**kwargs: Any) -> Any:
        """Invoke one epoch callback that must prune before training can continue."""
        callback = kwargs["epoch_end_callback"]
        callback(1, {"normalized_macro_rmse": 0.5, "global_step": 3.0})
        pytest.fail("pruning callback must stop training immediately")

    def epoch_callback(_session: Any) -> Callable[[int, dict[str, float]], None]:
        """Return a W&B-shaped callback that records only publication order."""

        def mirror(_epoch: int, _metrics: dict[str, float]) -> None:
            """Record one observer publication without performing SDK I/O."""
            events.append("wandb")

        return mirror

    monkeypatch.setattr(optuna_runtime.learning.training.loop, "train_loop", prune_training)
    monkeypatch.setattr(optuna_runtime.experiments.tracking, "epoch_callback", epoch_callback)
    monkeypatch.setattr(
        optuna_runtime,
        "_write_summary",
        lambda **kwargs: events.append(f"local:{kwargs['status']}"),
    )

    trial = _Trial(prune=True)
    with pytest.raises(optuna.TrialPruned):
        optuna_runtime.run_trial(config, trial)

    assert events == ["local:pruned", "wandb"]
    assert trial.reports == [(0.5, 1)]


@pytest.mark.parametrize(
    ("error_factory", "expected_status", "expected_error"),
    [
        pytest.param(
            lambda: torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB"),
            "oom_pruned",
            optuna.TrialPruned,
            id="torch-cuda-oom",
        ),
        pytest.param(lambda: MemoryError("allocator exhausted"), "oom_pruned", optuna.TrialPruned, id="memory-error"),
        pytest.param(
            lambda: RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"),
            "oom_pruned",
            optuna.TrialPruned,
            id="known-runtime-oom",
        ),
        pytest.param(
            lambda: RuntimeError("feature label says out of memory but allocator is healthy"),
            "failed",
            RuntimeError,
            id="unrelated-runtime-message",
        ),
        pytest.param(
            lambda: optuna_runtime.RecoverableTrialError("narrow recoverable trial failure"),
            "recoverable_failed",
            optuna_runtime.RecoverableTrialError,
            id="recoverable",
        ),
        pytest.param(lambda: FloatingPointError("non-finite at epoch 3: nan"), "nonfinite_pruned", optuna.TrialPruned, id="nonfinite"),
        pytest.param(lambda: RuntimeError("ordinary programming failure"), "failed", RuntimeError, id="unexpected"),
        pytest.param(lambda: KeyboardInterrupt("stop"), "interrupted", KeyboardInterrupt, id="keyboard"),
        pytest.param(lambda: SystemExit(7), "interrupted", SystemExit, id="system-exit"),
    ],
)
def test_running_trial_failure_taxonomy_is_precise_and_cpu_cleanup_is_guarded(
    error_factory: Callable[[], BaseException],
    expected_status: str,
    expected_error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject OOM variants, non-finite/recoverable failures, bugs, and interrupts on CPU.

    Each must publish its exact terminal status and propagation class; cleanup must
    avoid every CUDA query when the resolved trial device is CPU.
    """
    config = optuna_runtime.with_runtime_overrides(_load(), device="cpu", output_root=tmp_path)
    error = error_factory()
    _install_running_failure_harness(error, monkeypatch)

    def cuda_query_forbidden() -> bool:
        """Fail if CPU cleanup crosses the CUDA initialization boundary."""
        pytest.fail("CPU trial cleanup must not query or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_query_forbidden)
    trial = _Trial(number=0)
    with pytest.raises(expected_error):
        optuna_runtime.run_trial(config, trial)

    run_dir = common.paths.resolve_optuna_trial_dir(
        "steady_flow",
        config.study["name"],
        trial.number,
        output_root=tmp_path,
    )
    summary = experiments.run.read_run_summary(run_dir)
    assert summary["status"] == expected_status
    assert summary["failure_context"]["error_type"] == type(error).__name__
    assert summary["sampled_parameters"] == trial.attrs["overrides"]
    assert not list(run_dir.glob("artifacts*"))


def test_run_status_set_contains_every_terminal_outcome() -> None:
    """
    Compare the public run-status set with every initial, active, and terminal outcome.

    Exact equality must hold so summary validation neither rejects a classified trial
    nor admits an undocumented lifecycle state.
    """
    expected = {
        "initializing",
        "running",
        "completed",
        "pruned",
        "nonfinite_pruned",
        "oom_pruned",
        "recoverable_failed",
        "failed",
        "interrupted",
    }
    assert expected == experiments.run.RUN_STATUSES
