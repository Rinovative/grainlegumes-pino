# ruff: noqa: S101
"""Verify one resolved objective controls Optuna configuration and runtime."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
from src import experiments

if TYPE_CHECKING:
    from collections.abc import Sequence

_CONFIG_ROOT = Path(__file__).parents[2] / "configs" / "optuna"
_OPTUNA_CONFIGS = sorted(_CONFIG_ROOT.glob("*.yaml"))
_EXPECTED_OBJECTIVE = {
    "id": "normalized_relative_h1",
    "kind": "relative_h1",
    "space": "normalized",
    "fields": ["p", "u", "v"],
    "reduction": "sample_mean",
    "direction": "minimize",
}


optuna_runtime = experiments.tuning.optuna
search_space = experiments.tuning.search_space


class _Trial:
    """Minimal deterministic trial used by search-space and reporter tests."""

    def __init__(self, *, number: int = 3, prune: bool = False) -> None:
        self.number = number
        self.prune = prune
        self.attrs: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
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


@pytest.mark.parametrize("path", _OPTUNA_CONFIGS, ids=lambda path: path.stem)
def test_every_optuna_yaml_resolves_one_complete_objective(path: Path) -> None:
    """Study ID/direction and all metric semantics derive from the experiment."""
    config = optuna_runtime.load_optuna_study_config(path)
    raw = experiments.config.loader.load_yaml(path)

    assert "objective" not in raw["study"]
    assert "direction" not in raw["study"]
    assert "hard_prune_spike" not in raw["study"]
    assert experiments.config.loader.get_resolved_objective(config.base_config) == _EXPECTED_OBJECTIVE
    assert config.study["objective"] == _EXPECTED_OBJECTIVE["id"]
    assert config.study["direction"] == _EXPECTED_OBJECTIVE["direction"]
    assert "budgets" not in config.study
    assert config.study["report_epochs"] == [300, 400]
    assert "seed" not in config.study["sampler"]


def test_sampler_seed_is_a_stable_subseed_of_the_study_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public study orchestration passes the derived sampler seed to Optuna."""
    study = optuna_runtime.load_optuna_study_config(
        _CONFIG_ROOT / "steady_flow_fno_search.yaml",
    )
    external_settings = copy.deepcopy(study.study)
    external_settings["storage"] = "external://sampler-seed-test"
    study = replace(study, study=external_settings)
    output_root = tmp_path / "outputs"
    captured: dict[str, Any] = {}

    def capture_sampler(**settings: Any) -> object:
        captured["sampler"] = settings
        return object()

    user_attrs: dict[str, Any] = {}
    fake_study = SimpleNamespace(
        direction=SimpleNamespace(name="MINIMIZE"),
        user_attrs=user_attrs,
        trials=[],
        set_user_attr=user_attrs.__setitem__,
        optimize=lambda *_args, **_kwargs: None,
    )

    def create_study(**settings: Any) -> Any:
        captured["create_study"] = settings
        return fake_study

    monkeypatch.setattr(optuna.samplers, "TPESampler", capture_sampler)
    monkeypatch.setattr(optuna.pruners, "MedianPruner", lambda **_settings: object())
    monkeypatch.setattr(optuna, "create_study", create_study)

    result = optuna_runtime.run_optuna_study(
        study,
        n_trials=1,
        output_root=output_root,
    )

    assert result is fake_study
    assert captured["sampler"]["seed"] == experiments.run.derive_subseed(
        9,
        "optuna-sampler",
    )
    assert captured["create_study"]["sampler"] is not None
    assert not output_root.exists()


def test_search_space_uses_exact_per_kind_schemas() -> None:
    """Each search kind accepts only its explicit, fully named YAML schema."""
    invalid_specs: list[tuple[dict[str, Any], type[Exception], str]] = [
        (
            {"name": "choice", "kind": "categorical", "values": [1], "low": 0},
            ValueError,
            "invalid for categorical",
        ),
        (
            {"name": "constant", "kind": "fixed", "value": 1, "log": False},
            ValueError,
            "invalid for fixed",
        ),
        (
            {"name": "rate", "kind": "float", "low": 0.1, "high": 1.0, "values": [0.5]},
            ValueError,
            "invalid for float",
        ),
        (
            {"name": "count", "kind": "int", "low": 1, "high": 3, "value": 2},
            ValueError,
            "invalid for int",
        ),
        ({"kind": "float", "low": 0.1, "high": 1.0}, KeyError, "name is required"),
        ({"name": "rate", "low": 0.1, "high": 1.0}, KeyError, "kind is required"),
        ({"name": " ", "kind": "fixed", "value": 1}, TypeError, "non-empty string"),
        (
            {"name": "rate", "kind": "FLOAT", "low": 0.1, "high": 1.0},
            ValueError,
            "must be one of",
        ),
        ({"name": "choice", "kind": "categorical", "values": [1, 1]}, ValueError, "must be unique"),
        ({"name": "choice", "kind": "categorical", "values": [1 + 2j]}, TypeError, "supported scalar"),
        ({"name": "choice", "kind": "categorical", "values": [float("nan")]}, ValueError, "finite choices"),
    ]

    for spec, error, match in invalid_specs:
        with pytest.raises(error, match=match):
            search_space.parse_search_space({"optimizer.lr": spec})


def test_numeric_search_specs_reject_coercion_and_invalid_ranges() -> None:
    """Numeric suggestions are finite, ordered, and type exact before Optuna sees them."""
    invalid_specs: list[tuple[dict[str, Any], type[Exception], str]] = [
        ({"name": "rate", "kind": "float", "low": "0.1", "high": 1.0}, TypeError, "finite number"),
        ({"name": "rate", "kind": "float", "low": False, "high": 1.0}, TypeError, "finite number"),
        (
            {"name": "rate", "kind": "float", "low": 0.1, "high": float("inf")},
            ValueError,
            "must be finite",
        ),
        ({"name": "rate", "kind": "float", "low": 1.0, "high": 1.0}, ValueError, "low < high"),
        ({"name": "rate", "kind": "float", "low": 2.0, "high": 1.0}, ValueError, "low < high"),
        (
            {"name": "rate", "kind": "float", "low": 0.0, "high": 1.0, "log": True},
            ValueError,
            "positive bounds",
        ),
        (
            {"name": "rate", "kind": "float", "low": 0.1, "high": 1.0, "log": 1},
            TypeError,
            "must be a boolean",
        ),
        (
            {"name": "rate", "kind": "float", "low": 0.1, "high": 1.0, "step": 0.0},
            ValueError,
            "step must be positive",
        ),
        (
            {"name": "rate", "kind": "float", "low": 0.1, "high": 1.0, "step": float("nan")},
            ValueError,
            "must be finite",
        ),
        (
            {"name": "rate", "kind": "float", "low": 0.1, "high": 1.0, "step": 0.1, "log": True},
            ValueError,
            "cannot combine",
        ),
        ({"name": "count", "kind": "int", "low": 1.0, "high": 3}, TypeError, "must be an integer"),
        (
            {"name": "count", "kind": "int", "low": 1, "high": 3, "step": 1.0},
            TypeError,
            "must be an integer",
        ),
        (
            {"name": "count", "kind": "int", "low": 1, "high": 3, "step": -1},
            ValueError,
            "step must be positive",
        ),
        ({"name": "count", "kind": "int", "low": 1, "high": 4, "step": 2}, ValueError, "divide"),
        ({"name": "rate", "kind": "float", "low": 0.0, "high": 1.0, "step": 0.3}, ValueError, "divide"),
    ]

    for spec, error, match in invalid_specs:
        with pytest.raises(error, match=match):
            search_space.parse_search_space({"optimizer.lr": spec})


def test_study_scalars_are_type_exact_during_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Study scalars cannot enter runtime through YAML truthiness or integer coercion."""
    source_path = _CONFIG_ROOT / "steady_flow_fno_search.yaml"
    load_yaml = experiments.config.loader.load_yaml
    base_raw = load_yaml(source_path)
    invalid_settings: list[tuple[str, Any, type[Exception], str]] = [
        ("name", " ", TypeError, "study.name"),
        ("name", "../escape", ValueError, "single non-empty path component"),
        ("seed", True, TypeError, "study.seed"),
        ("seed", "9", TypeError, "study.seed"),
        ("n_trials", 3.0, TypeError, "study.n_trials"),
        ("report_epochs", [300.0], TypeError, "report_epochs entry"),
        ("report_epochs", [True], TypeError, "report_epochs entry"),
        ("report_epochs", [], ValueError, "must not be empty"),
        ("report_epochs", [300, 300], ValueError, "unique and strictly increasing"),
        ("report_epochs", [400, 300], ValueError, "unique and strictly increasing"),
        ("report_epochs", [301], ValueError, "reachable evaluation epochs"),
        ("report_epochs", [405], ValueError, "reachable evaluation epochs"),
        ("storage", " ", TypeError, "study.storage"),
        ("storage", 3, TypeError, "study.storage"),
    ]

    for study_key, invalid_value, error, match in invalid_settings:
        raw = copy.deepcopy(base_raw)
        raw["study"][study_key] = invalid_value
        monkeypatch.setattr(experiments.config.loader, "load_yaml", lambda _path, raw=raw: raw)
        with pytest.raises(error, match=match):
            optuna_runtime.load_optuna_study_config(source_path)


def test_derived_objective_direction_and_bespoke_spike_are_rejected_as_raw_study_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Objective semantics derive only from the resolved experiment."""
    source_path = _CONFIG_ROOT / "steady_flow_fno_search.yaml"
    base_raw = experiments.config.loader.load_yaml(source_path)

    for key, value in (
        ("objective", "normalized_relative_h1"),
        ("direction", "minimize"),
        ("hard_prune_spike", True),
    ):
        raw = copy.deepcopy(base_raw)
        raw["study"][key] = value
        monkeypatch.setattr(
            experiments.config.loader,
            "load_yaml",
            lambda _path, raw=raw: raw,
        )
        with pytest.raises(ValueError, match="study contains unknown key"):
            optuna_runtime.load_optuna_study_config(source_path)


def test_sampler_and_pruner_semantics_are_validated_during_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only explicit supported component kinds and kind-applicable settings load."""
    source_path = _CONFIG_ROOT / "steady_flow_fno_search.yaml"
    load_yaml = experiments.config.loader.load_yaml
    base_raw = load_yaml(source_path)
    invalid_components: list[tuple[str, Any, type[Exception], str]] = [
        ("sampler", "tpe", TypeError, "string shorthand"),
        ("sampler", {"kind": "tpemultivariate"}, ValueError, "Unsupported Optuna sampler"),
        ("sampler", {"kind": "tpe", "multivariate": 1}, TypeError, "must be a boolean"),
        (
            "sampler",
            {"kind": "random", "multivariate": False},
            ValueError,
            "only valid for the tpe",
        ),
        ("pruner", "median", TypeError, "string shorthand"),
        ("pruner", {"kind": "nop"}, ValueError, "Unsupported Optuna pruner"),
        ("pruner", {"kind": "noop"}, ValueError, "Unsupported Optuna pruner"),
        ("pruner", {"kind": "none", "n_startup_trials": 0}, ValueError, "invalid for none"),
        ("pruner", {"kind": "median", "n_startup_trials": -1}, ValueError, "at least 0"),
        ("pruner", {"kind": "median", "n_warmup_steps": 0.0}, TypeError, "must be an integer"),
        ("pruner", {"kind": "median", "interval_steps": 0}, ValueError, "at least 1"),
    ]

    for component, invalid_config, error, match in invalid_components:
        raw = copy.deepcopy(base_raw)
        raw["study"][component] = invalid_config
        monkeypatch.setattr(experiments.config.loader, "load_yaml", lambda _path, raw=raw: raw)
        with pytest.raises(error, match=match):
            optuna_runtime.load_optuna_study_config(source_path)


def test_storage_none_and_exact_valid_search_specs_remain_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict parsing preserves documented optional storage and all four search kinds."""
    source_path = _CONFIG_ROOT / "steady_flow_fno_search.yaml"
    raw = experiments.config.loader.load_yaml(source_path)
    raw["study"]["storage"] = None
    monkeypatch.setattr(experiments.config.loader, "load_yaml", lambda _path: raw)

    study = optuna_runtime.load_optuna_study_config(source_path)
    parsed = search_space.parse_search_space(
        {
            "a": {"name": "a", "kind": "categorical", "values": [1, 2]},
            "b": {"name": "b", "kind": "float", "low": 0, "high": 1, "step": 0.1},
            "c": {"name": "c", "kind": "int", "low": 1, "high": 3, "step": 1},
            "d": {"name": "d", "kind": "fixed", "value": None},
        }
    )

    assert study.study["storage"] is None
    assert [parameter.kind for parameter in parsed] == ["categorical", "float", "int", "fixed"]


def test_sampled_trial_config_is_resolved_and_revalidates_report_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public trial orchestration validates config, seed, metadata, and schedule."""
    study = optuna_runtime.load_optuna_study_config(
        _CONFIG_ROOT / "steady_flow_fno_search.yaml",
    )
    trial_number = 7
    trial = _Trial(number=trial_number)
    captured: dict[str, Any] = {}

    class PreparedTrialCaptureError(RuntimeError):
        """Stop after public trial preparation, before output allocation."""

    def capture_prepared_run(
        config: dict[str, Any],
        *,
        run_dir: Path,
        summary_extra: dict[str, Any],
    ) -> Path:
        captured["config"] = config
        captured["run_dir"] = run_dir
        captured["context"] = summary_extra
        raise PreparedTrialCaptureError

    monkeypatch.setattr(
        experiments.run,
        "prepare_fresh_run",
        capture_prepared_run,
    )
    with pytest.raises(PreparedTrialCaptureError):
        optuna_runtime.run_trial(study, trial)

    config = captured["config"]
    context = captured["context"]
    assert "optuna" not in config
    assert experiments.config.loader.validate_resolved_config(config) == config
    assert experiments.config.loader.get_resolved_objective(config) == _EXPECTED_OBJECTIVE
    assert context["study_name"] == study.study["name"]
    assert context["trial_number"] == trial_number
    assert config["run"]["seed"] == experiments.run.derive_subseed(
        9,
        f"trial-{trial_number}",
    )
    assert trial.attrs["run_seed"] == config["run"]["seed"]
    assert "capacity" not in trial.attrs

    output_root = tmp_path / "outputs"
    trial_base = copy.deepcopy(study.base_config)
    trial_base["paths"]["output_root"] = str(output_root)
    unreachable_schedule = replace(
        study,
        base_config=trial_base,
        search_space=(
            search_space.SearchSpaceParameter(
                path="training.epochs",
                name="epochs",
                kind="fixed",
                value=1,
            ),
        ),
    )
    invalid_trial = _Trial(number=trial_number + 1)
    with pytest.raises(ValueError, match="reachable evaluation epochs"):
        optuna_runtime.run_trial(unreachable_schedule, invalid_trial)
    assert invalid_trial.attrs == {}
    assert not output_root.exists()


def test_objective_and_derived_search_paths_are_rejected() -> None:
    """Search spaces cannot alter objective, task, path, or derived channel identity."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    forbidden = (
        "evaluation.objective.direction",
        "evaluation.metrics.0.kind",
        "task",
        "paths.dataset_root",
        "model.params.in_channels",
        "run.name",
        "run.seed",
        "run.suffix",
    )

    for path in forbidden:
        parameter = search_space.SearchSpaceParameter(
            path=path,
            name=path,
            kind="fixed",
            value="invalid",
        )
        with pytest.raises(ValueError, match=r"immutable|derived"):
            search_space.validate_search_space_paths(study.base_config, (parameter,))


def test_resolved_metric_drift_is_rejected_even_when_semantically_valid() -> None:
    """Changing relative H1 to relative L2 cannot retain the selected objective signature."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    drifted = copy.deepcopy(study.base_config)
    drifted["evaluation"]["metrics"][0]["kind"] = "relative_l2"

    with pytest.raises(ValueError, match="must exactly equal"):
        experiments.config.loader.validate_resolved_config(drifted)


def test_resolved_metric_unknown_keys_are_rejected() -> None:
    """Resolved metric mappings use the exact canonical semantic schema."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    drifted = copy.deepcopy(study.base_config)
    drifted["evaluation"]["metrics"][0]["unexpected"] = "value"

    with pytest.raises(ValueError, match="must contain exactly"):
        experiments.config.loader.validate_resolved_config(drifted)


@pytest.mark.parametrize(
    ("direction", "values", "best_epoch", "best_value"),
    [
        ("minimize", [1.0, 2.0], 1, 1.0),
        ("maximize", [1.0, 2.0], 2, 2.0),
    ],
)
def test_reporter_tracks_every_evaluation_in_both_directions(
    direction: str,
    values: list[float],
    best_epoch: int,
    best_value: float,
) -> None:
    """A filtered reporter retains the same best state as checkpoint selection."""
    trial = _Trial()
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=trial,
        objective_id="objective",
        direction=direction,
        report_epochs={2},
    )

    reporter(1, {"objective": values[0]})
    reporter(2, {"objective": values[1]})

    assert reporter.best_epoch == best_epoch
    assert reporter.best_value == best_value
    assert trial.reports == [(values[1], 0)]


def test_reporter_prunes_non_finite_objectives_explicitly() -> None:
    """NaN never reaches Optuna as an ordinary comparable value."""
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=_Trial(),
        objective_id="objective",
        direction="minimize",
        report_epochs=set(),
    )

    with pytest.raises(optuna.TrialPruned, match="Non-finite Optuna objective"):
        reporter(1, {"objective": float("nan")})


@pytest.mark.parametrize(
    ("failure_stage", "expected_status", "expected_error"),
    [
        ("before_running", "failed", FloatingPointError),
        ("after_running", "pruned", optuna.TrialPruned),
    ],
)
def test_run_trial_classifies_floating_point_failures_by_lifecycle(
    failure_stage: str,
    expected_status: str,
    expected_error: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only floating-point failures after entering running become pruned trials."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    statuses: list[str] = []
    transitions: list[str] = []
    processor = SimpleNamespace(
        state_dict=dict,
        in_normalizer=None,
        out_normalizer=None,
    )

    def create_dataloaders(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if failure_stage == "before_running":
            message = "non-finite setup value"
            raise FloatingPointError(message)
        return {
            "data_processor": processor,
            "split_indices": {},
            "train": object(),
            "eval": object(),
        }

    def fail_training(**_kwargs: Any) -> None:
        message = "non-finite objective"
        raise FloatingPointError(message)

    monkeypatch.setattr(
        experiments.run,
        "prepare_fresh_run",
        lambda *_args, **_kwargs: tmp_path / "trial",
    )
    monkeypatch.setattr(experiments.run, "configure_reproducibility", lambda _config: {"model_init": 1})
    monkeypatch.setattr(experiments.run, "seed_process", lambda _seed: None)
    monkeypatch.setattr(
        experiments.run,
        "transition_run_status",
        lambda _run_dir, status, **_kwargs: transitions.append(status),
    )
    monkeypatch.setattr(
        experiments.config.loader,
        "create_dataloaders_from_config",
        create_dataloaders,
    )
    monkeypatch.setattr(optuna_runtime.common.serialization, "atomic_torch_save", lambda *_args: None)
    monkeypatch.setattr(optuna_runtime.learning.models.factory, "build_model", lambda _config: object())
    monkeypatch.setattr(
        optuna_runtime.learning.losses.factory,
        "build_training_loss",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        optuna_runtime.learning.losses.factory,
        "build_eval_metrics",
        lambda _config: {},
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.optim,
        "build_optimizer",
        lambda _model, _config: object(),
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.optim,
        "build_scheduler",
        lambda _optimizer, _config: object(),
    )
    monkeypatch.setattr(
        optuna_runtime.learning.training.checkpoint,
        "build_checkpoint_identity",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(optuna_runtime.learning.training.loop, "train_loop", fail_training)
    monkeypatch.setattr(
        optuna_runtime,
        "_write_summary",
        lambda **kwargs: statuses.append(kwargs["status"]),
    )

    with pytest.raises(expected_error, match="non-finite"):
        optuna_runtime.run_trial(study, _Trial())

    assert statuses == [expected_status]
    assert transitions == ([] if failure_stage == "before_running" else ["running"])


def test_run_trial_revalidates_study_and_model_before_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct trial callers cannot allocate output from a drifted study config."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    invalid_settings = copy.deepcopy(study.study)
    invalid_settings["n_trials"] = 0
    trial = _Trial()
    monkeypatch.setattr(
        experiments.run,
        "prepare_fresh_run",
        lambda *_args, **_kwargs: pytest.fail("prepare_fresh_run must not run"),
    )

    with pytest.raises(ValueError, match=r"study\.n_trials"):
        optuna_runtime.run_trial(replace(study, study=invalid_settings), trial)

    invalid_base = copy.deepcopy(study.base_config)
    invalid_base["model"] = copy.deepcopy(invalid_base["model"])
    invalid_base["model"]["kind"] = "uno"
    invalid_base["model"]["params"] = {
        "in_channels": 7,
        "out_channels": 3,
        "hidden_channels": 4,
        "modes_x": 8,
        "modes_y": 8,
        "n_layers": 3,
        "channel_mlp_skip": "linear",
    }
    invalid_trial = _Trial(number=trial.number + 1)
    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        optuna_runtime.run_trial(
            replace(study, base_config=invalid_base),
            invalid_trial,
        )

    assert trial.attrs == {}
    assert invalid_trial.attrs == {}


def test_invalid_runtime_study_settings_fail_before_output_allocation(tmp_path: Path) -> None:
    """Trial count and sampler kind are validated before creating a study directory."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="n_trials must be positive"):
        optuna_runtime.run_optuna_study(study, n_trials=0, output_root=output_root)
    assert not output_root.exists()

    invalid_study = copy.deepcopy(study.study)
    invalid_study["sampler"] = {"kind": "unknown"}
    with pytest.raises(ValueError, match="Unsupported Optuna sampler"):
        optuna_runtime.run_optuna_study(
            replace(study, study=invalid_study),
            output_root=output_root,
        )
    assert not output_root.exists()


def test_study_mismatch_fails_before_output_allocation(tmp_path: Path) -> None:
    """Programmatic study drift is revalidated before any study path is created."""
    study = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    drifted = copy.deepcopy(study.study)
    drifted["direction"] = "maximize"
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="direction does not match"):
        optuna_runtime.run_optuna_study(
            replace(study, study=drifted),
            output_root=output_root,
        )
    assert not output_root.exists()


def test_existing_study_direction_is_checked_before_optimization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing opposite-direction study is never optimized under this objective."""
    study_config = optuna_runtime.load_optuna_study_config(_CONFIG_ROOT / "steady_flow_fno_search.yaml")
    external_study = copy.deepcopy(study_config.study)
    external_study["storage"] = "external://existing-study"
    study_config = replace(study_config, study=external_study)
    output_root = tmp_path / "outputs"
    existing = SimpleNamespace(
        direction=SimpleNamespace(name="MAXIMIZE"),
        user_attrs={"resolved_objective": _EXPECTED_OBJECTIVE},
        trials=[object()],
        optimize=lambda *_args, **_kwargs: pytest.fail("optimize must not run"),
    )
    fake_optuna = SimpleNamespace(create_study=lambda **_kwargs: existing)
    monkeypatch.setattr(optuna.pruners, "MedianPruner", lambda **_settings: object())
    monkeypatch.setattr(optuna.samplers, "TPESampler", lambda **_settings: object())
    monkeypatch.setattr(optuna, "create_study", fake_optuna.create_study)

    with pytest.raises(ValueError, match="Existing Optuna study direction"):
        optuna_runtime.run_optuna_study(study_config, output_root=output_root)
    assert not output_root.exists()
