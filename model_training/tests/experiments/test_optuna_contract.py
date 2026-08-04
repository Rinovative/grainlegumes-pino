# ruff: noqa: S101, SLF001
"""Protect Optuna semantics with artificial configs and controlled fake trials."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optuna
import pytest
from src import experiments
from support import configs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

optuna_runtime = experiments.tuning.optuna
search_space = experiments.tuning.search_space


class _FakeTrial:
    """Return explicit test-owned samples while recording suggestion APIs."""

    def __init__(
        self,
        selections: dict[str, Any],
        *,
        number: int = 4,
    ) -> None:
        self.number = number
        self.selections = selections
        self.calls: dict[str, tuple[str, object]] = {}
        self.attrs: dict[str, Any] = {}

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> Any:
        self.calls[name] = ("categorical", tuple(choices))
        return self.selections[name]

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        self.calls[name] = ("float", (low, high, log, step))
        return float(self.selections[name])

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        self.calls[name] = ("int", (low, high, log, step))
        return int(self.selections[name])

    def report(self, value: float, step: int) -> None:
        """Accept reporter calls outside suggestion-focused assertions."""
        del value, step

    def set_user_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def should_prune(self) -> bool:
        """Keep controlled suggestion trials unpruned."""
        return False


def _write_study(
    tmp_path: Path,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write one compact test-owned study request."""
    return configs.write_yaml(
        tmp_path / "synthetic-study.yaml",
        payload or configs.optuna_config(),
    )


def test_minimal_optuna_config_resolves_without_production_files(
    tmp_path: Path,
) -> None:
    """Resolve complete study, objective, and search policy from artificial YAML."""
    configured = configs.optuna_config(multivariate=False)
    study = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))

    assert study.study["name"] == configured["study"]["name"]
    assert study.study["seed"] == configured["study"]["seed"]
    assert study.study["sampler"]["multivariate"] is (configured["study"]["sampler"]["multivariate"])
    assert study.base_config["data"]["train_dataset"] == "synthetic_train"
    assert study.base_config["run"]["seed"] == (configured["experiment"]["run"]["seed"])
    objective = experiments.config.loader.get_resolved_objective(study.base_config)
    assert study.study["objective"] == objective["id"]
    assert study.study["direction"] == objective["direction"]


def test_fake_trial_uses_typed_apis_and_persists_applied_values(
    tmp_path: Path,
) -> None:
    """Apply explicit categorical, float, and integer samples to one trial."""
    configured = configs.optuna_config()
    configured["search_space"]["data.batch_size"] = {
        "name": "batch_size",
        "kind": "int",
        "low": 2,
        "high": 6,
        "step": 2,
    }
    study = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))
    original = copy.deepcopy(study.base_config)
    selections = {
        "hidden_channels": 12,
        "learning_rate": 5.0e-3,
        "batch_size": 4,
    }
    trial = _FakeTrial(selections)

    resolved, context = optuna_runtime._prepare_trial_config(study, trial)

    assert trial.calls["hidden_channels"][0] == "categorical"
    assert trial.calls["learning_rate"][0] == "float"
    assert trial.calls["batch_size"][0] == "int"
    assert resolved["model"]["params"]["hidden_channels"] == selections["hidden_channels"]
    assert resolved["optimizer"]["lr"] == pytest.approx(5.0e-3)
    assert resolved["data"]["batch_size"] == selections["batch_size"]
    assert study.base_config == original
    assert context["overrides"] == {
        "model.params.hidden_channels": 12,
        "optimizer.lr": 5.0e-3,
        "data.batch_size": 4,
    }
    assert trial.attrs["overrides"] == context["overrides"]
    assert trial.attrs["run_name"] == resolved["run"]["name"]
    assert experiments.config.loader.resolved_model_variant(resolved) in (resolved["run"]["name"])
    assert resolved["run"]["name"].endswith("__optuna_trial_004")


@pytest.mark.parametrize(
    ("spec", "error", "match"),
    [
        (
            {
                "name": "rate",
                "kind": "float",
                "low": True,
                "high": 1.0,
            },
            TypeError,
            "finite number",
        ),
        (
            {
                "name": "rate",
                "kind": "float",
                "low": 1.0,
                "high": 0.5,
            },
            ValueError,
            "low < high",
        ),
        (
            {
                "name": "rate",
                "kind": "unsupported",
                "low": 0.1,
                "high": 1.0,
            },
            ValueError,
            "must be one of",
        ),
    ],
)
def test_invalid_search_spec_fails_before_sampling(
    spec: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    """Reject representative type, range, and kind errors in the parser."""
    with pytest.raises(error, match=match):
        search_space.parse_search_space({"optimizer.lr": spec})


def test_unapproved_search_path_is_rejected_during_load(
    tmp_path: Path,
) -> None:
    """Prevent a study from sampling task-owned objective semantics."""
    configured = configs.optuna_config()
    configured["search_space"]["evaluation.objective.id"] = {
        "name": "objective",
        "kind": "categorical",
        "values": ["normalized_group_macro_rmse"],
    }

    with pytest.raises(ValueError, match="not approved"):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["study"].update({"seed": 1.0}),
        lambda payload: payload["study"].update({"n_trials": True}),
        lambda payload: payload["study"]["sampler"].update({"multivariate": 1}),
    ],
)
def test_study_scalars_require_exact_types(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    """Reject scalar coercion before creating sampler or storage state."""
    configured = configs.optuna_config()
    mutation(configured)

    with pytest.raises(TypeError):
        optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))


def test_sampler_pruner_and_metadata_mirror_configured_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass configured sampler and pruner settings through without snapshots."""
    configured = configs.optuna_config(multivariate=False)
    loaded = optuna_runtime.load_optuna_study_config(_write_study(tmp_path, configured))
    external_study = copy.deepcopy(loaded.study)
    external_study["storage"] = "external://synthetic-study"
    loaded = replace(loaded, study=external_study)
    captured: dict[str, Any] = {}

    def capture_sampler(**settings: Any) -> object:
        captured["sampler"] = settings
        return object()

    def capture_pruner(**settings: Any) -> object:
        captured["pruner"] = settings
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
    monkeypatch.setattr(optuna.pruners, "MedianPruner", capture_pruner)
    monkeypatch.setattr(optuna, "create_study", create_study)

    result = optuna_runtime.run_optuna_study(
        loaded,
        n_trials=1,
        output_root=tmp_path / "outputs",
    )

    sampler_config = loaded.study["sampler"]
    pruner_config = loaded.study["pruner"]
    assert result is fake_study
    assert captured["sampler"] == {
        "seed": loaded.study["seed"],
        "multivariate": sampler_config["multivariate"],
    }
    assert captured["pruner"] == {
        "n_startup_trials": pruner_config["n_startup_trials"],
        "n_warmup_steps": pruner_config["n_warmup_steps"],
        "interval_steps": pruner_config["interval_steps"],
    }
    assert user_attrs[optuna_runtime._SAMPLER_METADATA_ATTR] == {
        "kind": sampler_config["kind"],
        "multivariate": sampler_config["multivariate"],
        "seed": loaded.study["seed"],
    }
    assert captured["create_study"]["sampler"] is not None
    assert captured["create_study"]["pruner"] is not None
