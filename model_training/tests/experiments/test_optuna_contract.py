# ruff: noqa: S101, SLF001
"""
Protect objective-driven Optuna configuration, search parsing, and trial contracts.

Fake trials/studies exercise exact YAML schemas, fixed training and sampler seeds,
device/output identity exclusion, suggestion application, reporting, pruning, allocation, and
W&B-independent objective consumption. Real tiny SQLite continuation and the
complete failure taxonomy are covered by ``test_optuna_lifecycle``.
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
from src import datasets, experiments, learning
from support import configs

if TYPE_CHECKING:
    from collections.abc import Sequence

_OPTUNA_CONFIGS = configs.optuna_config_paths()
_FNO_CONFIG = configs.optuna_config_path(model_kind="fno", physics_enabled=False)
_MIN_SMOKE_TRIALS = 2
_EXPECTED_OBJECTIVE = {
    "id": "normalized_macro_rmse",
    "kind": "macro_rmse",
    "space": "normalized",
    "fields": ["p", "u", "v"],
    "reduction": "field_macro_element_mean",
    "direction": "minimize",
}


optuna_runtime = experiments.tuning.optuna
search_space = experiments.tuning.search_space


class _Trial:
    """
    Implement only the Optuna trial surface exercised by contract tests.

    Suggestions deterministically choose the first or lower-bound value; reports and
    user attributes are retained in memory, and one toggle controls pruning. No
    distribution validation, persistence, or real sampling occurs here.
    """

    def __init__(self, *, number: int = 3, prune: bool = False) -> None:
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


def test_every_optuna_yaml_resolves_one_complete_objective() -> None:
    """Validate every wrapper by schema, relationships, bounds, and owned identity."""
    names_by_role: dict[str, set[str]] = {"production": set(), "smoke": set()}
    trials_by_role: dict[str, list[int]] = {"production": [], "smoke": []}
    epochs_by_role: dict[str, list[int]] = {"production": [], "smoke": []}

    for path in _OPTUNA_CONFIGS:
        config = optuna_runtime.load_optuna_study_config(path)
        raw = experiments.config.loader.load_yaml(path)
        base = config.base_config
        objective = experiments.config.loader.get_resolved_objective(base)
        raw_study = raw["study"]
        role = config.study["role"]
        n_trials = config.study["n_trials"]
        epochs = base["training"]["epochs"]
        cadence = base["training"]["evaluation_interval"]
        pruner = config.study["pruner"]

        assert role in names_by_role
        assert config.study["role"] == raw_study["role"]
        assert isinstance(config.study["name"], str)
        assert config.study["name"]
        assert config.study["name"] not in names_by_role[role]
        names_by_role[role].add(config.study["name"])
        trials_by_role[role].append(n_trials)
        epochs_by_role[role].append(epochs)

        assert tuple(raw) == ("study", "experiment", "search_space")
        assert tuple(raw["experiment"]) == experiments.config.loader.CANONICAL_EXPERIMENT_SECTION_ORDER
        assert "objective" not in raw_study
        assert "direction" not in raw_study
        assert config.base_experiment == raw["experiment"]
        assert raw["experiment"]["evaluation"] == {"objective": {"id": objective["id"]}}
        assert config.study["objective"] == objective["id"]
        assert config.study["direction"] == objective["direction"]

        training_seed = base["run"]["seed"]
        sampler_seed = config.study["seed"]
        assert type(training_seed) is int
        assert type(sampler_seed) is int
        assert training_seed == raw["experiment"]["run"]["seed"]
        assert "prefix" not in raw["experiment"]["run"]
        assert sampler_seed == raw_study["seed"]
        assert "seed" not in config.study["sampler"]

        assert type(n_trials) is int
        assert n_trials > 0
        assert type(epochs) is int
        assert epochs > 0
        assert type(cadence) is int
        assert cadence > 0
        assert base["training"]["ood_evaluation_interval"] == cadence
        assert base["tracking"]["wandb"]["monitor"]["interval"] == cadence
        assert cadence <= epochs
        assert base["tracking"]["wandb"]["workflow"] == "optuna_trial"
        assert base["tracking"]["wandb"]["study"] == config.study["name"]
        assert "study" not in raw["experiment"]["tracking"]["wandb"]

        assert base["data"]["train_dataset"] == raw["experiment"]["data"]["train_dataset"]
        assert base["data"]["ood_datasets"] == raw["experiment"]["data"]["ood_datasets"]
        assert "train_dataset" not in raw_study
        assert "ood_datasets" not in raw_study
        assert "data.train_dataset" not in raw["search_space"]
        assert "data.ood_datasets" not in raw["search_space"]

        if pruner["kind"] == "median":
            assert 0 <= pruner["n_startup_trials"] <= n_trials
            assert cadence <= pruner["n_warmup_steps"] <= epochs
            assert pruner["n_warmup_steps"] % cadence == 0
            assert pruner["interval_steps"] == cadence
        else:
            assert pruner == {"kind": "none"}

        assert config.search_space
        sampled = [parameter for parameter in config.search_space if parameter.kind != "fixed"]
        assert sampled
        assert len({parameter.path for parameter in config.search_space}) == len(config.search_space)
        assert len({parameter.name for parameter in sampled}) == len(sampled)
        for parameter in config.search_space:
            assert parameter.kind in {"categorical", "float", "int", "fixed"}
            if parameter.kind == "categorical":
                assert parameter.values
                assert len(set(parameter.values)) == len(parameter.values)
            elif parameter.kind in {"float", "int"}:
                assert parameter.low is not None
                assert parameter.high is not None
                assert parameter.low <= parameter.high
                if parameter.log:
                    assert parameter.low > 0
                if parameter.step is not None:
                    assert parameter.step > 0
                if parameter.kind == "int":
                    step = 1 if parameter.step is None else int(parameter.step)
                    assert (int(parameter.high) - int(parameter.low)) % step == 0

        if role == "smoke":
            assert n_trials >= _MIN_SMOKE_TRIALS
            assert base["tracking"]["wandb"]["monitor"]["max_cases"] >= 1
            assert base["tracking"]["wandb"]["upload"]["evaluation_artifacts"] is False

    assert names_by_role["production"].isdisjoint(names_by_role["smoke"])
    if trials_by_role["production"] and trials_by_role["smoke"]:
        assert max(trials_by_role["smoke"]) < min(trials_by_role["production"])
        assert max(epochs_by_role["smoke"]) < min(epochs_by_role["production"])


def test_trial_analysis_config_contains_only_named_nonfixed_sampled_parameters() -> None:
    """Expose clean analysis axes while retaining fixed seed and native trial identity."""
    for path in _OPTUNA_CONFIGS:
        study = optuna_runtime.load_optuna_study_config(path)
        trial = _Trial()
        config, context = optuna_runtime._prepare_trial_config(study, trial)
        expected_names = {parameter.name for parameter in study.search_space if parameter.kind != "fixed"}

        assert set(context["analysis_parameters"]) == expected_names
        assert set(context["overrides"]) == {parameter.path for parameter in study.search_space}
        training_seed = study.base_config["run"]["seed"]
        sampler_seed = study.study["seed"]
        assert context["training_seed"] == training_seed
        assert context["sampler_seed"] == sampler_seed
        assert config["tracking"]["wandb"]["tags"] == [
            experiments.config.loader._model_variant(config),
            "optuna",
        ]
        assert "prefix" not in config["run"]
        assert config["run"]["seed"] == training_seed
        assert config["run"]["suffix"] == f"optuna_trial_{trial.number:03d}"
        assert str(study.study["name"]) not in config["run"]["name"]
        assert str(config["task"]) not in config["run"]["name"]
        seed_token = f"s{training_seed}"
        expected_parts = [experiments.config.loader.resolved_model_variant(config)]
        scientific_variant = experiments.config.loader.resolved_scientific_variant(config)
        if scientific_variant is not None:
            expected_parts.append(scientific_variant)
        expected_parts.extend([seed_token, f"optuna_trial_{trial.number:03d}"])
        assert config["run"]["name"].split("__") == expected_parts
        assert config["run"]["name"].split("__").count(seed_token) == 1
        assert trial.attrs["run_seed"] == training_seed
        assert trial.attrs["sampler_seed"] == sampler_seed


def test_pi_trial_name_places_automatic_science_before_seed_and_trial_suffix() -> None:
    """Keep PI trial identity canonical while native trial zero remains final."""
    study = optuna_runtime.load_optuna_study_config(
        configs.optuna_config_path(model_kind="fno", physics_enabled=True),
    )
    config, _context = optuna_runtime._prepare_trial_config(study, _Trial(number=0))

    assert config["run"]["name"] == ("pi-fno_m64x96_h32_l3__spectral_reflect_div_velocity__s9__optuna_trial_000")
    assert config["run"]["suffix"] == "optuna_trial_000"


@pytest.mark.parametrize("trial_number", [0, 1, 9, 10, 99, 100, 999, 1000])
def test_trial_names_use_native_zero_based_numbers_with_three_digit_minimum(trial_number: int) -> None:
    """Preserve Optuna's native number and allow natural expansion beyond three digits."""
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
    config, context = optuna_runtime._prepare_trial_config(study, _Trial(number=trial_number))

    token = f"optuna_trial_{trial_number:03d}"
    assert context["trial_number"] == trial_number
    assert config["run"]["suffix"] == token
    assert config["run"]["name"].endswith(f"__{token}")
    seed_token = f"s{study.base_config['run']['seed']}"
    assert config["run"]["name"].split("__").count(seed_token) == 1
    assert seed_token not in token.split("__")


@pytest.mark.parametrize(
    ("target", "invalid_version"),
    [
        ("study", True),
        ("payload", 1.0),
        ("task", True),
        ("trial_lifecycle", 1.0),
        ("run_summary", True),
    ],
)
def test_existing_study_requires_type_exact_schema_versions(
    target: str,
    invalid_version: object,
) -> None:
    """Reject boolean and floating-point lookalikes in every persisted version field."""
    config = optuna_runtime.load_optuna_study_config(
        _FNO_CONFIG,
    )
    signature = optuna_runtime.build_study_signature(config)
    objective = experiments.config.loader.get_resolved_objective(config.base_config)
    user_attrs: dict[str, Any] = {}
    study = SimpleNamespace(
        direction=SimpleNamespace(name="MINIMIZE"),
        user_attrs=user_attrs,
        set_user_attr=user_attrs.__setitem__,
    )
    optuna_runtime._publish_study_signature(study, signature, objective)
    payload = copy.deepcopy(user_attrs[optuna_runtime._STUDY_SIGNATURE_PAYLOAD_ATTR])
    user_attrs[optuna_runtime._STUDY_SIGNATURE_PAYLOAD_ATTR] = payload
    if target == "study":
        user_attrs[optuna_runtime._STUDY_SIGNATURE_SCHEMA_ATTR] = invalid_version
    elif target == "payload":
        payload["schema_version"] = invalid_version
    elif target == "task":
        payload["task"]["schema_version"] = invalid_version
    elif target == "trial_lifecycle":
        payload["trial_lifecycle"]["schema_version"] = invalid_version
    else:
        payload["trial_lifecycle"]["run_summary_schema_version"] = invalid_version

    with pytest.raises(ValueError, match="schema version"):
        optuna_runtime._validate_existing_study(
            study,
            signature=signature,
            objective=objective,
        )


def test_device_override_is_visible_but_outside_optuna_study_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep runtime device policy visible but outside semantic identity."""

    def summarize(dataset_id: str, *, task: Any, **_paths: Any) -> SimpleNamespace:
        """Return a stable metadata summary for this device-identity test."""
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_path=Path("/datasets") / dataset_id / f"{dataset_id}.pt",
            metadata_directory=Path("/metadata") / dataset_id,
            dataset_exists=True,
            task_id=task.id,
            task_contract_digest=task.contract_digest,
            fingerprint="a" * 64,
            sample_count=2,
        )

    monkeypatch.setattr(datasets.metadata, "load_dataset_metadata_summary", summarize)
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
    summaries = {
        policy: optuna_runtime.describe_optuna_study_config(
            optuna_runtime.with_runtime_overrides(study, device=policy),
        )
        for policy in ("auto", "cuda", "cpu")
    }

    assert {summary["device_policy"] for summary in summaries.values()} == {"auto", "cuda", "cpu"}
    assert len({summary["study"]["name"] for summary in summaries.values()}) == 1
    assert len({summary["storage"] for summary in summaries.values()}) == 1
    assert len({summary["study_dir"] for summary in summaries.values()}) == 1
    assert len({summary["trial_root"] for summary in summaries.values()}) == 1
    assert len({repr(summary["objective"]) for summary in summaries.values()}) == 1
    assert len({repr(summary["search_space"]) for summary in summaries.values()}) == 1
    assert len({summary["semantic_signature"]["digest"] for summary in summaries.values()}) == 1
    assert all("cuda_index" not in summary for summary in summaries.values())


def test_dry_run_dataset_identity_uses_shared_metadata_summary_without_loading_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the shared compact metadata boundary without loading tensor payloads."""
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
    config = copy.deepcopy(study.base_config)
    dataset_root = tmp_path / "raw"
    metadata_root = tmp_path / "meta"
    config["paths"]["dataset_root"] = str(dataset_root)
    config["paths"]["training_meta_root"] = str(metadata_root)
    resolved_task = experiments.config.loader.validate_resolved_task_contract(config)
    captured: list[dict[str, Any]] = []
    state = {"exists": True}

    def summarize(
        dataset_id: str,
        *,
        task: Any,
        dataset_root: Path,
        metadata_root: Path,
    ) -> SimpleNamespace:
        """Record the shared summary request and return compact validated identity."""
        captured.append(
            {
                "dataset_id": dataset_id,
                "task": task,
                "dataset_root": dataset_root,
                "metadata_root": metadata_root,
            }
        )
        return SimpleNamespace(
            dataset_id=dataset_id,
            dataset_path=dataset_root / dataset_id / f"{dataset_id}.pt",
            metadata_directory=metadata_root / dataset_id,
            dataset_exists=state["exists"],
            task_id=task.id,
            task_contract_digest=task.contract_digest,
            fingerprint="a" * 64,
            sample_count=2,
        )

    monkeypatch.setattr(datasets.metadata, "load_dataset_metadata_summary", summarize)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("dry-run must not load tensors"))
    roles = optuna_runtime._configured_dataset_identities(config)

    expected_ids = [config["data"]["train_dataset"], *config["data"]["ood_datasets"]]
    assert [request["dataset_id"] for request in captured] == expected_ids
    assert all(request["task"] == resolved_task for request in captured)
    assert all(request["dataset_root"] == dataset_root for request in captured)
    assert all(request["metadata_root"] == metadata_root for request in captured)
    assert roles["id"]["dataset_id"] == expected_ids[0]
    assert roles["ood"][0]["dataset_id"] == expected_ids[1]
    assert roles["id"]["validation"] == "metadata_package_and_artifact_stat"
    assert roles["id"]["fingerprint"] == "a" * 64

    state["exists"] = False
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        optuna_runtime._configured_dataset_identities(config)


def test_sampler_seed_is_the_explicit_study_seed_and_is_persisted_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the configured study seed to Optuna and persist its sampler metadata."""
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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

    result = optuna_runtime.run_optuna_study(study, n_trials=1, output_root=output_root)

    assert result is fake_study
    sampler_seed = study.study["seed"]
    assert captured["sampler"]["seed"] == sampler_seed
    assert captured["create_study"]["sampler"] is not None
    assert user_attrs[optuna_runtime._SAMPLER_METADATA_ATTR] == {
        "kind": "tpe",
        "multivariate": False,
        "seed": sampler_seed,
    }
    assert "training_seed" not in user_attrs[optuna_runtime._SAMPLER_METADATA_ATTR]
    assert not output_root.exists()


def test_search_space_uses_exact_per_kind_schemas() -> None:
    """
    Feed categorical, fixed, float, and integer parsers extra or missing fields.

    Every malformed schema and unsupported scalar must fail before trial suggestion,
    preventing permissive YAML shapes from silently changing search semantics.
    """
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
    """
    Vary numeric bounds, steps, log flags, exact types, and divisibility constraints.

    Each invalid float or integer family must fail during parsing so Optuna never
    receives coerced, non-finite, unordered, or unreachable distributions.
    """
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
    """
    Mutate each scalar study setting while retaining an otherwise valid raw recipe.

    Blank names/storage, coerced seeds/counts, and unsupported schedule keys must fail
    during load, keeping runtime orchestration free of YAML truthiness shortcuts.
    """
    source_path = _FNO_CONFIG
    load_yaml = experiments.config.loader.load_yaml
    base_raw = load_yaml(source_path)
    invalid_settings: list[tuple[str, Any, type[Exception], str]] = [
        ("name", " ", TypeError, "study.name"),
        ("name", "../escape", ValueError, "single non-empty path component"),
        ("seed", True, TypeError, "study.seed"),
        ("seed", "9", TypeError, "study.seed"),
        ("n_trials", 3.0, TypeError, "study.n_trials"),
        ("report_epochs", [300, 400], ValueError, "unknown key"),
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
    """
    Add objective, direction, and bespoke spike policy to the raw study mapping.

    Every duplicate semantic key must be rejected as unknown because the resolved
    experiment and general pruner alone own objective and pruning behavior.
    """
    source_path = _FNO_CONFIG
    base_raw = experiments.config.loader.load_yaml(source_path)

    for key, value in (
        ("objective", "normalized_macro_rmse"),
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


def test_optuna_tracking_identity_is_projected_only_from_the_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject conflicting workflow or duplicated study identity in an embedded experiment."""
    source_path = _FNO_CONFIG
    base_raw = experiments.config.loader.load_yaml(source_path)
    mutations = (
        (
            lambda raw: raw["experiment"].pop("tracking"),
            TypeError,
            "experiment.tracking must be a mapping",
        ),
        (
            lambda raw: raw["experiment"]["tracking"]["wandb"].update(
                {"study": raw["study"]["name"]},
            ),
            ValueError,
            "wrapper-owned",
        ),
        (
            lambda raw: raw["experiment"]["tracking"]["wandb"].update(
                {"workflow": "train"},
            ),
            ValueError,
            "must be 'optuna_trial'",
        ),
    )

    for mutate, error, match in mutations:
        raw = copy.deepcopy(base_raw)
        mutate(raw)
        monkeypatch.setattr(experiments.config.loader, "load_yaml", lambda _path, raw=raw: raw)
        with pytest.raises(error, match=match):
            optuna_runtime.load_optuna_study_config(source_path)


def test_sampler_and_pruner_semantics_are_validated_during_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Vary sampler/pruner shorthand, kinds, scalar types, and cross-kind settings.

    Every invalid component family must fail at load time, while error context names
    the unsupported kind or inapplicable setting before Optuna construction.
    """
    source_path = _FNO_CONFIG
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
    """
    Load a study with omitted persistence and parse one exact spec of every kind.

    ``None`` storage and categorical, float, integer, and fixed parameters must remain
    supported, proving strict rejection does not narrow the documented valid surface.
    """
    source_path = _FNO_CONFIG
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


def test_sampled_trial_config_preserves_seed_and_revalidates_shared_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture prepared metadata, then reject drift among genuine evaluation events."""
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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

    monkeypatch.setattr(experiments.run, "prepare_fresh_run", capture_prepared_run)
    with pytest.raises(PreparedTrialCaptureError):
        optuna_runtime.run_trial(study, trial)

    config = captured["config"]
    context = captured["context"]
    assert "optuna" not in config
    assert experiments.config.loader.validate_resolved_config(config) == config
    assert experiments.config.loader.get_resolved_objective(config) == _EXPECTED_OBJECTIVE
    assert context["study_name"] == study.study["name"]
    assert context["trial_number"] == trial_number
    training_seed = study.base_config["run"]["seed"]
    sampler_seed = study.study["seed"]
    assert context["training_seed"] == config["run"]["seed"] == training_seed
    assert context["sampler_seed"] == sampler_seed
    assert trial.attrs["run_seed"] == training_seed
    assert "capacity" not in trial.attrs

    output_root = tmp_path / "outputs"
    trial_base = copy.deepcopy(study.base_config)
    trial_base["paths"]["output_root"] = str(output_root)
    trial_base["training"]["evaluation_interval"] = 1
    invalid_cadence = replace(study, base_config=trial_base)
    invalid_trial = _Trial(number=trial_number + 1)
    with pytest.raises(ValueError, match="must share one completed-epoch cadence"):
        optuna_runtime.run_trial(invalid_cadence, invalid_trial)
    assert invalid_trial.attrs == {}
    assert not output_root.exists()


def test_objective_and_derived_search_paths_are_rejected() -> None:
    """
    Try fixed suggestions at objective, task, dataset, run, and channel-owned paths.

    Every path must fail the explicit search allowlist, preventing a sampled trial
    from changing scientific identity or allocation ownership.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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
        with pytest.raises(ValueError, match="not approved"):
            search_space.validate_search_space_paths(study.base_config, (parameter,))


def test_resolved_metric_drift_is_rejected_even_when_semantically_valid() -> None:
    """
    Change the selected metric to another valid kind and compatible reduction.

    Resolved validation must still reject it because objective metadata must exactly
    match the selected metric formula, not merely describe a valid metric.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
    drifted = copy.deepcopy(study.base_config)
    drifted["evaluation"]["metrics"][0]["kind"] = "rmse"
    drifted["evaluation"]["metrics"][0]["reduction"] = "element_mean"

    with pytest.raises(ValueError, match="must exactly equal"):
        experiments.config.loader.validate_resolved_config(drifted)


def test_resolved_metric_unknown_keys_are_rejected() -> None:
    """
    Add one unexpected field to an otherwise valid resolved objective metric.

    Validation must reject the noncanonical mapping so signature and telemetry code
    consume one exact semantic schema without hidden extension fields.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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
    """
    Feed increasing values through minimize and maximize reporters for two epochs.

    Both completed epochs must reach Optuna at their real steps, and the direction-
    specific best epoch/value must remain aligned with checkpoint selection.
    """
    trial = _Trial()
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=trial,
        objective_id="objective",
        direction=direction,
    )

    reporter(1, {"id/objective": values[0]})
    reporter(2, {"id/objective": values[1]})

    assert reporter.best_epoch == best_epoch
    assert reporter.best_value == best_value
    expected_terminal_epoch = len(values)
    assert trial.reports == [(values[0], 1), (values[1], expected_terminal_epoch)]
    assert trial.attrs["last_reported_epoch"] == expected_terminal_epoch


@pytest.mark.parametrize(
    ("target_epoch", "reported_epochs"),
    [(10, (5, 10)), (12, (5, 10, 12))],
)
def test_reporter_uses_interval_or_terminal_completed_epochs(
    target_epoch: int,
    reported_epochs: tuple[int, ...],
) -> None:
    """Report genuine sparse events at native completed-epoch steps, including terminal."""
    trial = _Trial()
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=trial,
        objective_id="objective",
        direction="minimize",
        evaluation_interval=5,
        target_epoch=target_epoch,
    )
    for epoch in reported_epochs:
        reporter(epoch, {"id/objective": float(epoch)})

    assert trial.reports == [(float(epoch), epoch) for epoch in reported_epochs]
    invalid_epoch = reported_epochs[-2] + 1 if len(reported_epochs) > 1 else 1
    with pytest.raises(ValueError, match="interval-or-terminal"):
        reporter(invalid_epoch, {"id/objective": 1.0})


def test_reporter_prunes_non_finite_objectives_explicitly() -> None:
    """
    Send a NaN held-out objective through an otherwise valid epoch reporter.

    A dedicated non-finite error must surface before Optuna can compare or prune the
    value as an ordinary observation, preserving explicit failure classification.
    """
    reporter = optuna_runtime.OptunaEpochReporter(
        trial=_Trial(),
        objective_id="objective",
        direction="minimize",
    )

    with pytest.raises(optuna_runtime.NonFiniteTrialError, match="Non-finite Optuna objective"):
        reporter(1, {"id/objective": float("nan")})


@pytest.mark.parametrize(
    ("failure_stage", "expected_status", "expected_error"),
    [
        ("before_running", "failed", FloatingPointError),
        ("after_running", "nonfinite_pruned", optuna.TrialPruned),
    ],
)
def test_run_trial_classifies_floating_point_failures_by_lifecycle(
    failure_stage: str,
    expected_status: str,
    expected_error: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Inject identical floating-point failures before and after the running transition.

    Setup failure must stay failed while running failure becomes non-finite-pruned;
    every stubbed runtime constructor must receive the resolved CPU device.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
    study = optuna_runtime.with_runtime_overrides(study, device="cpu")
    tracking_disabled_base = copy.deepcopy(study.base_config)
    tracking_disabled_base["tracking"]["wandb"]["mode"] = "disabled"
    study = replace(study, base_config=tracking_disabled_base)
    statuses: list[str] = []
    transitions: list[str] = []
    runtime_devices: list[torch.device] = []
    processor = SimpleNamespace(
        state_dict=dict,
        in_normalizer=None,
        out_normalizer=None,
        to=lambda _device: None,
    )

    def create_dataloaders(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Inject setup failure or return the minimum training-loader bundle."""
        if failure_stage == "before_running":
            message = "non-finite setup value"
            raise FloatingPointError(message)
        return {
            "data_processor": processor,
            "split_indices": {},
            "train": object(),
            "eval": object(),
            "ood": object(),
        }

    def fail_training(**kwargs: Any) -> None:
        """Record the training device, then inject post-running non-finiteness."""
        runtime_devices.append(kwargs["device"])
        message = "non-finite objective"
        raise FloatingPointError(message)

    def configure_reproducibility(
        _config: dict[str, Any],
        *,
        device: torch.device,
    ) -> dict[str, int]:
        """Record reproducibility placement and return one deterministic subseed."""
        runtime_devices.append(device)
        return {"model_init": 1}

    def seed_process(_seed: int, *, device: torch.device) -> None:
        """Record process-seeding placement without mutating global RNG state."""
        runtime_devices.append(device)

    def build_runtime_value(_config: dict[str, Any], *, device: torch.device) -> object:
        """Record model/loss placement and return an inert runtime object."""
        runtime_devices.append(device)
        return object()

    def build_runtime_metrics(_config: dict[str, Any], *, device: torch.device) -> dict[str, object]:
        """Record metric placement and return an empty evaluation mapping."""
        runtime_devices.append(device)
        return {}

    monkeypatch.setattr(
        experiments.run,
        "prepare_fresh_run",
        lambda *_args, **_kwargs: tmp_path / "trial",
    )
    monkeypatch.setattr(
        experiments.run,
        "configure_reproducibility",
        configure_reproducibility,
    )
    monkeypatch.setattr(experiments.run, "seed_process", seed_process)
    monkeypatch.setattr(
        experiments.run,
        "transition_run_status",
        lambda _run_dir, status, **_kwargs: transitions.append(status),
    )
    monkeypatch.setattr(
        experiments.run,
        "runtime_session_updates",
        lambda _run_dir, _resolution, **_kwargs: {},
    )
    monkeypatch.setattr(
        experiments.config.loader,
        "create_dataloaders_from_config",
        create_dataloaders,
    )
    monkeypatch.setattr(optuna_runtime.common.serialization, "atomic_torch_save", lambda *_args: None)
    monkeypatch.setattr(
        optuna_runtime.learning.models.factory,
        "build_model",
        build_runtime_value,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.losses.factory,
        "build_training_loss",
        build_runtime_value,
    )
    monkeypatch.setattr(
        optuna_runtime.learning.metrics.metrics,
        "build_evaluation_metrics",
        build_runtime_metrics,
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
    assert runtime_devices
    assert set(runtime_devices) == {torch.device("cpu")}


def test_strict_cuda_fails_before_optuna_trial_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Request strict CUDA while reporting no usable CUDA device and trap allocation.

    Resolution must raise before trial attributes or output leaves exist, proving a
    strict hardware request never degrades into CPU execution.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
    output_root = tmp_path / "outputs"
    study = optuna_runtime.with_runtime_overrides(
        study,
        device="cuda",
        output_root=output_root,
    )
    monkeypatch.setattr(optuna_runtime.learning.device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        experiments.run,
        "prepare_fresh_run",
        lambda *_args, **_kwargs: pytest.fail("trial allocation must not run"),
    )

    trial = _Trial()
    with pytest.raises(learning.device.DeviceResolutionError, match="strict CUDA"):
        optuna_runtime.run_trial(study, trial)

    assert trial.attrs == {}
    assert not output_root.exists()


def test_uno_optuna_strict_determinism_fails_before_trial_allocation_or_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the shared current-UNO CUDA preflight before any trial run mutation."""
    path = configs.optuna_config_path(model_kind="uno", physics_enabled=True)
    study = optuna_runtime.load_optuna_study_config(path)
    output_root = tmp_path / "outputs"
    study = optuna_runtime.with_runtime_overrides(study, device="cuda", output_root=output_root)
    base = copy.deepcopy(study.base_config)
    base["run"]["deterministic"] = True
    study = replace(study, base_config=base)
    resolution = SimpleNamespace(
        requested_policy="cuda",
        device=torch.device("cuda:0"),
        device_type="cuda",
    )
    monkeypatch.setattr(optuna_runtime.learning.device, "resolve_device", lambda *_args, **_kwargs: resolution)
    monkeypatch.setattr(
        experiments.run,
        "prepare_fresh_run",
        lambda *_args, **_kwargs: pytest.fail("unsupported UNO trial must not allocate"),
    )
    monkeypatch.setattr(
        experiments.tracking,
        "initialize_wandb",
        lambda *_args, **_kwargs: pytest.fail("unsupported UNO trial must not initialize W&B"),
    )

    with pytest.raises(learning.device.DeviceResolutionError, match=r"Set run\.deterministic: false"):
        optuna_runtime.run_trial(study, _Trial())

    assert not output_root.exists()


def test_run_trial_revalidates_study_and_model_before_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Call trial orchestration with invalid study count and unsupported UNO depth.

    Both direct-call drift families must be revalidated before preparation, leaving
    trial attributes empty and preventing output allocation from invalid semantics.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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
    """
    Pass a zero runtime trial count and a programmatically invalid sampler kind.

    Public study orchestration must reject both before creating its output root, so
    invalid invocation or component policy cannot leave partial study state.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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
    """
    Drift the derived direction in an already loaded study configuration.

    Orchestration must detect the objective mismatch before creating output, proving
    programmatic callers cannot bypass the same semantic checks as YAML loading.
    """
    study = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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
    """
    Simulate duplicate creation followed by loading an opposite-direction study.

    Direction validation must stop before optimization or local allocation, protecting
    existing external history from reuse under incompatible selection semantics.
    """
    study_config = optuna_runtime.load_optuna_study_config(_FNO_CONFIG)
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

    def duplicate_study(**_kwargs: Any) -> Any:
        """Force orchestration down the existing-external-study load path."""
        raise optuna.exceptions.DuplicatedStudyError

    monkeypatch.setattr(optuna.pruners, "MedianPruner", lambda **_settings: object())
    monkeypatch.setattr(optuna.samplers, "TPESampler", lambda **_settings: object())
    monkeypatch.setattr(optuna, "create_study", duplicate_study)
    monkeypatch.setattr(optuna, "load_study", lambda **_kwargs: existing)

    with pytest.raises(ValueError, match="Existing Optuna study direction"):
        optuna_runtime.run_optuna_study(study_config, output_root=output_root)
    assert not output_root.exists()
