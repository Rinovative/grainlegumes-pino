# ruff: noqa: S101
"""
Protect strict semantic experiment resolution and resume/scientific identity.

Maintained YAMLs, objective-derived scheduler behavior, device-only changes,
physics composition, unknown keys, and allocation-before-validation failures are
covered. Model implementation construction and Optuna search policy are tested
in the registry and Optuna modules, respectively.
"""

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from src import common, domain, experiments, learning
from support import configs
from torch.optim.sgd import SGD

_EXPERIMENTS = configs.experiment_config_paths()
_ACCEPTANCE_CONFIGS = configs.acceptance_config_paths()
_EXPECTED_OBJECTIVE_REQUEST = {"objective": {"id": "normalized_macro_rmse"}}
_EXPECTED_METRIC_DEFINITIONS = [
    {
        "id": "normalized_macro_rmse",
        "kind": "macro_rmse",
        "space": "normalized",
        "fields": ["p", "u", "v"],
        "reduction": "field_macro_element_mean",
        "direction": "minimize",
    },
    {
        "id": "normalized_rmse_p",
        "kind": "rmse",
        "space": "normalized",
        "fields": ["p"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
    {
        "id": "normalized_rmse_u",
        "kind": "rmse",
        "space": "normalized",
        "fields": ["u"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
    {
        "id": "normalized_rmse_v",
        "kind": "rmse",
        "space": "normalized",
        "fields": ["v"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
    {
        "id": "normalized_rmse",
        "kind": "rmse",
        "space": "normalized",
        "fields": ["p", "u", "v"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
    {
        "id": "normalized_relative_l2",
        "kind": "relative_l2",
        "space": "normalized",
        "fields": ["p", "u", "v"],
        "reduction": "sample_mean",
        "direction": "minimize",
    },
    {
        "id": "normalized_relative_h1",
        "kind": "relative_h1",
        "space": "normalized",
        "fields": ["p", "u", "v"],
        "reduction": "sample_mean",
        "direction": "minimize",
    },
    {
        "id": "physical_rmse_p",
        "kind": "rmse",
        "space": "physical",
        "fields": ["p"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
    {
        "id": "physical_rmse_u",
        "kind": "rmse",
        "space": "physical",
        "fields": ["u"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
    {
        "id": "physical_rmse_v",
        "kind": "rmse",
        "space": "physical",
        "fields": ["v"],
        "reduction": "element_mean",
        "direction": "minimize",
    },
]


def test_every_executable_model_family_uses_its_supported_determinism_policy() -> None:
    """Resolve dynamic experiment, acceptance, and Optuna requests without a frozen inventory."""
    resolved = [
        experiments.config.loader.load_and_resolve_config(path) for path in (*configs.experiment_config_paths(), *configs.acceptance_config_paths())
    ]
    resolved.extend(experiments.tuning.optuna.load_optuna_study_config(path).base_config for path in configs.optuna_config_paths())
    deterministic_policy = {"fno": True, "uno": False}
    assert resolved
    for config in resolved:
        kind = str(config["model"]["kind"])
        seed = config["run"]["seed"]
        assert type(seed) is int
        assert kind in deterministic_policy
        assert config["run"]["deterministic"] is deterministic_policy[kind]


def test_task_spec_owns_the_complete_steady_flow_metric_contract() -> None:
    """Keep all ten full metric definitions in the TaskSpec-owned semantic registry."""
    task = domain.tasks.registry.get_task("steady_flow")
    metrics = [
        {
            "id": metric.id,
            "kind": metric.kind,
            "space": metric.space,
            "fields": list(metric.fields),
            "reduction": metric.reduction,
            "direction": metric.direction,
        }
        for metric in task.default_metrics
    ]

    assert metrics == _EXPECTED_METRIC_DEFINITIONS
    assert not hasattr(task, "default_objective")


def test_every_executable_yaml_uses_canonical_explicit_orchestration() -> None:
    """Discover every family and enforce one raw order, objective selector, and workflow block."""
    acceptance_paths = set(configs.acceptance_config_paths())
    direct_paths = (*configs.experiment_config_paths(), *acceptance_paths)
    for path in direct_paths:
        request = experiments.config.loader.load_yaml(path)
        workflow = "gpu_smoke" if path in acceptance_paths else "train"
        resolved = experiments.config.loader.resolve_config(request)
        objective = experiments.config.loader.get_resolved_objective(resolved)
        wandb = resolved["tracking"]["wandb"]
        cadence = resolved["training"]["evaluation_interval"]

        assert tuple(request) == experiments.config.loader.CANONICAL_EXPERIMENT_SECTION_ORDER
        assert request["evaluation"] == {"objective": {"id": objective["id"]}}
        assert request["tracking"]["wandb"]["workflow"] == workflow
        assert wandb["workflow"] == workflow
        assert wandb["mode"] in {"disabled", "offline", "online"}
        assert type(cadence) is int
        assert cadence >= 1
        assert resolved["training"]["ood_evaluation_interval"] == cadence
        assert wandb["monitor"]["interval"] == cadence
        assert "prefix" not in request["run"]
        assert request["run"].get("suffix") is None or isinstance(request["run"]["suffix"], str)
        assert sum(metric["id"] == objective["id"] for metric in resolved["evaluation"]["metrics"]) == 1

    for path in configs.optuna_config_paths():
        wrapper = experiments.config.loader.load_yaml(path)
        request = wrapper["experiment"]
        study = experiments.tuning.optuna.load_optuna_study_config(path)
        base = study.base_config
        objective = experiments.config.loader.get_resolved_objective(base)
        wandb = base["tracking"]["wandb"]
        cadence = base["training"]["evaluation_interval"]

        assert tuple(wrapper) == ("study", "experiment", "search_space")
        assert not {"objective", "direction"}.intersection(wrapper["study"])
        assert tuple(request) == experiments.config.loader.CANONICAL_EXPERIMENT_SECTION_ORDER
        assert request["evaluation"] == {"objective": {"id": objective["id"]}}
        assert request["tracking"]["wandb"]["workflow"] == "optuna_trial"
        assert "study" not in request["tracking"]["wandb"]
        assert study.base_experiment == request
        assert wandb["workflow"] == "optuna_trial"
        assert wandb["study"] == study.study["name"]
        assert base["training"]["ood_evaluation_interval"] == cadence
        assert wandb["monitor"]["interval"] == cadence
        assert study.study["objective"] == objective["id"]
        assert study.study["direction"] == objective["direction"]
        if study.study["role"] == "smoke":
            assert wandb["monitor"]["max_cases"] >= 1
            assert wandb["upload"]["evaluation_artifacts"] is False


def _raw_fno() -> dict[str, Any]:
    """Return a maintained full-metric FNO request under generic train defaults."""
    raw = experiments.config.loader.load_yaml(configs.acceptance_config_path())
    raw.pop("tracking", None)
    raw["run"]["device"] = "auto"
    raw["run"]["suffix"] = "test_context"
    default_evaluation = copy.deepcopy(
        experiments.config.defaults.get_task_defaults(str(raw["task"]))["evaluation"],
    )
    raw["evaluation"] = {
        "metrics": [{key: value for key, value in metric.items() if key != "direction"} for metric in default_evaluation["metrics"]],
        "objective": copy.deepcopy(_EXPECTED_OBJECTIVE_REQUEST["objective"]),
    }
    return raw


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda path: path.stem)
def test_every_production_experiment_yaml_resolves_current_semantics(path: Path) -> None:
    """Resolve each discovered production recipe through its TaskSpec-owned contract."""
    raw = experiments.config.loader.load_yaml(path)
    config = experiments.config.loader.resolve_config(raw)
    task = experiments.config.loader.validate_resolved_task_contract(config)
    objective = experiments.config.loader.get_resolved_objective(config)

    assert task.id == configs.directory_task(path)
    assert raw["task"] == task.id
    assert config["task"] == task.id
    assert "prefix" not in raw["run"]
    assert raw["run"].get("suffix") is None or isinstance(raw["run"]["suffix"], str)
    assert raw["evaluation"] == {"objective": {"id": objective["id"]}}
    assert config["data"]["train_dataset"] == raw["data"]["train_dataset"]
    assert config["data"]["ood_datasets"] == raw["data"]["ood_datasets"]
    assert config["task_contract"]["digest"] == task.contract_digest
    assert config["run"]["device"] == "auto"
    assert config["loss"]["physics"]["continuity"] in task.physics.allowed_continuities
    assert config["model"]["params"]["in_channels"] == len(task.input_names)
    assert config["model"]["params"]["out_channels"] == len(task.output_names)
    assert objective in config["evaluation"]["metrics"]
    assert objective["id"] == raw["evaluation"]["objective"]["id"]
    metric_spec = next(metric for metric in task.default_metrics if metric.id == objective["id"])
    assert objective["direction"] == metric_spec.direction


@pytest.mark.parametrize(
    "objective",
    [
        pytest.param(None, id="null"),
        pytest.param({}, id="empty"),
        pytest.param({"id": ""}, id="empty-id"),
        pytest.param({"id": 1}, id="non-string-id"),
    ],
)
def test_executable_objective_must_be_present_and_name_one_exact_metric_id(
    objective: object,
) -> None:
    """Reject null, empty, and non-string selectors before task defaults can mask them."""
    raw = _raw_fno()
    raw["evaluation"]["objective"] = objective
    with pytest.raises((TypeError, experiments.config.loader.ConfigError)):
        experiments.config.loader.resolve_config(raw)

    missing = _raw_fno()
    missing["evaluation"].pop("objective")
    with pytest.raises(experiments.config.loader.ConfigError, match="objective is required"):
        experiments.config.loader.resolve_config(missing)


def test_metric_declaration_order_does_not_select_the_objective() -> None:
    """Resolve the same explicit metric ID after reversing the declaration tuple."""
    raw = _raw_fno()
    baseline = experiments.config.loader.resolve_config(copy.deepcopy(raw))
    reversed_request = copy.deepcopy(raw)
    reversed_request["evaluation"]["metrics"].reverse()
    reordered = experiments.config.loader.resolve_config(reversed_request)
    assert reordered["evaluation"]["metrics"][0]["id"] != "normalized_macro_rmse"
    assert reordered["evaluation"]["objective"] == baseline["evaluation"]["objective"]
    assert reordered["evaluation"]["objective"]["id"] == "normalized_macro_rmse"


def test_run_prefix_is_not_a_current_schema_field() -> None:
    """Reject the removed no-op prefix field even when explicitly null."""
    raw = _raw_fno()
    raw["run"]["prefix"] = None
    with pytest.raises(experiments.config.loader.ConfigError, match=r"run contains unknown key"):
        experiments.config.loader.resolve_config(raw)


@pytest.mark.parametrize(
    ("kind", "extension", "continuity", "expected"),
    [
        ("physical", "none", "div_velocity", "physical_div_velocity"),
        ("physical", "none", "div_eps_velocity", "physical_div_eps_velocity"),
        ("spectral", "reflect", "div_velocity", "spectral_reflect_div_velocity"),
        ("spectral", "reflect", "div_eps_velocity", "spectral_reflect_div_eps_velocity"),
    ],
)
def test_pi_scientific_variant_is_derived_from_resolved_physics(
    kind: str,
    extension: str,
    continuity: str,
    expected: str,
) -> None:
    """Derive every supported PI strategy/formulation combination without suffix input."""
    raw = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="fno", physics_enabled=True),
    )
    raw["run"]["suffix"] = None
    raw["loss"]["physics"]["derivatives"] = {"kind": kind, "extension": extension}
    raw["loss"]["physics"]["continuity"] = continuity
    resolved = experiments.config.loader.resolve_config(raw)

    assert experiments.config.loader.resolved_scientific_variant(resolved) == expected
    assert resolved["run"]["name"] == (f"{experiments.config.loader.resolved_model_variant(resolved)}__{expected}__s{resolved['run']['seed']}")


def test_non_pi_and_disabled_physics_have_no_scientific_variant() -> None:
    """Keep supervised names free of physics identity, even if disabled fields are malformed later."""
    raw = _raw_fno()
    raw["run"]["suffix"] = None
    resolved = experiments.config.loader.resolve_config(raw)
    resolved["loss"]["physics"]["derivatives"] = {"kind": "unsupported", "extension": "unsupported"}
    resolved["loss"]["physics"]["continuity"] = "unsupported"

    assert experiments.config.loader.resolved_scientific_variant(resolved) is None
    assert resolved["run"]["name"] == f"fno_m128x160_h64_l3__s{resolved['run']['seed']}"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("derivative", "unsupported", "derivative"),
        ("continuity", "unsupported", "continuity"),
    ],
)
def test_pi_scientific_variant_fails_closed_for_unsupported_identifiers(
    field: str,
    value: str,
    message: str,
) -> None:
    """Reject incomplete PI identity rather than emitting a partial run-name component."""
    raw = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="fno", physics_enabled=True),
    )
    raw["run"]["suffix"] = None
    resolved = experiments.config.loader.resolve_config(raw)
    if field == "derivative":
        resolved["loss"]["physics"]["derivatives"]["kind"] = value
    else:
        resolved["loss"]["physics"]["continuity"] = value

    with pytest.raises(experiments.config.loader.ConfigError, match=message):
        experiments.config.loader.resolved_scientific_variant(resolved)


def test_representative_exact_canonical_run_names() -> None:
    """Freeze stable supervised and PI best-of-class grammar examples."""
    fno_path = next(path for path in _EXPERIMENTS if path.name == "fno_m128x160_h64_l3.yaml")
    piuno_path = next(
        path
        for path in _EXPERIMENTS
        if path.name == "piuno_m64x64_h32_l7_mr0p495_physical_div_eps_velocity.yaml" and path.parent.name == "best_of_class"
    )

    fno = experiments.config.loader.load_and_resolve_config(fno_path)
    piuno = experiments.config.loader.load_and_resolve_config(piuno_path)
    assert fno["run"]["name"] == "fno_m128x160_h64_l3__s9__best_of_class"
    assert piuno["run"]["name"] == ("pi-uno_m64x64_h32_l7_r0p49513182__physical_div_eps_velocity__s9__best_of_class")


@pytest.mark.parametrize(
    "suffix",
    [
        "best_of_class_physical_div_eps_velocity",
        "pi-uno_best_of_class",
        "steady_flow_best_of_class",
        "best_of_class_s9",
    ],
)
def test_suffix_rejects_token_bounded_derived_identity(suffix: str) -> None:
    """Reject scientific, model, task, and seed duplication with an actionable correction."""
    raw = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="uno", physics_enabled=True),
    )
    raw["run"]["suffix"] = suffix
    with pytest.raises(experiments.config.loader.ConfigError, match=r"run\.suffix .*duplicates canonical derived identity.*Suggested"):
        experiments.config.loader.resolve_config(raw)


@pytest.mark.parametrize(
    "suffix",
    [
        "best_of_class",
        "low_capacity",
        "lr_0p01",
        "low_capacity_lr_0p000625",
        "gpu_smoke",
        "optuna_trial_000",
        "steady_flowing_profile",
    ],
)
def test_suffix_accepts_normalized_non_derived_experiment_context(suffix: str) -> None:
    """Accept genuine roles and token-boundary lookalikes without substring false positives."""
    raw = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="fno", physics_enabled=True),
    )
    raw["run"]["suffix"] = suffix
    resolved = experiments.config.loader.resolve_config(raw)
    assert resolved["run"]["name"].endswith(f"__{suffix}")


@pytest.mark.parametrize("suffix", ["GPU_smoke", "gpu__smoke", "_gpu_smoke", "gpu_smoke_", "gpu-smoke"])
def test_suffix_rejects_noncanonical_grammar(suffix: str) -> None:
    """Require one lowercase underscore-separated suffix component."""
    raw = _raw_fno()
    raw["run"]["suffix"] = suffix
    with pytest.raises(experiments.config.loader.ConfigError, match="lowercase underscore-separated"):
        experiments.config.loader.resolve_config(raw)


def test_explicit_dataset_selection_precedes_fallback_without_identity_drift() -> None:
    """Keep recipe selection authoritative while identical fallback spelling stays semantic-noop."""
    explicit_raw = _raw_fno()
    explicit = experiments.config.loader.resolve_config(copy.deepcopy(explicit_raw))

    fallback_raw = copy.deepcopy(explicit_raw)
    fallback_raw["data"].pop("train_dataset")
    fallback_raw["data"].pop("ood_datasets")
    fallback = experiments.config.loader.resolve_config(fallback_raw)

    assert fallback["data"] == explicit["data"]
    assert explicit["run"]["name"] == fallback["run"]["name"]
    assert common.serialization.canonical_json_sha256(explicit) == common.serialization.canonical_json_sha256(fallback)
    assert learning.training.checkpoint.config_digest(explicit) == learning.training.checkpoint.config_digest(fallback)
    assert learning.training.checkpoint.resume_contract_digest(explicit) == learning.training.checkpoint.resume_contract_digest(fallback)
    assert experiments.run.validate_resume_config(explicit, fallback) == int(explicit["training"]["epochs"])

    override_raw = copy.deepcopy(explicit_raw)
    override_raw["data"] = {
        "train_dataset": "recipe_specific_id",
        "ood_datasets": ["recipe_specific_ood"],
    }
    overridden = experiments.config.loader.resolve_config(override_raw)
    assert overridden["data"]["train_dataset"] == "recipe_specific_id"
    assert overridden["data"]["ood_datasets"] == ["recipe_specific_ood"]


@pytest.mark.parametrize("path", _ACCEPTANCE_CONFIGS, ids=lambda path: path.stem)
def test_standalone_acceptance_yaml_explicitly_selects_one_dataset_pair(path: Path) -> None:
    """Keep standalone acceptance requests independent of omitted-data fallback selection."""
    raw = experiments.config.loader.load_yaml(path)
    train_dataset = raw["data"]["train_dataset"]
    ood_datasets = raw["data"]["ood_datasets"]
    assert isinstance(train_dataset, str)
    assert train_dataset
    assert isinstance(ood_datasets, list)
    assert ood_datasets
    assert train_dataset not in ood_datasets
    resolved = experiments.config.loader.resolve_config(raw)
    assert resolved["data"]["train_dataset"] == train_dataset
    assert resolved["data"]["ood_datasets"] == ood_datasets


def test_resolved_task_contract_requires_complete_current_schema() -> None:
    """Reject lookalike and unsupported task-contract schema versions in one focused check."""
    for schema_version in (True, 1.0, 2):
        config = experiments.config.loader.resolve_config(_raw_fno())
        contract = config["task_contract"]
        assert isinstance(contract, dict)
        contract["schema_version"] = schema_version
        with pytest.raises(
            experiments.config.loader.ConfigError,
            match="does not exactly match registered task",
        ):
            experiments.config.loader.validate_resolved_task_contract(config)


def test_device_only_changes_preserve_scientific_and_resume_identity() -> None:
    """Keep all supported device-policy transitions outside scientific identity."""
    raw = _raw_fno()
    for saved_policy, requested_policy in (
        ("auto", "cpu"),
        ("auto", "cuda"),
        ("cuda", "cpu"),
        ("cpu", "cuda"),
    ):
        saved_raw = copy.deepcopy(raw)
        requested_raw = copy.deepcopy(raw)
        saved_raw["run"]["device"] = saved_policy
        requested_raw["run"]["device"] = requested_policy
        saved = experiments.config.loader.resolve_config(saved_raw)
        requested = experiments.config.loader.resolve_config(requested_raw)

        assert learning.training.checkpoint.config_digest(requested) == learning.training.checkpoint.config_digest(saved)
        assert learning.training.checkpoint.resume_contract_digest(requested) == learning.training.checkpoint.resume_contract_digest(saved)
        assert experiments.run.validate_resume_config(requested, saved) == int(saved["training"]["epochs"])
        assert requested["evaluation"]["objective"] == saved["evaluation"]["objective"]


def test_scheduler_mode_derives_from_the_resolved_objective_direction() -> None:
    """
    Build plateau schedulers from otherwise equal minimize and maximize objectives.

    Scheduler mode must follow the resolved objective direction exactly, preventing
    a second independent selection direction from drifting out of sync.
    """
    config = experiments.config.loader.resolve_config(_raw_fno())
    parameter = torch.nn.Parameter(torch.zeros(()))
    minimizing = learning.training.optim.build_scheduler(
        SGD([parameter], lr=0.1),
        config,
    )
    assert minimizing is not None
    assert minimizing.mode == "min"

    maximizing_config = copy.deepcopy(config)
    maximizing_config["evaluation"]["objective"]["direction"] = "maximize"
    maximizing = learning.training.optim.build_scheduler(
        SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.1),
        maximizing_config,
    )
    assert maximizing is not None
    assert maximizing.mode == "max"


def test_physics_informed_model_families_use_loss_composition() -> None:
    """Keep PI-FNO and PI-UNO as base model families with active semantic loss composition."""
    for expected_kind in ("fno", "uno"):
        path = configs.experiment_config_path(model_kind=expected_kind, physics_enabled=True)
        raw = experiments.config.loader.load_yaml(path)
        config = experiments.config.loader.resolve_config(raw)
        assert config["model"]["kind"] == expected_kind
        assert config["loss"]["physics"]["enabled"] is True
        assert config["loss"]["physics"]["continuity"] in config["task_contract"]["physics"]["allowed_continuities"]


def test_supervised_model_families_remain_physics_disabled_with_task_default() -> None:
    """Resolve supervised FNO and UNO recipes with inactive but complete physics semantics."""
    for model_kind in ("fno", "uno"):
        path = configs.experiment_config_path(model_kind=model_kind, physics_enabled=False)
        config = experiments.config.loader.load_and_resolve_config(path)
        assert config["loss"]["physics"]["enabled"] is False
        assert config["loss"]["physics"]["continuity"] == "div_eps_velocity"


def test_objective_change_is_resume_incompatible_without_task_schema_change() -> None:
    """
    Change only the selected objective from macro RMSE to normalized relative H1.

    Task schema must remain identical, but effective/resume identities and resume
    validation must change because model selection semantics are continuation-critical.
    """
    raw = _raw_fno()
    macro_config = experiments.config.loader.resolve_config(copy.deepcopy(raw))
    h1_raw = copy.deepcopy(raw)
    h1_raw["evaluation"]["objective"] = {"id": "normalized_relative_h1"}
    h1_config = experiments.config.loader.resolve_config(h1_raw)

    assert macro_config["task_contract"]["schema_version"] == 1
    assert h1_config["task_contract"]["schema_version"] == 1
    assert learning.training.checkpoint.config_digest(macro_config) != learning.training.checkpoint.config_digest(h1_config)
    assert learning.training.checkpoint.resume_contract_digest(macro_config) != learning.training.checkpoint.resume_contract_digest(h1_config)
    with pytest.raises(ValueError, match=r"evaluation\.objective\.(id|kind|reduction)"):
        experiments.run.validate_resume_config(macro_config, h1_config)


def test_continuity_selection_changes_effective_digest_and_resume_contract() -> None:
    """
    Resolve both task-allowed continuity formulations from one PI recipe.

    The task contract remains fixed, but effective/resume digests must differ and
    cross-formulation resume must fail because optimized physics changed.
    """
    raw = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="fno", physics_enabled=True),
    )
    resolved_by_continuity = {}
    for continuity in ("div_velocity", "div_eps_velocity"):
        selected = copy.deepcopy(raw)
        selected["loss"]["physics"]["continuity"] = continuity
        resolved_by_continuity[continuity] = experiments.config.loader.resolve_config(selected)
        assert resolved_by_continuity[continuity]["loss"]["physics"]["continuity"] == continuity

    plain = resolved_by_continuity["div_velocity"]
    conservative = resolved_by_continuity["div_eps_velocity"]
    assert plain["task_contract"] == conservative["task_contract"]
    assert learning.training.checkpoint.config_digest(plain) != learning.training.checkpoint.config_digest(conservative)
    assert learning.training.checkpoint.resume_contract_digest(plain) != learning.training.checkpoint.resume_contract_digest(conservative)
    with pytest.raises(ValueError, match=r"loss\.physics\.continuity"):
        experiments.run.validate_resume_config(plain, conservative)


def test_unknown_continuity_identifier_fails_with_exact_config_path() -> None:
    """
    Pass a neutral unsupported identifier at one PI config path.

    It must fail with the exact ``loss.physics.continuity`` path, proving resolution
    accepts only task-declared semantic identifiers.
    """
    raw = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="fno", physics_enabled=True),
    )
    unknown = "unsupported"
    raw["loss"]["physics"]["continuity"] = unknown

    with pytest.raises(
        experiments.config.loader.ConfigError,
        match=rf"Unknown continuity identifier '{unknown}' at loss\.physics\.continuity",
    ):
        experiments.config.loader.resolve_config(raw)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda cfg: cfg.update({"task": "unknown_task"}), "Unknown task"),
        (lambda cfg: cfg["model"].update({"kind": "unsupported"}), "Unknown model identifier"),
        (lambda cfg: cfg["loss"]["data"].update({"kind": "unsupported"}), "not allowed by task"),
        (lambda cfg: cfg["evaluation"]["metrics"][0].update({"kind": "unsupported"}), "Unknown metric identifier"),
        (lambda cfg: cfg["evaluation"]["metrics"][0].update({"field": "p"}), "unknown key"),
        (lambda cfg: cfg["evaluation"]["metrics"][0].update({"fields": ["temperature"]}), "unknown task output"),
        (
            lambda cfg: cfg["evaluation"]["metrics"][0].update({"fields": ["p", "u"]}),
            "must select every TaskSpec output field",
        ),
        (
            lambda cfg: cfg["evaluation"]["metrics"][0].update({"reduction": "element_mean"}),
            "does not support reduction",
        ),
        (lambda cfg: cfg["evaluation"]["objective"].update({"id": "unknown_metric"}), "is not a declared metric id"),
        (lambda cfg: cfg["evaluation"]["objective"].update({"direction": "maximize"}), "unknown key"),
        (lambda cfg: cfg["model"]["params"].update({"in_channels": 8}), "task-fixed channel"),
        (lambda cfg: cfg.setdefault("data", {}).update({"ood_datasets": ["first", "second"]}), "exactly one"),
        (lambda cfg: cfg.setdefault("data", {}).update({"train_dataset": "../escape"}), "single non-empty path component"),
    ],
)
def test_invalid_semantic_identifiers_and_overrides_fail(mutation: object, match: str) -> None:
    """
    Mutate one semantic identifier, task-owned field, derived channel, or logical path.

    Every parametrized family must fail with contextual resolution evidence while
    the surrounding FNO recipe stays valid, isolating strict config ownership.
    """
    config = copy.deepcopy(_raw_fno())
    mutation(config)  # type: ignore[operator]
    with pytest.raises(ValueError, match=match):
        experiments.config.loader.resolve_config(config)


def test_unsupported_uno_depth_fails_before_fresh_run_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Set a PI-UNO recipe to an unsupported three-layer topology and trap allocation.

    Public resolution must fail before any run directory is attempted or created,
    protecting fresh lifecycle state from structurally invalid models.
    """
    config = experiments.config.loader.load_yaml(
        configs.experiment_config_path(model_kind="uno", physics_enabled=True),
    )
    model = config["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, dict)
    params["n_layers"] = 3
    allocation_attempted = False

    def fail_allocation(_run_dir: Path | str) -> Path:
        """Record and fail if invalid config crosses the allocation boundary."""
        nonlocal allocation_attempted
        allocation_attempted = True
        pytest.fail("run allocation must not be attempted")

    monkeypatch.setattr(experiments.run.config_loader, "load_yaml", lambda _path: config)
    monkeypatch.setattr(experiments.run, "allocate_run_directory", fail_allocation)
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        experiments.run.run_experiment("invalid-uno.yaml", output_root=output_root)

    assert allocation_attempted is False
    assert not output_root.exists()


def test_duplicate_metric_id_fails() -> None:
    """
    Duplicate one metric ID while leaving two otherwise valid definitions intact.

    Resolution must reject the collision so the objective and telemetry mappings
    cannot refer to an ambiguous metric declaration.
    """
    config = copy.deepcopy(_raw_fno())
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    metrics = evaluation["metrics"]
    assert isinstance(metrics, list)
    first, second = metrics[:2]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["id"] = first["id"]
    with pytest.raises(ValueError, match="Duplicate evaluation metric id"):
        experiments.config.loader.resolve_config(config)
