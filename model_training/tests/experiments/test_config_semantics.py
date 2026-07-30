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

import pytest
import torch
from src import domain, experiments, learning
from torch.optim.sgd import SGD

_CONFIG_ROOT = Path(__file__).parents[2] / "configs"
_EXPERIMENTS = sorted((_CONFIG_ROOT / "experiments").glob("*.yaml"))


def _raw_fno() -> dict[str, object]:
    """Load a fresh mutable copy of the maintained supervised FNO YAML."""
    return experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_fno.yaml")


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda path: path.stem)
def test_every_experiment_yaml_resolves(path: Path) -> None:
    """
    Resolve all four maintained experiment YAMLs through the public loader.

    Each must produce the same exact TaskSpec/objective semantics while retaining
    its declared architecture and physics composition, preventing recipe drift.
    """
    config = experiments.config.loader.load_and_resolve_config(path)
    task = domain.tasks.registry.get_task("steady_flow")
    assert config["task_contract"]["digest"] == task.contract_digest
    assert config["run"]["device"] == "auto"
    assert config["loss"]["physics"]["continuity"] == task.physics.continuity
    assert config["model"]["params"]["in_channels"] == len(task.input_names)
    assert config["model"]["params"]["out_channels"] == len(task.output_names)
    expected_objective = {
        "id": "normalized_macro_rmse",
        "kind": "macro_rmse",
        "space": "normalized",
        "fields": ["p", "u", "v"],
        "reduction": "field_macro_element_mean",
        "direction": "minimize",
    }
    assert config["loss"]["data"] == {
        "kind": "relative_h1",
        "space": "normalized",
        "weight": 1.0,
    }
    assert [metric["id"] for metric in config["evaluation"]["metrics"]] == [
        "normalized_macro_rmse",
        "normalized_rmse_p",
        "normalized_rmse_u",
        "normalized_rmse_v",
        "normalized_rmse",
        "normalized_relative_l2",
        "normalized_relative_h1",
        "physical_rmse_p",
        "physical_rmse_u",
        "physical_rmse_v",
    ]
    assert config["evaluation"]["objective"] == expected_objective
    selected = next(metric for metric in config["evaluation"]["metrics"] if metric["id"] == expected_objective["id"])
    assert selected == expected_objective


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_resolved_task_contract_requires_complete_current_schema(schema_version: object) -> None:
    """Reject a saved task contract that differs from the registered contract."""
    config = experiments.config.loader.load_and_resolve_config(_CONFIG_ROOT / "experiments/steady_flow_fno.yaml")
    contract = config["task_contract"]
    assert isinstance(contract, dict)
    contract["schema_version"] = schema_version

    with pytest.raises(
        experiments.config.loader.ConfigError,
        match="does not exactly match registered task",
    ):
        experiments.config.loader.validate_resolved_task_contract(config)


@pytest.mark.parametrize(
    ("saved_policy", "requested_policy"),
    [("auto", "cpu"), ("auto", "cuda"), ("cuda", "cpu"), ("cpu", "cuda")],
)
def test_device_only_changes_preserve_scientific_and_resume_identity(
    saved_policy: str,
    requested_policy: str,
) -> None:
    """
    Cross saved and requested ``auto``, ``cpu``, and ``cuda`` device policies.

    Every transition must retain effective and resume digests plus objective
    semantics, proving device location is operational rather than scientific identity.
    """
    raw = _raw_fno()
    saved_raw = copy.deepcopy(raw)
    requested_raw = copy.deepcopy(raw)
    saved_run = saved_raw["run"]
    requested_run = requested_raw["run"]
    assert isinstance(saved_run, dict)
    assert isinstance(requested_run, dict)
    saved_run["device"] = saved_policy
    requested_run["device"] = requested_policy
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
    config = experiments.config.loader.load_and_resolve_config(_CONFIG_ROOT / "experiments/steady_flow_fno.yaml")
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


def test_descriptive_pi_yaml_names_use_loss_composition() -> None:
    """
    Resolve descriptive PI-FNO and PI-UNO recipe filenames through the public loader.

    Each must retain its base ``fno``/``uno`` model ID and enable physics through
    loss composition, keeping filenames outside the semantic model registry.
    """
    for filename, expected_kind in (
        ("steady_flow_pifno.yaml", "fno"),
        ("steady_flow_piuno.yaml", "uno"),
    ):
        path = _CONFIG_ROOT / "experiments" / filename
        raw = experiments.config.loader.load_yaml(path)
        assert raw["loss"]["physics"]["continuity"] == "div_eps_velocity"
        config = experiments.config.loader.resolve_config(raw)
        assert config["model"]["kind"] == expected_kind
        assert config["loss"]["physics"]["enabled"] is True
        assert config["loss"]["physics"]["continuity"] == "div_eps_velocity"


@pytest.mark.parametrize("filename", ["steady_flow_fno.yaml", "steady_flow_uno.yaml"])
def test_supervised_yaml_remains_physics_disabled_with_task_default(filename: str) -> None:
    """
    Resolve both supervised model-family YAMLs with their shared task contract.

    Physics must remain disabled while continuity still resolves to the task default,
    keeping semantic metadata complete without activating residual loss.
    """
    config = experiments.config.loader.load_and_resolve_config(_CONFIG_ROOT / "experiments" / filename)

    assert config["loss"]["physics"]["enabled"] is False
    assert config["loss"]["physics"]["continuity"] == "div_eps_velocity"


def test_objective_change_is_resume_incompatible_without_task_schema_change() -> None:
    """
    Change only the selected objective from macro RMSE to normalized relative H1.

    Task schema must remain identical, but effective/resume identities and resume
    validation must change because model selection semantics are continuation-critical.
    """
    raw = experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_fno.yaml")
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
    raw = experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_pifno.yaml")
    configs = {}
    for continuity in ("div_velocity", "div_eps_velocity"):
        selected = copy.deepcopy(raw)
        selected["loss"]["physics"]["continuity"] = continuity
        configs[continuity] = experiments.config.loader.resolve_config(selected)
        assert configs[continuity]["loss"]["physics"]["continuity"] == continuity

    plain = configs["div_velocity"]
    conservative = configs["div_eps_velocity"]
    assert plain["task_contract"] == conservative["task_contract"]
    assert learning.training.checkpoint.config_digest(plain) != learning.training.checkpoint.config_digest(conservative)
    assert learning.training.checkpoint.resume_contract_digest(plain) != learning.training.checkpoint.resume_contract_digest(conservative)
    with pytest.raises(ValueError, match=r"loss\.physics\.continuity"):
        experiments.run.validate_resume_config(plain, conservative)


@pytest.mark.parametrize("unknown", ["div_u", "div_eps_u", "automatic"])
def test_unknown_continuity_identifier_fails_with_exact_config_path(unknown: str) -> None:
    """
    Vary unsupported residual spellings and an automatic sentinel at one PI config path.

    Every spelling must fail with the exact ``loss.physics.continuity`` path, proving
    resolution accepts only task-declared semantic identifiers.
    """
    raw = experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_pifno.yaml")
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
        (lambda cfg: cfg["model"].update({"kind": "PI-FNO"}), "Unknown model identifier"),
        (lambda cfg: cfg["loss"]["data"].update({"kind": "H1Loss"}), "not allowed by task"),
        (lambda cfg: cfg["evaluation"]["metrics"][0].update({"kind": "RMSEOverall"}), "Unknown metric identifier"),
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
        (lambda cfg: cfg["run"].update({"prefix": "../escape"}), "single non-empty path component"),
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
    config = experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_piuno.yaml")
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
