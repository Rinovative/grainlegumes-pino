# ruff: noqa: S101
"""Protect configuration semantics with complete test-owned requests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from src import domain, experiments, learning
from support import configs

if TYPE_CHECKING:
    from collections.abc import Callable

_CONFIG_ROOT = Path(__file__).parents[2] / "configs"
_SYNTHETIC_BATCH_SIZE = 3


@pytest.mark.parametrize("model_kind", ["fno", "uno"])
def test_minimal_direct_config_resolves_and_preserves_user_values(
    model_kind: str,
) -> None:
    """Resolve representative model families without a production YAML fixture."""
    raw = configs.direct_config(model_kind=model_kind)
    raw["data"]["batch_size"] = _SYNTHETIC_BATCH_SIZE
    raw["optimizer"]["lr"] = 4.321e-3

    resolved = experiments.config.loader.resolve_config(raw)
    task = domain.tasks.registry.get_task("steady_flow")

    assert resolved["model"]["kind"] == model_kind
    assert resolved["model"]["params"]["in_channels"] == task.in_channels
    assert resolved["model"]["params"]["out_channels"] == task.out_channels
    assert resolved["data"]["train_dataset"] == "synthetic_train"
    assert resolved["data"]["ood_datasets"] == ["synthetic_ood"]
    assert resolved["data"]["batch_size"] == _SYNTHETIC_BATCH_SIZE
    assert resolved["optimizer"]["lr"] == pytest.approx(4.321e-3)
    assert resolved["evaluation"]["objective"]["id"] == "normalized_group_macro_rmse"


def test_minimal_physics_config_preserves_explicit_semantics() -> None:
    """Propagate explicitly supplied formulation and weight values unchanged."""
    raw = configs.direct_config(physics_enabled=True)
    physics = raw["loss"]["physics"]
    physics["derivatives"] = {"kind": "spectral", "extension": "reflect"}
    physics["continuity"] = "div_eps_velocity"
    physics["residual_weight"]["target"] = 4.0e-3
    physics["boundary_weight"]["target"] = 6.0e-3

    resolved = experiments.config.loader.resolve_config(raw)

    assert resolved["loss"]["physics"]["derivatives"] == {
        "kind": "spectral",
        "extension": "reflect",
    }
    assert resolved["loss"]["physics"]["continuity"] == "div_eps_velocity"
    assert resolved["loss"]["physics"]["residual_weight"]["target"] == pytest.approx(4.0e-3)
    assert resolved["loss"]["physics"]["boundary_weight"]["target"] == pytest.approx(6.0e-3)


def test_canonical_name_has_one_explicit_synthetic_grammar_example() -> None:
    """Keep intentional name grammar stable for fully specified artificial input."""
    raw = configs.direct_config(physics_enabled=True)
    physics = raw["loss"]["physics"]
    physics["derivatives"] = {"kind": "spectral", "extension": "reflect"}
    physics["continuity"] = "div_eps_velocity"
    physics["residual_weight"]["target"] = 4.0e-3
    physics["boundary_weight"]["target"] = 6.0e-3

    resolved = experiments.config.loader.resolve_config(raw)

    assert resolved["run"]["name"] == ("pi-fno_m4x5_h8_l2__spec_div_eps_vel_lamphys4e-03_lamp6e-03__synthetic_train__s17__fixture")
    assert experiments.config.loader.resolved_scientific_variant(resolved) == ("spec_div_eps_vel_lamphys4e-03_lamp6e-03")


def test_scientific_change_updates_name_and_resume_identity() -> None:
    """Treat an explicit physics-weight change as continuation-incompatible."""
    first_raw = configs.direct_config(physics_enabled=True)
    second_raw = copy.deepcopy(first_raw)
    second_raw["loss"]["physics"]["residual_weight"]["target"] = 7.0e-3
    first = experiments.config.loader.resolve_config(first_raw)
    second = experiments.config.loader.resolve_config(second_raw)

    assert first["run"]["name"] != second["run"]["name"]
    assert learning.training.checkpoint.config_digest(first) != (learning.training.checkpoint.config_digest(second))
    assert learning.training.checkpoint.resume_contract_digest(first) != (learning.training.checkpoint.resume_contract_digest(second))
    with pytest.raises(ValueError, match=r"loss\.physics\.residual_weight\.target"):
        experiments.run.validate_resume_config(second, first)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("device", "cuda"),
        ("output_root", "/mnt/relocated-synthetic-output"),
    ],
)
def test_runtime_location_changes_preserve_scientific_and_resume_identity(
    field: str,
    replacement: str,
) -> None:
    """Exclude device and output location from run naming and resume semantics."""
    first = experiments.config.loader.resolve_config(configs.direct_config())
    second = copy.deepcopy(first)
    if field == "device":
        second["run"]["device"] = replacement
    else:
        second["paths"]["output_root"] = replacement

    assert second["run"]["name"] == first["run"]["name"]
    assert learning.training.checkpoint.resume_contract_digest(second) == (learning.training.checkpoint.resume_contract_digest(first))
    assert experiments.run.validate_resume_config(second, first) == int(first["training"]["epochs"])


def test_objective_change_is_resume_incompatible() -> None:
    """Bind model selection semantics to persisted continuation identity."""
    first = experiments.config.loader.resolve_config(configs.direct_config())
    changed_raw = configs.direct_config()
    changed_raw["evaluation"]["objective"] = {"id": "normalized_relative_h1"}
    second = experiments.config.loader.resolve_config(changed_raw)

    assert learning.training.checkpoint.resume_contract_digest(first) != (learning.training.checkpoint.resume_contract_digest(second))
    with pytest.raises(ValueError, match=r"evaluation\.objective"):
        experiments.run.validate_resume_config(second, first)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda config: config.pop("task"), "task"),
        (
            lambda config: config["model"].update({"kind": "unsupported"}),
            "Unknown model identifier",
        ),
        (
            lambda config: config["data"].update({"batch_size": 0}),
            "positive",
        ),
        (
            lambda config: config["run"].update({"seed": True}),
            "integer",
        ),
        (
            lambda config: config["data"].update({"ood_datasets": ["../escape"]}),
            "single non-empty path component",
        ),
    ],
)
def test_invalid_config_fails_at_the_public_resolver(
    mutation: Callable[[dict[str, Any]], object],
    match: str,
) -> None:
    """Reject one representative invalid value from each important branch."""
    raw = configs.direct_config()
    mutation(raw)

    with pytest.raises((TypeError, ValueError), match=match):
        experiments.config.loader.resolve_config(raw)


def test_unknown_continuity_fails_with_the_semantic_path() -> None:
    """Reject a physics formulation not declared by the selected task."""
    raw = configs.direct_config(physics_enabled=True)
    raw["loss"]["physics"]["continuity"] = "unsupported"

    with pytest.raises(
        experiments.config.loader.ConfigError,
        match=r"loss\.physics\.continuity",
    ):
        experiments.config.loader.resolve_config(raw)


def test_duplicate_metric_identifier_is_rejected() -> None:
    """Prevent an objective from resolving against an ambiguous declaration."""
    resolved = experiments.config.loader.resolve_config(configs.direct_config())
    raw = configs.direct_config()
    metrics = copy.deepcopy(resolved["evaluation"]["metrics"])
    for metric in metrics:
        metric.pop("direction", None)
    metrics[1]["id"] = metrics[0]["id"]
    raw["evaluation"]["metrics"] = metrics

    with pytest.raises(ValueError, match="Duplicate evaluation metric id"):
        experiments.config.loader.resolve_config(raw)


def test_resolved_task_contract_rejects_an_unsupported_schema() -> None:
    """Fail closed when persisted task semantics use another schema version."""
    resolved = experiments.config.loader.resolve_config(configs.direct_config())
    resolved["task_contract"]["schema_version"] += 1

    with pytest.raises(
        experiments.config.loader.ConfigError,
        match="does not exactly match registered task",
    ):
        experiments.config.loader.validate_resolved_task_contract(resolved)


def test_current_yaml_files_are_dynamically_parseable() -> None:
    """Admit every current YAML without freezing names, locations, counts, or values."""
    paths = tuple(sorted(_CONFIG_ROOT.rglob("*.yaml")))
    assert paths
    for path in paths:
        raw = experiments.config.loader.load_yaml(path)
        if "study" in raw and "experiment" in raw:
            experiments.tuning.optuna.load_optuna_study_config(path)
        else:
            experiments.config.loader.resolve_config(raw)
