# ruff: noqa: S101, PLR2004
"""
Protect the immutable registered task, ordered fields, units, and derived semantics.

The tests establish the exact steady-flow contract, stable serialization/digest,
registry immutability, ordered declaration rejection, and public domain exports.
Dataset content identity and experiment-default projection are covered by the
data and config suites rather than duplicated here.
"""

from dataclasses import FrozenInstanceError, replace

import pytest
from src import domain


def test_steady_flow_contract_is_exact_and_task_owned() -> None:
    """
    Resolve the sole registered task and assert its complete steady-flow contract.

    Fields, units, layout, datasets, losses, metrics, physics, objective, and digest
    must remain task-owned so downstream components share one semantic authority.
    """
    task = domain.tasks.registry.get_task("steady_flow")

    assert domain.tasks.registry.available_tasks() == ("steady_flow",)
    assert task.input_names == ("x", "y", "kxx", "kxy", "kyy", "eps", "p_bc")
    assert task.output_names == ("p", "u", "v")
    assert task.in_channels == 7
    assert task.out_channels == 3
    assert task.tensor_layout == ("batch", "channel", "y", "x")
    assert task.operator_axes == (2, 3)
    assert task.normalization_axes == (0, 2, 3)
    assert task.operator_dimensionality == 2
    assert task.default_datasets.train == "lhs_var80_seed3001"
    assert task.default_datasets.ood == ("lhs_var120_seed4001",)
    assert task.preprocessing.fit_split == "train"
    assert task.data_losses == ("relative_h1", "relative_l2")
    assert task.schema_version == 1
    assert task.physics.kind == "steady_2d_brinkman"
    assert task.physics.continuity == "div_eps_velocity"
    assert task.physics.allowed_continuities == ("div_velocity", "div_eps_velocity")
    assert [metric.id for metric in task.default_metrics] == [
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
    assert task.default_objective.kind == "macro_rmse"
    assert task.default_objective.space == "normalized"
    assert task.default_objective.fields == task.output_names
    assert task.default_objective.reduction == "field_macro_element_mean"
    assert task.default_objective.direction == "minimize"
    assert {field.name: field.unit for field in (*task.inputs, *task.outputs)} == {
        "x": "m",
        "y": "m",
        "kxx": "m^2",
        "kxy": "m^2",
        "kyy": "m^2",
        "eps": "1",
        "p_bc": "Pa",
        "p": "Pa",
        "u": "m/s",
        "v": "m/s",
    }
    assert task.field("kxx").representation == "dimensionless_log10_ratio_to_1_m2"
    assert task.field("kxy").representation == "dimensionless_cross_component_ratio_to_geometric_mean"
    assert len(task.contract_digest) == 64
    resolved = task.resolved_contract()
    physics = resolved["physics"]
    assert isinstance(physics, dict)
    assert resolved["digest"] == task.contract_digest
    assert physics["continuity"] == "div_eps_velocity"
    assert physics["allowed_continuities"] == [
        "div_velocity",
        "div_eps_velocity",
    ]


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_task_schema_requires_exact_integer_one(schema_version: object) -> None:
    """Reject alternate runtime representations and unsupported task versions."""
    task = domain.tasks.registry.get_task("steady_flow")

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        replace(task, schema_version=schema_version)  # type: ignore[arg-type]


def test_task_contract_is_immutable() -> None:
    """
    Attempt scalar and tuple-item mutation on the registered frozen task.

    Both mutations must fail and a new lookup must remain unchanged, protecting
    the process-wide registry from caller-owned state drift.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    with pytest.raises(FrozenInstanceError):
        task.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        task.input_names[0] = "changed"  # type: ignore[index]
    assert domain.tasks.registry.get_task("steady_flow").input_names[0] == "x"


@pytest.mark.parametrize(
    "actual",
    [
        ("x", "y", "kxx", "kyy", "kxy", "eps", "p_bc"),
        ("x", "y", "kxx", "kxy", "kyy", "eps"),
        ("x", "y", "kxx", "kxy", "kxy", "eps", "p_bc"),
    ],
    ids=("swapped-kxy-kyy", "missing-p-bc", "duplicate-kxy"),
)
def test_ordered_contract_validator_rejects_drift(actual: tuple[str, ...]) -> None:
    """
    Vary an expected declaration by swapping, omitting, or duplicating one field.

    Every family must fail ordered-field validation while the canonical target
    remains fixed, protecting channel meaning before tensor use.
    """
    expected = domain.tasks.registry.get_task("steady_flow").input_names
    with pytest.raises(ValueError, match=r"duplicate|does not match|wrong channel order"):
        domain.field_sets.validate_ordered_fields(actual, expected, label="inputs")


def test_public_domain_exports_resolve_and_noncanonical_fields_fail() -> None:
    """
    Resolve the intended domain exports and query noncanonical task/field names.

    Public aliases must reach their canonical objects while noncanonical names
    fail explicitly, keeping the public API limited to canonical names.
    """
    assert domain.tasks.spec.TaskSpec is type(domain.tasks.steady_flow.STEADY_FLOW)
    assert domain.tasks.registry.get_task("steady_flow") is domain.tasks.steady_flow.STEADY_FLOW
    assert domain.tasks.steady_flow.STEADY_FLOW.id == "steady_flow"
    assert domain.fields.require_known_field("eps") == "eps"
    with pytest.raises(ValueError, match="Unknown task"):
        domain.tasks.registry.get_task("unregistered_task")
    with pytest.raises(ValueError, match="Unknown field"):
        domain.fields.require_known_field("unknown_field")
    with pytest.raises(ValueError, match="Unknown field"):
        domain.fields.require_known_field("pbc")


def test_task_declarations_fail_closed_on_runtime_literals_and_layout(
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Construct invalid field roles, metric directions, layouts, and operator axes.

    Each runtime value outside the typed/current 2D contract must fail explicitly,
    because persisted configuration cannot rely on static type checking alone.
    """
    spec = domain.tasks.spec
    with pytest.raises(ValueError, match="unsupported role"):
        spec.FieldSpec("bad", "unsupported", "1", "identity")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported direction"):
        spec.MetricSpec(
            id="bad_direction",
            kind="rmse",
            space="physical",
            fields=("response_b",),
            reduction="element_mean",
            direction="sideways",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="current 2D layout"):
        replace(synthetic_task, tensor_layout=("channel", "batch", "y", "x"))
    with pytest.raises(ValueError, match="current 2D operator/normalizer support"):
        replace(synthetic_task, operator_axes=(1, 2))
