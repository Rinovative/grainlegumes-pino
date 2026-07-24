# ruff: noqa: S101, PLR2004
"""Verify the immutable semantic task and ordered field contract."""

from dataclasses import FrozenInstanceError, replace

import pytest
from src import domain


def test_steady_flow_contract_is_exact_and_task_owned() -> None:
    """The sole registered task owns every fixed tensor/config semantic."""
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
    assert task.physics.kind == "steady_2d_brinkman"
    assert [metric.id for metric in task.default_metrics] == [
        "normalized_relative_h1",
        "normalized_rmse",
        "physical_rmse_p",
        "physical_rmse_u",
        "physical_rmse_v",
    ]
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
    assert task.resolved_contract()["digest"] == task.contract_digest


def test_task_contract_is_immutable() -> None:
    """Registry callers cannot mutate the authoritative descriptor."""
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
    """Swaps, missing fields, and duplicates fail before tensor use."""
    expected = domain.tasks.registry.get_task("steady_flow").input_names
    with pytest.raises(ValueError, match=r"duplicate|does not match|wrong channel order"):
        domain.field_sets.validate_ordered_fields(actual, expected, label="inputs")


def test_public_domain_exports_resolve_and_noncanonical_fields_fail() -> None:
    """Public imports resolve while noncanonical field names fail."""
    assert domain.tasks.spec.TaskSpec
    assert domain.tasks.registry.get_task
    assert domain.tasks.steady_flow.STEADY_FLOW.id == "steady_flow"
    assert domain.fields.require_known_field("eps") == "eps"
    with pytest.raises(ValueError, match="Unknown task"):
        domain.tasks.registry.get_task("transient_heat_moisture")
    with pytest.raises(ValueError, match="Unknown field"):
        domain.fields.require_known_field("phi")
    with pytest.raises(ValueError, match="Unknown field"):
        domain.fields.require_known_field("pbc")


def test_task_declarations_fail_closed_on_runtime_literals_and_layout(
    future_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Static Literal hints and current 2D tensor support are enforced at runtime."""
    spec = domain.tasks.spec
    with pytest.raises(ValueError, match="unsupported role"):
        spec.FieldSpec("bad", "unsupported", "1", "identity")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported direction"):
        spec.MetricSpec(
            id="bad_direction",
            kind="rmse",
            space="physical",
            fields=("temperature",),
            reduction="element_mean",
            direction="sideways",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="current 2D layout"):
        replace(future_task, tensor_layout=("channel", "batch", "y", "x"))
    with pytest.raises(ValueError, match="current 2D operator/normalizer support"):
        replace(future_task, operator_axes=(1, 2))
