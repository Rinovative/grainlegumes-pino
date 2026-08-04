"""
===============================================================================
domain_task_steady_flow.py
===============================================================================
Declare the authoritative steady two-dimensional porous-flow task contract.

Responsibilities:
  - Exact ordered steady-flow input and output fields
  - Field units, stored representations, tensor axes, and preprocessing
  - Fallback datasets for omitted config selection, semantic metrics, losses, and physics

Design principles:
  - The declaration is immutable and contains only canonical identifiers
  - Learned channel counts derive only from the ordered field declarations
  - The task selects physics semantically without implementing equations

This module does NOT:
  - Load, fingerprint, or validate stored datasets
  - Implement derivatives, residual equations, losses, or metrics
  - Select datasets for explicit executable experiment recipes
  - Define checkpoint, resume, inference, or artifact lifecycle behavior
===============================================================================
"""

from __future__ import annotations

from src.domain.physics import domain_physics_contracts as physics_contracts

from .domain_task_spec import (
    TASK_SCHEMA_VERSION,
    DatasetDefaults,
    FieldSpec,
    MetricSpec,
    OutputGroupSpec,
    PhysicsSpec,
    PreprocessingSpec,
    TaskSpec,
)

STEADY_FLOW = TaskSpec(
    id="steady_flow",
    schema_version=TASK_SCHEMA_VERSION,
    inputs=(
        FieldSpec("x", "coordinate", "m", "identity"),
        FieldSpec("y", "coordinate", "m", "identity"),
        FieldSpec(
            "kxx",
            "permeability",
            "m^2",
            "dimensionless_log10_ratio_to_1_m2",
        ),
        FieldSpec(
            "kxy",
            "permeability",
            "m^2",
            "dimensionless_cross_component_ratio_to_geometric_mean",
        ),
        FieldSpec(
            "kyy",
            "permeability",
            "m^2",
            "dimensionless_log10_ratio_to_1_m2",
        ),
        FieldSpec("eps", "porosity", "1", "identity", source_name="int4(x,y)"),
        FieldSpec("p_bc", "boundary", "Pa", "identity", source_name="int5(x,y)"),
    ),
    outputs=(
        FieldSpec("p", "state", "Pa", "identity_before_train_normalization"),
        FieldSpec("u", "state", "m/s", "identity_before_train_normalization"),
        FieldSpec("v", "state", "m/s", "identity_before_train_normalization"),
    ),
    output_groups=(
        OutputGroupSpec("pressure", ("p",)),
        OutputGroupSpec("velocity", ("u", "v")),
    ),
    tensor_layout=("batch", "channel", "y", "x"),
    operator_axes=(2, 3),
    normalization_axes=(0, 2, 3),
    default_datasets=DatasetDefaults(
        train="lhs_var80_seed3001",
        ood=("lhs_var120_seed4001",),
    ),
    preprocessing=PreprocessingSpec(
        input_normalization="train_fitted_per_channel_standardization",
        output_normalization="train_fitted_per_channel_standardization",
        fit_split="train",
    ),
    data_losses=("relative_h1", "relative_l2"),
    default_metrics=(
        MetricSpec(
            id="normalized_group_macro_rmse",
            kind="group_macro_rmse",
            space="physical",
            fields=("p", "u", "v"),
            reduction="group_macro_element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_rmse_p",
            kind="rmse",
            space="normalized",
            fields=("p",),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_rmse_u",
            kind="rmse",
            space="normalized",
            fields=("u",),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_rmse_v",
            kind="rmse",
            space="normalized",
            fields=("v",),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_velocity_vector_rmse",
            kind="group_rmse",
            space="physical",
            fields=("u", "v"),
            reduction="group_element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_rmse",
            kind="rmse",
            space="normalized",
            fields=("p", "u", "v"),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_relative_l2",
            kind="relative_l2",
            space="normalized",
            fields=("p", "u", "v"),
            reduction="sample_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_relative_h1",
            kind="relative_h1",
            space="normalized",
            fields=("p", "u", "v"),
            reduction="sample_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="physical_rmse_p",
            kind="rmse",
            space="physical",
            fields=("p",),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="physical_rmse_u",
            kind="rmse",
            space="physical",
            fields=("u",),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="physical_rmse_v",
            kind="rmse",
            space="physical",
            fields=("v",),
            reduction="element_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="physical_velocity_vector_rmse",
            kind="vector_rmse",
            space="physical",
            fields=("u", "v"),
            reduction="vector_element_mean",
            direction="minimize",
        ),
    ),
    physics=PhysicsSpec(
        kind=physics_contracts.STEADY_BRINKMAN_KIND,
        equation_set=physics_contracts.STEADY_BRINKMAN_EQUATION_SET,
        continuity=physics_contracts.DEFAULT_CONTINUITY_KIND,
        allowed_continuities=physics_contracts.available_continuity_kinds(),
        boundary=physics_contracts.PRESSURE_BOUNDARY_KIND,
    ),
)
