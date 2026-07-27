"""
===============================================================================
domain_task_steady_flow.py
===============================================================================
Declare the authoritative steady two-dimensional porous-flow task contract.

Defines:
  - Exact ordered steady-flow input and output fields
  - Field units, stored representations, tensor axes, and preprocessing
  - Default datasets, semantic metrics, losses, and physics selection

Design principles:
  - The declaration is immutable and contains only canonical identifiers
  - Learned channel counts derive only from the ordered field declarations
  - The task selects physics semantically without implementing equations

This module does NOT:
  - Load, fingerprint, or validate stored datasets
  - Implement derivatives, residual equations, losses, or metrics
  - Define checkpoint, resume, inference, or artifact lifecycle behavior
===============================================================================
"""

from __future__ import annotations

from .domain_task_spec import (
    DatasetDefaults,
    FieldSpec,
    MetricSpec,
    PhysicsSpec,
    PreprocessingSpec,
    TaskSpec,
)

STEADY_FLOW = TaskSpec(
    id="steady_flow",
    schema_version=1,
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
            id="normalized_macro_rmse",
            kind="macro_rmse",
            space="normalized",
            fields=("p", "u", "v"),
            reduction="field_macro_element_mean",
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
    ),
    physics=PhysicsSpec(
        kind="steady_2d_brinkman",
        equation_set="steady_two_dimensional_brinkman",
        continuity="div_eps_velocity",
        allowed_continuities=("div_velocity", "div_eps_velocity"),
        boundary="pressure_inlet_zero_pressure_outlet",
    ),
)
