"""
Build the unregistered synthetic TaskSpec used by generic contract tests.

The test-only three-input/two-output contract uses arbitrary neutral field names,
units, metrics, and dataset IDs to prove current TaskSpec consumers do not hardcode
the steady-flow channel contract. It is not a production task implementation and
is never registered, trained, or published.
"""

from src import domain


def build_synthetic_task() -> domain.tasks.spec.TaskSpec:
    """
    Return a coherent task with distinct fields, per-field metrics, and no physics.

    Returns
    -------
    domain.tasks.spec.TaskSpec
        Unregistered two-output contract used to prove consumers derive channel
        counts, names, units, and metric definitions from TaskSpec.

    Notes
    -----
    The fixture intentionally simplifies production assumptions: fields use
    arbitrary units, preprocessing uses generic standardization, dataset IDs are
    non-executable labels, and physics kind/equations/boundary are ``none``.

    """
    spec = domain.tasks.spec
    return spec.TaskSpec(
        id="synthetic_field_task",
        schema_version=1,
        inputs=(
            spec.FieldSpec("feature_a", "state", "unit_in_a", "identity"),
            spec.FieldSpec("feature_b", "state", "unit_in_b", "identity"),
            spec.FieldSpec("feature_c", "state", "unit_in_c", "identity"),
        ),
        outputs=(
            spec.FieldSpec("response_a", "state", "unit_out_a", "identity"),
            spec.FieldSpec("response_b", "state", "unit_out_b", "identity"),
        ),
        tensor_layout=("batch", "channel", "y", "x"),
        operator_axes=(2, 3),
        normalization_axes=(0, 2, 3),
        default_datasets=spec.DatasetDefaults(train="synthetic_train", ood=("synthetic_ood",)),
        preprocessing=spec.PreprocessingSpec(
            input_normalization="standard",
            output_normalization="standard",
            fit_split="train",
        ),
        data_losses=("relative_l2",),
        default_metrics=(
            spec.MetricSpec(
                id="normalized_relative_l2",
                kind="relative_l2",
                space="normalized",
                fields=("response_a", "response_b"),
                reduction="sample_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_response_a",
                kind="rmse",
                space="physical",
                fields=("response_a",),
                reduction="element_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_response_b",
                kind="rmse",
                space="physical",
                fields=("response_b",),
                reduction="element_mean",
                direction="minimize",
            ),
        ),
        physics=spec.PhysicsSpec(
            kind="none",
            equation_set="none",
            continuity="none",
            allowed_continuities=("none",),
            boundary="none",
        ),
    )
