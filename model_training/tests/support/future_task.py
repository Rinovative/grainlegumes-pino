"""Build one internally consistent unregistered future-task contract."""

from src import domain


def build_future_task() -> domain.tasks.spec.TaskSpec:
    """Return a task with distinct fields, per-field metrics, and no physics."""
    spec = domain.tasks.spec
    return spec.TaskSpec(
        id="future_transport",
        schema_version=1,
        inputs=(
            spec.FieldSpec("source_rate", "state", "kg/s", "identity"),
            spec.FieldSpec("ambient_temperature", "state", "K", "identity"),
            spec.FieldSpec("material_fraction", "state", "1", "identity"),
        ),
        outputs=(
            spec.FieldSpec("transported_mass", "state", "kg", "identity"),
            spec.FieldSpec("temperature", "state", "K", "identity"),
        ),
        tensor_layout=("batch", "channel", "y", "x"),
        operator_axes=(2, 3),
        normalization_axes=(0, 2, 3),
        default_datasets=spec.DatasetDefaults(train="future_id", ood=("future_ood",)),
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
                fields=("transported_mass", "temperature"),
                reduction="sample_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_mass",
                kind="rmse",
                space="physical",
                fields=("transported_mass",),
                reduction="element_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_temperature",
                kind="rmse",
                space="physical",
                fields=("temperature",),
                reduction="element_mean",
                direction="minimize",
            ),
        ),
        physics=spec.PhysicsSpec(
            kind="none",
            equation_set="none",
            continuity="none",
            boundary="none",
        ),
    )
