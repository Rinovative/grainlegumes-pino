# ruff: noqa: S101
"""
Protect public semantic registries for models, losses, metrics, derivatives, and physics.

The tests require canonical identifiers to resolve, implementation/display aliases
to fail, structural dimensionality/depth constraints to agree with builders, and
CPU construction not to query CUDA. Numerical model quality and training lifecycle
are deliberately covered elsewhere.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from src import domain, experiments, learning


def test_semantic_model_loss_metric_and_physics_ids_resolve() -> None:
    """
    Resolve every maintained model, data-loss, metric, and physics identifier.

    Each public semantic ID must reach its registered specification, protecting
    configuration from depending on implementation class names.
    """
    assert learning.models.factory.resolve_model_kind("fno").kind == "fno"
    assert learning.models.factory.resolve_model_kind("uno").kind == "uno"
    assert learning.losses.factory.resolve_data_loss_kind("relative_h1").kind == "relative_h1"
    assert learning.losses.factory.resolve_data_loss_kind("relative_l2").kind == "relative_l2"
    assert learning.metrics.metrics.resolve_metric_kind("relative_h1").direction == "minimize"
    assert learning.metrics.metrics.resolve_metric_kind("rmse").direction == "minimize"
    assert domain.tasks.registry.resolve_physics("steady_2d_brinkman").kind == "steady_2d_brinkman"
    assert learning.losses.factory.resolve_physics_loss_implementation("steady_2d_brinkman").kind == "steady_2d_brinkman"


def test_nonsemantic_implementation_identifiers_fail_clearly() -> None:
    """
    Query class names, display labels, shorthand, capitalization drift, and unknown IDs.

    Every nonsemantic family must fail at its owning registry so no undocumented
    compatibility vocabulary enters persisted configuration.
    """
    for identifier in ("PI-FNO", "PINOLoss", "FNO"):
        with pytest.raises(ValueError, match="Unknown model identifier"):
            learning.models.factory.resolve_model_kind(identifier)
    for identifier in ("PINOLoss", "H1Loss", "LpLoss", "h1"):
        with pytest.raises(ValueError, match="Unknown loss identifier"):
            learning.losses.factory.resolve_data_loss_kind(identifier)
    with pytest.raises(ValueError, match="Unknown metric identifier"):
        learning.metrics.metrics.resolve_metric_kind("RMSEOverall")
    with pytest.raises(ValueError, match="Unknown physics identifier"):
        domain.tasks.registry.resolve_physics("PINOLoss")
    with pytest.raises(ValueError, match="Unknown physics loss identifier"):
        learning.losses.factory.resolve_physics_loss_implementation("PINOLoss")


def test_model_factory_rejects_unsupported_dimensionality_and_uno_depth() -> None:
    """
    Request a three-axis FNO and unsupported three-layer UNO through public boundaries.

    Validation and direct UNO construction must reject the same structural limits,
    preventing a config from validating only to fail deeper in model creation.
    """
    fno_params = {
        "in_channels": 7,
        "out_channels": 3,
        "n_modes": [8, 8, 8],
        "hidden_channels": 4,
        "n_layers": 2,
    }
    with pytest.raises(ValueError, match="exactly two operator axes"):
        learning.models.factory.validate_model_params(
            "fno",
            fno_params,
            require_channels=True,
            operator_dimensionality=3,
        )

    uno_params = {
        "hidden_channels": 4,
        "modes_x": 8,
        "modes_y": 8,
        "n_layers": 3,
    }
    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        learning.models.factory.validate_model_params(
            "uno",
            uno_params,
            require_channels=False,
            operator_dimensionality=2,
        )

    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        learning.models.factory.build_uno(
            in_channels=7,
            out_channels=3,
            n_layers=3,
            hidden_channels=4,
            modes_x=8,
            modes_y=8,
            uno_scalings=[[1.0, 1.0]] * 3,
        )


def test_model_factory_uses_only_the_required_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Build a tiny model on explicit CPU while making every CUDA query fail.

    Construction must use only the concrete device supplied and reject string or
    unindexed CUDA inputs, preserving device resolution as orchestration ownership.
    """
    config_path = Path(__file__).parents[2] / "configs/experiments/steady_flow_fno.yaml"
    config = experiments.config.loader.load_and_resolve_config(config_path)
    config["model"]["params"].update(
        {"n_modes": [2, 2], "hidden_channels": 2, "n_layers": 1},
    )

    def unexpected_cuda_query(*_args: Any, **_kwargs: Any) -> Any:
        message = "model factory queried CUDA availability"
        raise AssertionError(message)

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_query)
    model = learning.models.factory.build_model(config, device=torch.device("cpu"))
    assert {parameter.device for parameter in model.parameters()} == {torch.device("cpu")}

    with pytest.raises(TypeError, match=r"concrete CPU or CUDA torch\.device"):
        learning.models.factory.build_model(config, device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="indexed CUDA device"):
        learning.models.factory.build_model(config, device=torch.device("cuda"))
