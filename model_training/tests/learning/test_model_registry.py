# ruff: noqa: S101
"""Exercise semantic registry lookup and model construction boundaries."""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from src import domain, experiments, learning
from support import configs


def test_representative_semantic_identifiers_resolve() -> None:
    """Resolve one supported identifier through each public semantic registry."""
    assert learning.models.factory.resolve_model_kind("fno").kind == "fno"
    assert learning.losses.factory.resolve_data_loss_kind("relative_h1").kind == "relative_h1"
    assert learning.metrics.metrics.resolve_metric_kind("rmse").direction == "minimize"
    assert domain.tasks.registry.resolve_physics("steady_2d_brinkman").kind == "steady_2d_brinkman"


@pytest.mark.parametrize(
    ("resolver", "message"),
    [
        (learning.models.factory.resolve_model_kind, "Unknown model identifier"),
        (learning.losses.factory.resolve_data_loss_kind, "Unknown loss identifier"),
        (learning.metrics.metrics.resolve_metric_kind, "Unknown metric identifier"),
        (domain.tasks.registry.resolve_physics, "Unknown physics identifier"),
    ],
)
def test_unknown_semantic_identifier_fails_at_registry_boundary(
    resolver: Callable[[str], object],
    message: str,
) -> None:
    """Reject a neutral unknown value without depending on registry inventory."""
    with pytest.raises(ValueError, match=message):
        resolver("artificial-unsupported-id")


def test_model_validation_rejects_unsupported_axes_and_uno_depth() -> None:
    """Keep structural limits consistent between validation and construction."""
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


def test_model_factory_uses_only_supplied_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build on explicit CPU without querying CUDA or resolving a policy token."""
    config = experiments.config.loader.resolve_config(configs.direct_config())
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
