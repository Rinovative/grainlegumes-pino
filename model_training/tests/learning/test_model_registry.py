# ruff: noqa: S101
"""Verify semantic model, loss, metric, and physics implementation registries."""

import pytest
from src import domain, learning


def test_semantic_model_loss_metric_and_physics_ids_resolve() -> None:
    """Public semantic identifiers resolve to their implementation specifications."""
    assert learning.models.factory.resolve_model_kind("fno").kind == "fno"
    assert learning.models.factory.resolve_model_kind("uno").kind == "uno"
    assert learning.losses.factory.resolve_data_loss_kind("relative_h1").kind == "relative_h1"
    assert learning.losses.factory.resolve_data_loss_kind("relative_l2").kind == "relative_l2"
    assert learning.metrics.metrics.resolve_metric_kind("relative_h1").direction == "minimize"
    assert learning.metrics.metrics.resolve_metric_kind("rmse").direction == "minimize"
    assert domain.tasks.registry.resolve_physics("steady_2d_brinkman").kind == "steady_2d_brinkman"
    assert learning.losses.factory.resolve_physics_loss_implementation("steady_2d_brinkman").kind == "steady_2d_brinkman"


def test_nonsemantic_implementation_identifiers_fail_clearly() -> None:
    """Class, display, shorthand, and unknown names are not public identifiers."""
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
    """Public validation and builders agree on current two-dimensional model support."""
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
