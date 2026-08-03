# ruff: noqa: S101
"""
Protect public semantic registries for models, losses, metrics, and physics.

The tests require canonical identifiers to resolve, neutral unsupported identifiers
to fail, structural dimensionality/depth constraints to agree with builders, and
CPU construction not to query CUDA. Numerical model quality and training lifecycle
are deliberately covered elsewhere.
"""

import importlib
import sys
from typing import Any

import pytest
import torch
from src import domain, experiments, learning
from support import configs


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


def test_current_uno_dependency_resampling_remains_bicubic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the maintained 2D neuraloperator resampling fact used by preflight."""
    resampling = importlib.import_module("neuralop.layers.resample")
    observed: dict[str, Any] = {}

    def interpolate(value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        observed.update(kwargs)
        return value

    monkeypatch.setattr(resampling.F, "interpolate", interpolate)
    value = torch.zeros(1, 1, 8, 8)
    assert resampling.resample(value, 0.5, [2, 3]) is value
    assert observed["mode"] == "bicubic"
    assert observed["align_corners"] is True
    assert learning.models.factory.UNO_RESAMPLING_MODE == "bicubic"


def test_unknown_semantic_identifiers_fail_clearly() -> None:
    """Reject one neutral unsupported identifier at each canonical registry."""
    with pytest.raises(ValueError, match="Unknown model identifier"):
        learning.models.factory.resolve_model_kind("unsupported")
    with pytest.raises(ValueError, match="Unknown loss identifier"):
        learning.losses.factory.resolve_data_loss_kind("unsupported")
    with pytest.raises(ValueError, match="Unknown metric identifier"):
        learning.metrics.metrics.resolve_metric_kind("unsupported")
    with pytest.raises(ValueError, match="Unknown physics identifier"):
        domain.tasks.registry.resolve_physics("unsupported")


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


def test_real_uno_constructor_has_no_repeated_skip_debug_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Construct a small real UNO without dependency-owned skip-option print noise."""
    learning.models.factory.build_uno(
        in_channels=4,
        out_channels=3,
        n_layers=5,
        hidden_channels=4,
        modes_x=8,
        modes_y=8,
    )

    captured = capsys.readouterr()
    assert "fno_skip=" not in captured.out
    assert "channel_mlp_skip=" not in captured.out


def test_uno_noise_filter_reemits_unrelated_diagnostics_and_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Filter only exact known debug lines while preserving architecture args and failures."""
    neuralop_models = importlib.import_module("neuralop.models")
    observed: dict[str, Any] = {}

    class FailingUNO:
        def __init__(self, **kwargs: Any) -> None:
            observed.update(kwargs)
            print("fno_skip='linear'")
            print("channel_mlp_skip='linear'")
            print("preserved constructor diagnostic")
            print("preserved constructor warning", file=sys.stderr)
            message = "visible constructor failure"
            raise RuntimeError(message)

    monkeypatch.setattr(neuralop_models, "UNO", FailingUNO)
    with pytest.raises(RuntimeError, match="visible constructor failure"):
        learning.models.factory.build_uno(
            in_channels=4,
            out_channels=3,
            n_layers=5,
            hidden_channels=4,
            modes_x=8,
            modes_y=8,
            channel_mlp_skip="identity",
        )

    captured = capsys.readouterr()
    assert "fno_skip=" not in captured.out
    assert "channel_mlp_skip=" not in captured.out
    assert "preserved constructor diagnostic" in captured.out
    assert "preserved constructor warning" in captured.err
    assert observed["channel_mlp_skip"] == "identity"
    assert observed["uno_n_modes"][0] == [8, 8]


def test_model_factory_uses_only_the_required_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Build a tiny model on explicit CPU while making every CUDA query fail.

    Construction must use only the concrete device supplied and reject string or
    unindexed CUDA inputs, preserving device resolution as orchestration ownership.
    """
    config_path = configs.experiment_config_path(model_kind="fno", physics_enabled=False)
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
