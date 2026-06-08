"""
===============================================================================
learning_models_factory.py
===============================================================================
Construct FNO and UNO models from resolved experiment configs.

Responsibilities:
  - Build FNO models from channel, mode and layer settings
  - Build UNO models with configured mode schedules
  - Dispatch architecture names from resolved configs

Design principles:
  - Neuraloperator provides the architecture implementations
  - Parameter passing stays explicit and traceable
  - Device placement happens only when requested by the caller

Boundaries:
  - UNO checkpoint wrapping belongs to learning.models.uno
  - Training orchestration belongs to learning.training.loop
===============================================================================
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
from neuralop.models import FNO, UNO

from . import learning_models_uno as uno

_SkipConnection = Literal["linear", "identity", "soft-gating"]
_UNO_LAYERS_5 = 5
_UNO_LAYERS_7 = 7
_MIN_UNO_MODE = 8
_FNO_MODE_DIMENSIONS = 2


def _validate_skip(name: str, value: str) -> _SkipConnection:
    """Validate a neuralop skip-connection option."""
    if value not in ("linear", "identity", "soft-gating"):
        msg = f"{name} must be one of 'linear', 'identity', 'soft-gating', got: {value!r}"
        raise ValueError(msg)
    return cast("_SkipConnection", value)


def build_fno(
    in_channels: int,
    out_channels: int,
    n_modes: list[int] | tuple[int, int],
    hidden_channels: int,
    n_layers: int,
    lifting_channel_ratio: float = 2,
    projection_channel_ratio: float = 2,
    fno_skip: str = "linear",
    channel_mlp_skip: str = "soft-gating",
    implementation: str = "factorized",
    device: torch.device | str | None = None,
) -> FNO:
    """
    Build a Fourier Neural Operator (FNO) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : list[int] | tuple[int, ...]
        Number of Fourier modes for each spatial dimension
    hidden_channels : int
        Number of hidden channels
    n_layers : int
        Number of Fourier layers
    lifting_channel_ratio : float, optional
        Channel ratio for lifting layer (default: 2)
    projection_channel_ratio : float, optional
        Channel ratio for projection layer (default: 2)
    fno_skip : str, optional
        Skip connection type for FNO blocks (default: "linear")
    channel_mlp_skip : str, optional
        Skip connection type for channel MLP (default: "soft-gating")
    implementation : str, optional
        Implementation type (default: "factorized")
    device : torch.device | str | None, optional
        Device to place model on (default: None - caller handles)

    Returns
    -------
    FNO
        Initialized FNO model

    """
    n_modes_tuple = tuple(int(mode) for mode in n_modes)
    if len(n_modes_tuple) != _FNO_MODE_DIMENSIONS:
        msg = f"FNO requires exactly two n_modes entries, got: {n_modes!r}"
        raise ValueError(msg)

    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes_tuple,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=projection_channel_ratio,
        fno_skip=_validate_skip("fno_skip", fno_skip),
        channel_mlp_skip=_validate_skip("channel_mlp_skip", channel_mlp_skip),
        implementation=implementation,
    )

    if device is not None:
        model.to(device)

    return model


def build_uno(
    in_channels: int,
    out_channels: int,
    n_layers: int,
    hidden_channels: int,
    modes_x: int,
    modes_y: int,
    mode_ratio: float = 0.5,
    uno_scalings: list[list[float]] | None = None,
    channel_mlp_skip: str = "linear",
    device: torch.device | str | None = None,
    use_checkpoint: bool = True,
) -> UNO | uno.UNOWithCheckpoint:
    """
    Build a U-shaped Neural Operator (UNO) model.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_layers : int
        Number of UNO layers
    hidden_channels : int
        Number of hidden channels
    modes_x : int
        Base number of Fourier modes in x-direction
    modes_y : int
        Base number of Fourier modes in y-direction
    mode_ratio : float, optional
        Ratio for computing intermediate layer modes (default: 0.5)
    uno_scalings : list[list[float]] | None, optional
        Layer-specific spatial scalings (default: None - auto-computed)
    channel_mlp_skip : str, optional
        Skip connection type for channel MLP (default: "linear")
    device : torch.device | str | None, optional
        Device to place model on (default: None - caller handles)
    use_checkpoint : bool, optional
        If True, use UNOWithCheckpoint wrapper (default: True)

    Returns
    -------
    UNO | uno.UNOWithCheckpoint
        Initialized UNO model, optionally wrapped with checkpoint support

    """
    # Auto-compute mode schedule if not provided
    if uno_scalings is None:
        if n_layers == _UNO_LAYERS_5:
            uno_scalings = [
                [1.0, 1.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        elif n_layers == _UNO_LAYERS_7:
            uno_scalings = [
                [1.0, 1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        else:
            msg = f"Unsupported n_layers={n_layers}. Provide explicit uno_scalings."
            raise ValueError(msg)
    elif len(uno_scalings) != n_layers:
        msg = f"uno_scalings length must match n_layers={n_layers}, got {len(uno_scalings)}."
        raise ValueError(msg)

    # Compute mode schedule from base modes and ratio
    mid_x = max(_MIN_UNO_MODE, int(modes_x * mode_ratio))
    mid_y = max(_MIN_UNO_MODE, int(modes_y * mode_ratio))

    if n_layers == _UNO_LAYERS_5:
        uno_n_modes = [
            [modes_x, modes_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [modes_x, modes_y],
        ]
    elif n_layers == _UNO_LAYERS_7:
        uno_n_modes = [
            [modes_x, modes_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [mid_x, mid_y],
            [modes_x, modes_y],
        ]
    else:
        msg = f"Unsupported n_layers={n_layers}. Provide a supported UNO layer count."
        raise ValueError(msg)

    uno_out_channels = [hidden_channels] * n_layers

    # Build UNO
    model_class = uno.UNOWithCheckpoint if use_checkpoint else UNO
    model = model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        uno_out_channels=uno_out_channels,
        uno_n_modes=uno_n_modes,
        uno_scalings=uno_scalings,
        channel_mlp_skip=_validate_skip("channel_mlp_skip", channel_mlp_skip),
    )

    if device is not None:
        model.to(device)

    return model


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """
    Build a model from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with keys:
        - model.architecture: "FNO", "PI-FNO", "UNO", or "PI-UNO"
        - model.params: architecture-specific parameters
        - run.device: target device

    Returns
    -------
    torch.nn.Module
        Initialized model

    Raises
    ------
    ValueError
        If architecture is unknown or required parameters are missing

    """
    arch = config["model"]["architecture"]
    params = config["model"].get("params", {})
    device = config["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Normalize architecture name for builder dispatch
    builder_name = arch.replace("-", "").lower()

    if builder_name in ("fno", "pifno"):
        return build_fno(**params, device=device)
    if builder_name in ("uno", "piuno"):
        return build_uno(**params, device=device, use_checkpoint=True)
    msg = f"Unknown model architecture: {arch}"
    raise ValueError(msg)
