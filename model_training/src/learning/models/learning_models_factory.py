"""
===============================================================================
learning_models_factory.py
===============================================================================
Construct FNO and UNO models from resolved experiment configs.

Responsibilities:
  - Build FNO models from channel, mode and layer settings
  - Build UNO models with configured mode schedules
  - Declare the maintained neuraloperator UNO resampling semantics
  - Resolve semantic model identifiers from strict registries

Design principles:
  - Neuraloperator provides the architecture implementations
  - Parameter passing stays explicit and traceable
  - Semantic identifiers remain independent of implementation class names
  - Device placement happens only when requested by the caller

This module does NOT:
  - Implement FNO or UNO architectures; ``neuraloperator`` supplies them
  - Orchestrate training; ``learning.training.loop`` owns execution
  - Derive task channels or axes; task contracts and config resolution own them
===============================================================================
"""

from __future__ import annotations

import copy
import io
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import torch
    from neuralop.models import FNO, UNO

_SkipConnection = Literal["linear", "identity", "soft-gating"]
_UNO_LAYERS_5 = 5
_UNO_LAYERS_7 = 7
_MIN_UNO_MODE = 8
_FNO_MODE_DIMENSIONS = 2
UNO_RESAMPLING_MODE = "bicubic"


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
    from neuralop.models import FNO  # noqa: PLC0415

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


@contextmanager
def _filter_uno_constructor_stdout() -> Iterator[None]:
    """Suppress only neuraloperator's known UNO skip-option debug lines."""
    captured = io.StringIO()
    known_noise = {f"{name}={value!r}" for name in ("fno_skip", "channel_mlp_skip") for value in ("linear", "identity", "soft-gating", None)}
    try:
        with redirect_stdout(captured):
            yield
    finally:
        for line in captured.getvalue().splitlines(keepends=True):
            if line.rstrip("\r\n") not in known_noise:
                sys.stdout.write(line)


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
) -> UNO:
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

    Returns
    -------
    UNO
        Initialized UNO model

    """
    from neuralop.models import UNO  # noqa: PLC0415

    if n_layers not in {_UNO_LAYERS_5, _UNO_LAYERS_7}:
        msg = f"UNO supports exactly {_UNO_LAYERS_5} or {_UNO_LAYERS_7} layers, got {n_layers}."
        raise ValueError(msg)

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
        msg = f"Internal UNO layer validation failed for n_layers={n_layers}."
        raise AssertionError(msg)

    uno_out_channels = [hidden_channels] * n_layers

    with _filter_uno_constructor_stdout():
        model = UNO(
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


@dataclass(frozen=True, slots=True)
class ModelKindSpec:
    """
    Describe one semantic model kind.

    Attributes
    ----------
    kind : str
        Canonical saved configuration identifier.
    builder : Callable[..., torch.nn.Module]
        Internal model-construction callable.
    defaults : Mapping[str, Any]
        Immutable implementation defaults.
    allowed_params : frozenset[str]
        Accepted resolved model parameter names.
    required_params : frozenset[str]
        Required resolved model parameter names.

    """

    kind: str
    builder: Callable[..., torch.nn.Module]
    defaults: Mapping[str, Any]
    allowed_params: frozenset[str]
    required_params: frozenset[str]


_MODEL_KINDS = MappingProxyType(
    {
        "fno": ModelKindSpec(
            kind="fno",
            builder=build_fno,
            defaults=MappingProxyType(
                {
                    "lifting_channel_ratio": 2,
                    "projection_channel_ratio": 2,
                    "fno_skip": "linear",
                    "channel_mlp_skip": "soft-gating",
                    "implementation": "factorized",
                }
            ),
            allowed_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_modes",
                    "hidden_channels",
                    "n_layers",
                    "lifting_channel_ratio",
                    "projection_channel_ratio",
                    "fno_skip",
                    "channel_mlp_skip",
                    "implementation",
                }
            ),
            required_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_modes",
                    "hidden_channels",
                    "n_layers",
                }
            ),
        ),
        "uno": ModelKindSpec(
            kind="uno",
            builder=build_uno,
            defaults=MappingProxyType({"channel_mlp_skip": "linear"}),
            allowed_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_layers",
                    "hidden_channels",
                    "modes_x",
                    "modes_y",
                    "mode_ratio",
                    "uno_scalings",
                    "channel_mlp_skip",
                }
            ),
            required_params=frozenset(
                {
                    "in_channels",
                    "out_channels",
                    "n_layers",
                    "hidden_channels",
                    "modes_x",
                    "modes_y",
                }
            ),
        ),
    }
)


def available_model_kinds() -> tuple[str, ...]:
    """
    Return registered semantic model identifiers.

    Returns
    -------
    tuple[str, ...]
        Exact model kinds accepted by the factory.

    """
    return tuple(sorted(_MODEL_KINDS))


def resolve_model_kind(kind: str) -> ModelKindSpec:
    """
    Resolve an exact semantic model identifier.

    Parameters
    ----------
    kind : str
        Canonical model kind.

    Returns
    -------
    ModelKindSpec
        Immutable model schema and implementation descriptor.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    try:
        return _MODEL_KINDS[kind]
    except KeyError as error:
        available = ", ".join(available_model_kinds())
        msg = f"Unknown model identifier {kind!r}. Available models: {available}."
        raise ValueError(msg) from error


def model_defaults(kind: str) -> dict[str, Any]:
    """
    Return an isolated copy of defaults owned by a model kind.

    Parameters
    ----------
    kind : str
        Canonical model kind.

    Returns
    -------
    dict[str, Any]
        Mutable copy of the registered model defaults.

    Raises
    ------
    ValueError
        If `kind` is not registered.

    """
    return copy.deepcopy(dict(resolve_model_kind(kind).defaults))


def validate_model_params(
    kind: str,
    params: Mapping[str, Any],
    *,
    require_channels: bool,
    operator_dimensionality: int,
) -> None:
    """
    Validate model parameters against a semantic model schema.

    Parameters
    ----------
    kind : str
        Canonical model kind.
    params : Mapping[str, Any]
        Candidate model parameter mapping.
    require_channels : bool
        Whether derived input/output channel parameters must already be present.
    operator_dimensionality : int
        Number of task-owned spatial operator axes.

    Raises
    ------
    ValueError
        If the model kind, parameter names, required values, or mode shape are invalid.

    """
    spec = resolve_model_kind(kind)
    if operator_dimensionality != _FNO_MODE_DIMENSIONS:
        msg = f"Model kind {kind!r} currently supports exactly two operator axes, got {operator_dimensionality}."
        raise ValueError(msg)
    unknown = sorted(set(params).difference(spec.allowed_params))
    if unknown:
        msg = f"model.params contains unknown key(s) for {kind!r}: {unknown}."
        raise ValueError(msg)

    required = set(spec.required_params)
    if not require_channels:
        required.difference_update({"in_channels", "out_channels"})
    missing = sorted(required.difference(params))
    if missing:
        msg = f"model.params is missing required key(s) for {kind!r}: {missing}."
        raise ValueError(msg)

    if kind == "fno" and "n_modes" in params:
        n_modes = params["n_modes"]
        if not isinstance(n_modes, (list, tuple)) or len(n_modes) != operator_dimensionality:
            msg = f"model.params.n_modes must contain exactly {operator_dimensionality} entries for this task, got: {n_modes!r}."
            raise ValueError(msg)

    if kind == "uno":
        n_layers = params["n_layers"]
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers not in (_UNO_LAYERS_5, _UNO_LAYERS_7):
            msg = f"UNO supports exactly {_UNO_LAYERS_5} or {_UNO_LAYERS_7} layers, got {n_layers!r}."
            raise ValueError(msg)


def build_model(config: dict[str, Any], *, device: torch.device) -> torch.nn.Module:
    """
    Build a registered semantic model from a resolved configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved configuration containing model kind, parameters, and task contract.
    device : torch.device
        Concrete device resolved by the top-level runtime service.

    Returns
    -------
    torch.nn.Module
        Constructed FNO or UNO implementation.

    Raises
    ------
    TypeError
        If required resolved config sections have invalid types.
    ValueError
        If the semantic model identifier or parameters are invalid.

    """
    import torch  # noqa: PLC0415

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        msg = "Resolved config must contain a model mapping."
        raise TypeError(msg)
    kind = model_config.get("kind")
    if not isinstance(kind, str):
        msg = "Resolved config must contain model.kind as a string."
        raise TypeError(msg)
    params = model_config.get("params")
    if not isinstance(params, dict):
        msg = "Resolved config must contain model.params as a mapping."
        raise TypeError(msg)

    task_contract = config.get("task_contract", {})
    operator_dimensionality = int(task_contract.get("operator_dimensionality", _FNO_MODE_DIMENSIONS))
    validate_model_params(
        kind,
        params,
        require_channels=True,
        operator_dimensionality=operator_dimensionality,
    )
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Model construction requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda" and device.index is None:
        msg = "Model construction requires an indexed CUDA device resolved by the runtime boundary."
        raise ValueError(msg)
    return resolve_model_kind(kind).builder(**params, device=device)
