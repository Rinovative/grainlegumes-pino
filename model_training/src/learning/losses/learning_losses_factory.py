"""
===============================================================================
learning_losses_factory.py
===============================================================================
Construct supervised, PINO and evaluation losses from configs.

Responsibilities:
  - Build supervised data losses
  - Build PINO losses with configured derivative modes
  - Build evaluation loss dictionaries
  - Inject normalizer requirements for physical-channel metrics

Design principles:
  - Factory functions preserve YAML semantics
  - Loss mathematics stays in loss classes and derivative modules
  - Device placement remains a caller responsibility

Boundaries:
  - PINO residual math belongs to learning.losses.pino
  - Training dynamics belong to learning.training.loop
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from neuralop import H1Loss, LpLoss

from src import learning

from . import learning_losses_pino as pino

if TYPE_CHECKING:
    from torch import nn


def build_supervised_loss(
    data_loss: str = "h1",
) -> nn.Module:
    """
    Build supervised loss (data loss only, no physics).

    Parameters
    ----------
    data_loss : str, optional
        Data loss type ("h1" or "l2", default: "h1")

    Returns
    -------
    nn.Module
        Supervised loss instance

    Raises
    ------
    ValueError
        If data_loss type is unknown

    """
    if data_loss == "h1":
        return _as_module(H1Loss(d=2))
    if data_loss == "l2":
        return _as_module(LpLoss(d=2, p=2))
    msg = f"Unknown data loss: {data_loss}"
    raise ValueError(msg)


def build_pino_loss(
    loss_type: str,
    data_loss: str = "h1",
    lambda_phys: float = 1e-4,
    lambda_p: float = 5e-4,
    grad_mode: str = "fft_reflect",
    interior_pad: int = 2,
    in_normalizer: Any | None = None,
    out_normalizer: Any | None = None,
) -> nn.Module:
    """
    Build PINO loss (data loss + physics residuals).

    Parameters
    ----------
    loss_type : str
        PINO loss type: "ps_eps", "ps_div", "sp_eps", "sp_div"
        - ps: physical space derivatives
        - sp: spectral space (FFT) derivatives
        - eps: div(ε u) continuity (conservative)
        - div: div(u) continuity (plain)
    data_loss : str, optional
        Data loss component ("h1" or "l2", default: "h1")
    lambda_phys : float, optional
        Physics loss weight (default: 1e-4)
    lambda_p : float, optional
        Pressure BC loss weight (default: 5e-4)
    grad_mode : str, optional
        Gradient computation mode for spectral ("fft" or "fft_reflect", default: "fft_reflect")
    interior_pad : int, optional
        Interior cropping for gradient-based losses (default: 2)
    in_normalizer : Any | None, optional
        Input normalizer (default: None)
    out_normalizer : Any | None, optional
        Output normalizer (default: None)

    Returns
    -------
    nn.Module
        PINO loss instance

    Raises
    ------
    ValueError
        If loss_type or data_loss is unknown

    """
    # Build data loss component
    try:
        data_loss_fn = build_supervised_loss(data_loss=data_loss)
    except ValueError as e:
        msg = f"Invalid data_loss in PINO config: {e}"
        raise ValueError(msg) from e

    if loss_type in ("ps_eps", "ps_u", "ps_div"):
        physical_loss_class = pino.PINOPhysicalLossEps if loss_type == "ps_eps" else pino.PINOPhysicalLossDiv
        return physical_loss_class(
            data_loss=data_loss_fn,
            lambda_phys=lambda_phys,
            lambda_p=lambda_p,
            in_normalizer=in_normalizer,
            out_normalizer=out_normalizer,
            interior_pad=interior_pad,
        )

    if loss_type in ("sp_eps", "sp_u", "sp_div"):
        if grad_mode not in ("fft", "fft_reflect"):
            msg = f"Unknown spectral grad_mode: {grad_mode}"
            raise ValueError(msg)
        grad_mode_value = cast('Literal["fft", "fft_reflect"]', grad_mode)
        spectral_loss_class = pino.PINOSpectralLossEps if loss_type == "sp_eps" else pino.PINOSpectralLossDiv
        return spectral_loss_class(
            data_loss=data_loss_fn,
            lambda_phys=lambda_phys,
            lambda_p=lambda_p,
            in_normalizer=in_normalizer,
            out_normalizer=out_normalizer,
            grad_mode=grad_mode_value,
            interior_pad=interior_pad,
        )

    msg = "Unknown PINO loss type: {loss_type}. Options: ps_eps, ps_u, ps_div, sp_eps, sp_u, sp_div"
    raise ValueError(msg.format(loss_type=loss_type))


def _float_loss_value(loss_config: dict[str, Any], name: str, default: float) -> float:
    """Read a floating-point loss value from config."""
    return float(loss_config.get(name, default))


def _int_loss_value(loss_config: dict[str, Any], name: str, default: int) -> int:
    """Read an integer loss value from config."""
    return int(loss_config.get(name, default))


def _str_loss_value(loss_config: dict[str, Any], name: str, default: str) -> str:
    """Read a string loss value from config."""
    return str(loss_config.get(name, default))


def _require_out_normalizer(
    out_normalizer: learning.metrics.metrics.TensorNormalizer | None,
    loss_name: str,
) -> learning.metrics.metrics.TensorNormalizer:
    """Return the output normalizer required by physical-channel metrics."""
    if out_normalizer is None:
        msg = f"Evaluation loss {loss_name!r} requires an output normalizer."
        raise ValueError(msg)
    return out_normalizer


def _as_module(loss: Any) -> nn.Module:
    """Return a loss object typed as torch module for factory outputs."""
    return cast("nn.Module", loss)


def build_training_loss(config: dict[str, Any]) -> nn.Module:
    """
    Build training loss from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration with keys:
        - loss.type: "supervised" or "pino"
        - loss.*: type-specific parameters

    Returns
    -------
    nn.Module
        Training loss instance

    Raises
    ------
    ValueError
        If loss configuration is invalid

    """
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "supervised")

    if loss_type == "supervised":
        return build_supervised_loss(data_loss=_str_loss_value(loss_config, "data_loss", "h1"))
    if loss_type == "pino":
        return build_pino_loss(
            loss_type=_str_loss_value(loss_config, "loss_type", "ps_eps"),
            data_loss=_str_loss_value(loss_config, "data_loss", "h1"),
            lambda_phys=_float_loss_value(loss_config, "lambda_phys", 1e-4),
            lambda_p=_float_loss_value(loss_config, "lambda_p", 5e-4),
            grad_mode=_str_loss_value(loss_config, "grad_mode", "fft_reflect"),
            interior_pad=_int_loss_value(loss_config, "interior_pad", 2),
        )
    msg = f"Unknown loss type: {loss_type}"
    raise ValueError(msg)


def build_eval_losses(
    config: dict[str, Any],
    out_normalizer: learning.metrics.metrics.TensorNormalizer | None = None,
) -> dict[str, nn.Module]:
    """
    Build evaluation loss suite from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration with keys:
        - evaluation.losses: dict mapping loss names to enabled flags
    out_normalizer : Any | None, optional
        Output normalizer required for physical-channel RMSE metrics

    Returns
    -------
    dict[str, nn.Module]
        Dictionary of evaluation losses

    Raises
    ------
    ValueError
        If loss names are unknown

    """
    eval_config = config.get("evaluation", {})
    enabled_losses = eval_config.get("losses", {})

    losses: dict[str, nn.Module] = {}
    for loss_name, enabled in enabled_losses.items():
        if not enabled:
            continue
        if loss_name == "h1":
            losses[loss_name] = _as_module(H1Loss(d=2))
        elif loss_name == "l2":
            losses[loss_name] = _as_module(LpLoss(d=2, p=2))
        elif loss_name == "overall_rmse":
            losses[loss_name] = learning.metrics.metrics.RMSEOverall()
        elif loss_name == "rmse_p_pa":
            losses[loss_name] = learning.metrics.metrics.RMSEChannelPhysical(0, _require_out_normalizer(out_normalizer, loss_name))
        elif loss_name == "rmse_u_ms":
            losses[loss_name] = learning.metrics.metrics.RMSEChannelPhysical(1, _require_out_normalizer(out_normalizer, loss_name))
        elif loss_name == "rmse_v_ms":
            losses[loss_name] = learning.metrics.metrics.RMSEChannelPhysical(2, _require_out_normalizer(out_normalizer, loss_name))
        else:
            msg = f"Unknown evaluation loss: {loss_name}"
            raise ValueError(msg)

    return losses
