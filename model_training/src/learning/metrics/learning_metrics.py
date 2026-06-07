"""
===============================================================================
learning_metrics.py
===============================================================================
Error metrics and statistical measures for model training and evaluation.

Provides:
  - classical error metrics (MSE, RMSE, MAE)
  - relative and normalized error variants
  - per-sample and aggregated error statistics
  - Pearson correlation computation
  - PyTorch nn.Module implementations for training (RMSEOverall, RMSEChannelPhysical, RelRMSEChannel)
  - error map construction across sample dimensions

Design principles:
  - all array functions operate on NumPy arrays
  - PyTorch modules are used during training for device-aware computation
  - metrics are deterministic and independent of dataset sampling
  - physical units and denormalization are explicit in the API

This module does NOT:
  - handle logging or output writing
  - contain dataset-specific metric aggregation (that belongs in analysis)
  - include plot generation or visualization
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

NumberArray: TypeAlias = NDArray[np.float64]


class TensorNormalizer(Protocol):
    """Normalizer interface required by physical-unit tensor metrics."""

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a normalized tensor back to physical units."""
        ...


# ============================================================================
# Conversion utilities
# ============================================================================


def _to_numpy(array: Any, *, copy: bool = False) -> NumberArray:
    """
    Convert arbitrary numeric input into a float64 NumPy array.

    Supports NumPy arrays, PyTorch tensors and generic sequences. The output
    is always a float64 array for consistent behaviour across all metrics.

    Parameters
    ----------
    array : Any
        Input array-like data to convert.
    copy : bool, optional
        If True, always return a copy of the data. If False, avoid copying if

    Returns
    -------
        np.ndarray: Converted array in float64 dtype.

    """
    if isinstance(array, np.ndarray):
        if array.dtype == np.float64 and not copy:
            return array
        return array.astype(np.float64, copy=copy)

    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy().astype(np.float64)

    return np.asarray(array, dtype=np.float64)


# ============================================================================
# Core error metrics
# ============================================================================


def mse(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
) -> NumberArray:
    """
    Compute the mean squared error between prediction and ground truth.

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    axis : int or tuple of ints, optional
        Axis or axes along which to average. If None, average over all elements.

    Returns
    -------
        np.ndarray: Mean squared error as a float64 NumPy array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = yp - yt
    return np.asarray(np.mean(diff * diff, axis=axis), dtype=np.float64)


def rmse(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
) -> NumberArray:
    """
    Compute the root mean squared error between prediction and ground truth.

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    axis : int or tuple of ints, optional
        Axis or axes along which to average. If None, average over all elements.


    Returns
    -------
        np.ndarray: Root mean squared error as a float64 NumPy array.

    """
    return np.sqrt(mse(y_true=y_true, y_pred=y_pred, axis=axis))


def mae(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
) -> NumberArray:
    """
    Compute the mean absolute error between prediction and ground truth.

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    axis : int or tuple of ints, optional
        Axis or axes along which to average. If None, average over all elements.


    Returns
    -------
        np.ndarray: Mean absolute error as a float64 NumPy array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return np.asarray(np.mean(np.abs(yp - yt), axis=axis), dtype=np.float64)


# ============================================================================
# Relative error metrics
# ============================================================================


def mean_relative_error(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> NumberArray:
    """
    Compute the mean absolute relative error.

    Defined elementwise as:
        |y_pred - y_true| / (|y_true| + eps)

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    axis : int or tuple of ints, optional
        Axis or axes along which to average. If None, average over all elements.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero. Default is 1e-12.

    Returns
    -------
        np.ndarray: Mean absolute relative error as float64.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    rel = np.abs(yp - yt) / (np.abs(yt) + eps)
    return np.asarray(np.mean(rel, axis=axis), dtype=np.float64)


def l1_relative_error(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> NumberArray:
    """
    Compute the L1 relative error.

    Defined as:
        ||y_pred - y_true||_1 / (||y_true||_1 + eps)

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    axis : int or tuple of ints, optional
        Axis or axes along which to sum. If None, sum over all elements.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero. Default is 1e-12.

    Returns
    -------
        np.ndarray: L1 relative error as float64.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    num = np.sum(np.abs(yp - yt), axis=axis)
    denom = np.sum(np.abs(yt), axis=axis) + eps
    return num / denom


def l2_relative_error(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> NumberArray:
    """
    Compute the L2 relative error.

    Defined as:
        ||y_pred - y_true||_2 / (||y_true||_2 + eps)

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    axis : int or tuple of ints, optional
        Axis or axes along which to sum. If None, sum over all elements.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero. Default is 1e-12.

    Returns
    -------
        np.ndarray: L2 relative error as float64.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = yp - yt
    num = np.sqrt(np.sum(diff * diff, axis=axis))
    denom = np.sqrt(np.sum(yt * yt, axis=axis)) + eps
    return num / denom


# ============================================================================
# Error maps
# ============================================================================


def mean_absolute_error_map(
    y_true: Any,
    y_pred: Any,
    sample_axis: int = 0,
) -> NumberArray:
    """
    Compute a mean absolute error map across samples.

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    sample_axis : int, optional
        Axis along which to compute the mean absolute error over samples. Default is 0.

    Returns
    -------
        np.ndarray: Mean absolute error per spatial location.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return np.mean(np.abs(yp - yt), axis=sample_axis)


def std_error_map(
    y_true: Any,
    y_pred: Any,
    sample_axis: int = 0,
    ddof: int = 0,
) -> NumberArray:
    """
    Compute a standard deviation error map across samples.

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    sample_axis : int, optional
        Axis along which to compute the standard deviation of error over samples. Default is 0.
    ddof : int, optional
        Degrees of freedom for standard deviation calculation. Default is 0 (population std). Use ddof=1 for sample std.

    Returns
    -------
        np.ndarray: Standard deviation of signed error per location.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = yp - yt
    return np.std(diff, axis=sample_axis, ddof=ddof)


# ============================================================================
# Correlation
# ============================================================================


def pearson_correlation(
    x: Any,
    y: Any,
    eps: float = 1e-12,
) -> float:
    """
    Compute the Pearson correlation coefficient between two arrays.

    Both arrays are flattened before computing the correlation.

    Parameters
    ----------
    x : Any
        First input array. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y : Any
        Second input array. Must be compatible in shape with `x`.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero. Default is 1e-

    Returns
    -------
        float: Pearson correlation in the range [-1, 1].

    """
    x_arr = _to_numpy(x).ravel()
    y_arr = _to_numpy(y).ravel()

    if x_arr.size != y_arr.size:
        msg = "Input arrays must have the same number of elements."
        raise ValueError(msg)

    x_centered = x_arr - np.mean(x_arr)
    y_centered = y_arr - np.mean(y_arr)

    num = float(np.mean(x_centered * y_centered))
    denom = float(np.sqrt(np.mean(x_centered**2)) * np.sqrt(np.mean(y_centered**2)) + eps)

    return num / denom


# ============================================================================
# Per-sample aggregated statistics
# ============================================================================


def per_sample_error_statistics(
    y_true: Any,
    y_pred: Any,
    sample_axis: int = 0,
    eps: float = 1e-12,
) -> Mapping[str, NumberArray]:
    """
    Compute a set of error statistics per sample.

    Aggregates all metrics along all non-sample axes.

    Parameters
    ----------
    y_true : Any
        Ground truth values. Can be a NumPy array, PyTorch tensor, or any array-like structure.
    y_pred : Any
        Predicted values. Must be compatible in shape with `y_true`.
    sample_axis : int, optional
        Axis along which to compute the statistics over samples. Default is 0.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero. Default is 1e-

    Returns
    -------
        Mapping[str, np.ndarray]: Metrics (shape n_samples,).

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    if sample_axis != 0:
        yt = np.moveaxis(yt, sample_axis, 0)
        yp = np.moveaxis(yp, sample_axis, 0)

    n_samples = yt.shape[0]
    yt_flat = yt.reshape(n_samples, -1)
    yp_flat = yp.reshape(n_samples, -1)
    diff_flat = yp_flat - yt_flat

    mse_vals = np.mean(diff_flat * diff_flat, axis=1)
    rmse_vals = np.sqrt(mse_vals)
    mae_vals = np.mean(np.abs(diff_flat), axis=1)

    denom_abs = np.abs(yt_flat) + eps
    mean_rel_vals = np.mean(np.abs(diff_flat) / denom_abs, axis=1)

    num_l1 = np.sum(np.abs(diff_flat), axis=1)
    denom_l1 = np.sum(np.abs(yt_flat), axis=1) + eps
    l1_rel_vals = num_l1 / denom_l1

    num_l2 = np.sqrt(np.sum(diff_flat * diff_flat, axis=1))
    denom_l2 = np.sqrt(np.sum(yt_flat * yt_flat, axis=1)) + eps
    l2_rel_vals = num_l2 / denom_l2

    return {
        "mse": mse_vals,
        "rmse": rmse_vals,
        "mae": mae_vals,
        "mean_relative_error": mean_rel_vals,
        "l1_relative_error": l1_rel_vals,
        "l2_relative_error": l2_rel_vals,
    }


# ============================================================================
# Overall RMSE (absolute)
# ============================================================================


class RMSEOverall(nn.Module):
    """
    Compute the Root Mean Squared Error (RMSE) across all channels and spatial dimensions.

    This metric accepts arbitrary keyword arguments so evaluation callers can
    forward additional keys such as ``x=`` or ``meta=``.

    The RMSE is computed as:

        RMSE = sqrt( mean( (pred - y)^2 ) )

    Both ``pred`` and ``y`` must have identical shapes.

    Returns a scalar tensor.
    """

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Compute the overall RMSE.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor of shape (batch, C, H, W).
        y : torch.Tensor
            Ground truth tensor with identical shape.
        **kwargs : torch.Tensor
            Ignored extra inputs forwarded by evaluation callers.

        Returns
        -------
        torch.Tensor
            Scalar RMSE value.

        """
        diff = pred - y
        return torch.sqrt(torch.mean(diff * diff))


# ============================================================================
# Channel-wise RMSE in PHYSICAL units
# ============================================================================


class RMSEChannelPhysical(nn.Module):
    """
    Compute the RMSE for a specific output channel in physical units.

    This metric denormalizes both predictions and targets using the provided
    normalizer before computing the RMSE for the selected channel.

    Parameters
    ----------
        channel: Index of the output channel to evaluate.
        out_normalizer: Normalizer with an `inverse_transform` method
                        to denormalize model outputs.

    Returns
    -------
        torch.Tensor: Scalar RMSE value for the specified channel.

    """

    def __init__(self, channel: int, out_normalizer: TensorNormalizer) -> None:
        """
        Initialize the channel-wise physical RMSE metric.

        Parameters
        ----------
        channel : int
            Index of the output channel to evaluate.
        out_normalizer
            Normalizer with an `inverse_transform` method to denormalize model outputs.

        """
        super().__init__()
        self.channel = channel
        self.out_normalizer = out_normalizer

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Compute the RMSE for the selected channel in physical units.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor of shape (batch, C, H, W).
        y : torch.Tensor
            Ground truth tensor with identical shape.
        **kwargs : torch.Tensor
            Ignored additional arguments forwarded by evaluation callers.

        Returns
        -------
        torch.Tensor
            Scalar RMSE value for the channel.

        """
        # Denormalize predictions and targets
        pred_phys = self.out_normalizer.inverse_transform(pred)
        y_phys = self.out_normalizer.inverse_transform(y)

        diff = pred_phys[:, self.channel] - y_phys[:, self.channel]
        return torch.sqrt(torch.mean(diff * diff))


# ============================================================================
# Channel-wise relative RMSE (percent)
# ============================================================================


class RelRMSEChannel(nn.Module):
    """
    Compute the relative RMSE (in percent) for a specific output channel.

    This metric is physically interpretable and allows direct comparison
    across channels with different numerical scales (for example pressure
    vs velocity). It is defined as:

        rel_RMSE = 100 * RMSE / mean(|y|)

    Extra keyword arguments are ignored so callers may forward batch metadata.
    """

    def __init__(self, channel: int) -> None:
        """
        Initialize the channel-wise relative RMSE metric.

        Parameters
        ----------
        channel : int
            Index of the output channel to evaluate.

        """
        super().__init__()
        self.channel = channel

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Compute the relative RMSE for the selected channel.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor of shape (batch, C, H, W).
        y : torch.Tensor
            Ground truth tensor with identical shape.
        **kwargs : torch.Tensor
            Ignored additional arguments forwarded by evaluation callers.

        Returns
        -------
        torch.Tensor
            Scalar relative RMSE value (percent) for the channel.

        """
        yt = y[:, self.channel]
        pt = pred[:, self.channel]

        diff = pt - yt
        rmse = torch.sqrt(torch.mean(diff * diff))

        denom = torch.mean(torch.abs(yt)) + 1e-8  # avoid division by zero

        return 100.0 * rmse / denom
