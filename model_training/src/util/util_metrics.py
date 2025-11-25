"""
Utility functions for numerical error and statistics metrics.

This module provides functions to:
- Convert arbitrary numeric inputs into NumPy arrays
- Compute classical error metrics (MSE, RMSE, MAE)
- Compute relative and normalized errors
- Build error maps across a sample axis (mean abs, std)
- Compute Pearson correlations between numeric arrays
- Aggregate per-sample error statistics for downstream analysis

All functions operate on NumPy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

NumberArray = NDArray[np.float64]


# ============================================================================
# Conversion utilities
# ============================================================================


def _to_numpy(array: Any, *, copy: bool = False) -> NumberArray:
    """
    Convert arbitrary numeric input into a float64 NumPy array.

    Supports NumPy arrays, PyTorch tensors and generic sequences. The output
    is always a float64 array for consistent behaviour across all metrics.

    Args:
        array: Input data (NumPy array, PyTorch tensor, list, tuple, etc.).
        copy: If True, force a copy of the underlying data.

    Returns:
        np.ndarray: Float64 NumPy array representation of the input.

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

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axis or axes along which to average. If None, average over all elements.

    Returns:
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

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axis or axes along which to average. If None, average over all elements.

    Returns:
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

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axis or axes along which to average. If None, average over all elements.

    Returns:
        np.ndarray: Mean absolute error as a float64 NumPy array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return np.mean(np.abs(yp - yt), axis=axis)


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

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axes along which to average. If None, use all elements.
        eps: Small constant added to avoid division by zero.

    Returns:
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

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axes along which to sum. If None, sum over all elements.
        eps: Small constant to avoid division by zero.

    Returns:
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

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axes along which to sum squares. If None, use all elements.
        eps: Small constant to avoid division by zero.

    Returns:
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

    Args:
        y_true: Ground truth values with a sample axis.
        y_pred: Predicted values.
        sample_axis: Axis indexing samples.

    Returns:
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

    Args:
        y_true: Ground truth values with a sample axis.
        y_pred: Predicted values.
        sample_axis: Axis indexing samples.
        ddof: Delta degrees of freedom.

    Returns:
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

    Args:
        x: First input array.
        y: Second input array.
        eps: Small constant to avoid division by zero.

    Returns:
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

    Args:
        y_true: Ground truth values with sample axis.
        y_pred: Predicted values.
        sample_axis: Axis indexing samples.
        eps: Small constant to avoid division by zero.

    Returns:
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

    This metric accepts arbitrary keyword arguments to remain compatible
    with `neuralop.Trainer`, which forwards additional keys such as
    ``x=`` or ``meta=`` during evaluation.

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

        Args:
            pred: Predicted tensor of shape (batch, C, H, W).
            y: Ground truth tensor with identical shape.
            **kwargs: Ignored extra inputs for compatibility with Trainer.

        Returns:
            torch.Tensor: Scalar RMSE value.

        """
        diff = pred - y
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

    Extra keyword arguments are ignored to maintain compatibility with
    `neuralop.Trainer`.
    """

    def __init__(self, channel: int) -> None:
        """
        Initialize the channel-wise relative RMSE metric.

        Args:
            channel: Index of the output channel to evaluate.

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

        Args:
            pred: Predicted tensor of shape (batch, C, H, W).
            y: Ground truth tensor with identical shape.
            **kwargs: Ignored additional arguments forwarded by Trainer.

        Returns:
            torch.Tensor: Scalar relative RMSE value (percent) for the channel.

        """
        yt = y[:, self.channel]
        pt = pred[:, self.channel]

        diff = pt - yt
        rmse = torch.sqrt(torch.mean(diff * diff))

        denom = torch.mean(torch.abs(yt)) + 1e-8  # avoid division by zero

        return 100.0 * rmse / denom
