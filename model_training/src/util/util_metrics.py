"""
Utility functions for numerical error and statistics metrics.

This module provides functions to:
- Convert arbitrary numeric inputs into NumPy arrays
- Compute classical error metrics (MSE, RMSE, MAE)
- Compute relative and normalized errors
- Build error maps across a sample axis (mean abs, std)
- Compute Pearson correlations between numeric arrays
- Aggregate per-sample error statistics for downstream analysis

All functions operate on NumPy arrays
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Mapping

NumberArray = NDArray[np.float64]


def _to_numpy(array: Any, *, copy: bool = False) -> NumberArray:
    """
    Convert arbitrary numeric input into a float64 NumPy array.

    Supports NumPy arrays, PyTorch tensors and generic sequences. The
    output is always a float64 array, which ensures consistent behavior
    across all metrics.

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
        axis: Axis or axes along which to average. If None, average over
            all elements.

    Returns:
        np.ndarray: Mean squared error. A scalar is returned as a 0D array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = yp - yt
    out = np.mean(diff * diff, axis=axis)
    return np.asarray(out, dtype=np.float64)


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
        axis: Axis or axes along which to average. If None, average over
            all elements.

    Returns:
        np.ndarray: Root mean squared error. A scalar is returned as a 0D array.

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
        axis: Axis or axes along which to average. If None, average over
            all elements.

    Returns:
        np.ndarray: Mean absolute error. A scalar is returned as a 0D array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = np.abs(yp - yt)
    return np.mean(diff, axis=axis)


# ============================================================================
# Relative and normalized errors
# ============================================================================


def mean_relative_error(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> NumberArray:
    """
    Compute the mean absolute relative error between prediction and ground truth.

    The relative error is defined elementwise as:
        |y_pred - y_true| / (|y_true| + eps)

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axis or axes along which to average. If None, average over
            all elements.
        eps: Small constant added to the denominator to avoid division by zero.

    Returns:
        np.ndarray: Mean absolute relative error. A scalar is returned as
            a 0D array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = np.abs(yp - yt)
    denom = np.abs(yt) + eps
    rel = diff / denom

    out = np.mean(rel, axis=axis)
    return np.asarray(out, dtype=np.float64)


def l1_relative_error(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> NumberArray:
    """
    Compute the L1 relative error between prediction and ground truth.

    The L1 relative error is defined as:
        ||y_pred - y_true||_1 / (||y_true||_1 + eps)

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axis or axes along which to sum. If None, use all elements.
        eps: Small constant added to the denominator to avoid division by zero.

    Returns:
        np.ndarray: L1 relative error. A scalar is returned as a 0D array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = np.abs(yp - yt)

    num = np.sum(diff, axis=axis)
    denom = np.sum(np.abs(yt), axis=axis) + eps
    return num / denom


def l2_relative_error(
    y_true: Any,
    y_pred: Any,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> NumberArray:
    """
    Compute the L2 relative error between prediction and ground truth.

    The L2 relative error is defined as:
        ||y_pred - y_true||_2 / (||y_true||_2 + eps)

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        axis: Axis or axes along which to sum squares. If None, use all
            elements.
        eps: Small constant added to the denominator to avoid division by zero.

    Returns:
        np.ndarray: L2 relative error. A scalar is returned as a 0D array.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = yp - yt

    num = np.sqrt(np.sum(diff * diff, axis=axis))
    denom = np.sqrt(np.sum(yt * yt, axis=axis)) + eps
    return num / denom


# ============================================================================
# Error maps across samples
# ============================================================================


def mean_absolute_error_map(
    y_true: Any,
    y_pred: Any,
    sample_axis: int = 0,
) -> NumberArray:
    """
    Compute a mean absolute error map across a sample axis.

    This function is intended for spatial error maps where multiple samples
    share the same spatial grid. It computes the mean absolute error per
    grid location across the specified sample axis.

    Args:
        y_true: Ground truth values with a dedicated sample axis.
        y_pred: Predicted values with the same shape as y_true.
        sample_axis: Axis that indexes samples (usually 0).

    Returns:
        np.ndarray: Mean absolute error map with the sample axis removed.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = np.abs(yp - yt)
    return np.mean(diff, axis=sample_axis)


def std_error_map(
    y_true: Any,
    y_pred: Any,
    sample_axis: int = 0,
    ddof: int = 0,
) -> NumberArray:
    """
    Compute a standard deviation error map across a sample axis.

    The map is based on the signed error (y_pred - y_true) at each grid
    location and computes the standard deviation across samples.

    Args:
        y_true: Ground truth values with a dedicated sample axis.
        y_pred: Predicted values with the same shape as y_true.
        sample_axis: Axis that indexes samples (usually 0).
        ddof: Delta degrees of freedom passed to np.std.

    Returns:
        np.ndarray: Standard deviation error map with the sample axis removed.

    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    diff = yp - yt
    return np.std(diff, axis=sample_axis, ddof=ddof)


# ============================================================================
# Correlation utilities
# ============================================================================


def pearson_correlation(
    x: Any,
    y: Any,
    eps: float = 1e-12,
) -> float:
    """
    Compute the Pearson correlation coefficient between two arrays.

    Both inputs are flattened before computing the correlation. This
    function is intended as a generic building block, for example for
    correlations between error and ground truth or between error and
    derived quantities like velocity magnitude.

    Args:
        x: First array.
        y: Second array with the same number of elements as x.
        eps: Small constant added to the denominator to avoid division by zero.

    Returns:
        float: Pearson correlation coefficient in the range [-1, 1].

    Raises:
        ValueError: If x and y do not have the same number of elements.

    """
    x_arr = _to_numpy(x).ravel()
    y_arr = _to_numpy(y).ravel()

    if x_arr.size != y_arr.size:
        msg = "pearson_correlation requires x and y to have the same number of elements."
        raise ValueError(msg)

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))

    x_centered = x_arr - x_mean
    y_centered = y_arr - y_mean

    num = float(np.mean(x_centered * y_centered))
    x_std = float(np.sqrt(np.mean(x_centered * x_centered)))
    y_std = float(np.sqrt(np.mean(y_centered * y_centered)))

    denom = x_std * y_std + eps
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
    Compute a consistent set of error statistics per sample.

    The function treats the specified axis as the sample axis and aggregates
    over all remaining dimensions. It returns one-dimensional arrays of
    metrics indexed by sample. The following metrics are provided:

    - "mse": Mean squared error per sample
    - "rmse": Root mean squared error per sample
    - "mae": Mean absolute error per sample
    - "mean_relative_error": Mean absolute relative error per sample
    - "l1_relative_error": L1 relative error per sample
    - "l2_relative_error": L2 relative error per sample

    Args:
        y_true: Ground truth values with a dedicated sample axis.
        y_pred: Predicted values with the same shape as y_true.
        sample_axis: Axis that indexes samples (default 0).
        eps: Small constant used in relative error computations.

    Returns:
        Mapping[str, np.ndarray]: Dictionary mapping metric names to
            one-dimensional arrays of shape (n_samples,).

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

    # Core metrics
    mse_vals = np.mean(diff_flat * diff_flat, axis=1)
    rmse_vals = np.sqrt(mse_vals)
    mae_vals = np.mean(np.abs(diff_flat), axis=1)

    # Mean absolute relative error (MAPE style)
    denom_abs = np.abs(yt_flat) + eps
    mean_rel_vals = np.mean(np.abs(diff_flat) / denom_abs, axis=1)

    # L1 and L2 relative errors
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
