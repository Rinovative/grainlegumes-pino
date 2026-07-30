"""
===============================================================================
evaluation_plot_physical_consistency.py
===============================================================================
Compare steady-flow momentum, dual-continuity, and pressure-boundary evidence.

Responsibilities:
  - Plot scalar residual distributions from current artifact tables
  - Render full-grid ``Rx``, ``Ry``, ``div_u``, and ``div_eps_u`` fields
  - Keep inlet MSE and squared per-sample outlet-mean pressure distinguishable
  - Annotate, but never substitute, the continuity selected during training

Design principles:
  - Scalar gradient diagnostics use the provenance-declared interior crop
  - Spatial residual arrays remain full-grid and retain their equation labels
  - ``div_velocity`` and ``div_eps_velocity`` are always reported independently
  - Pressure quantities remain in pascals or squared pascals as labeled

This module does NOT:
  - Recompute residual equations, derivative operators, or boundary masks
  - Admit artifact schemas or substitute one continuity diagnostic for another
===============================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_RESIDUAL_LABELS = {
    "momentum_residual_mse": (r"momentum mean($R_x^2+R_y^2$)", r"$(Pa/m)^2$"),
    "div_velocity_mse": (r"mean($div(u)^2$)", r"$s^{-2}$"),
    "div_eps_velocity_mse": (r"mean($div(eps u)^2$)", r"$s^{-2}$"),
    "pressure_boundary_mse": ("inlet MSE + squared outlet mean", r"$Pa^2$"),
}


def _cdf(axis: Axes, values: np.ndarray, *, label: str) -> None:
    """Draw one empirical CDF from finite artifact scalars."""
    numeric = np.asarray(values, dtype=float)
    if numeric.size == 0 or not np.isfinite(numeric).all() or (numeric < 0.0).any():
        msg = f"Residual CDF values for {label!r} must be finite and non-negative."
        raise ValueError(msg)
    ordered = np.sort(numeric)
    probability = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    axis.step(ordered, probability, where="post", label=label)


def _physics_provenance(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Return validated steady-flow physics provenance."""
    provenance = dataframe.require_complete_provenance(frame)
    physics = provenance.get("physics")
    if not isinstance(physics, Mapping):
        msg = "Steady-flow plots require physics provenance."
        raise dataframe.ComparisonCompatibilityError(msg)
    return physics


def plot_residual_distributions(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot canonical steady-flow scalar residual distributions.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible steady-flow frames with residual schema 1.

    Returns
    -------
    matplotlib.figure.Figure
        Separate empirical CDF axes for momentum, ``div(u)``, ``div(eps*u)``,
        and pressure-boundary diagnostics with roles, counts, and selected
        training continuity disclosed.

    Raises
    ------
    ComparisonCompatibilityError, TypeError, ValueError
        If physics provenance/formulas are incompatible or scalar values are not
        finite and non-negative.

    Notes
    -----
    Both continuity formulations are always shown; training selection changes
    annotation only. Scalar crop and pressure-mask regions come from provenance.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, metric in zip(axes.flat, dataframe.STEADY_PHYSICS_METRICS, strict=True):
        formula, unit = _RESIDUAL_LABELS[metric]
        for label, frame in datasets.items():
            physics = _physics_provenance(frame)
            continuity = physics["selected_training_continuity"]
            values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=float)
            _cdf(
                axis,
                values,
                label=f"{label} ({dataframe.dataset_role(frame)}); n={len(frame)}; trained={continuity}",
            )
        axis.set_title(metric)
        axis.set_xlabel(f"{formula} [{unit}]")
        axis.set_ylabel("empirical cumulative probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    first_physics = _physics_provenance(next(iter(datasets.values())))
    figure.suptitle(f"Canonical physical residual distributions; scalar crop={first_physics['interior_crop']} cells; pressure uses full-grid masks")
    return figure


def _mean_residual_maps(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, cases.EvaluationCase]:
    """
    Aggregate three full-grid residual magnitudes across complete membership.

    Momentum is the casewise ``sqrt(Rx**2 + Ry**2)`` magnitude; continuity maps
    are ``abs(div_u)`` and ``abs(div_eps_u)``. All cases must expose identical
    coordinates, shapes, and the complete four-array residual contract.
    """
    loaded = list(cases.iter_cases(frame))
    first = loaded[0]
    required = {"Rx", "Ry", "div_u", "div_eps_u"}
    if set(first.residuals) != required:
        msg = "Steady-flow residual maps require Rx, Ry, div_u, and div_eps_u."
        raise dataframe.ComparisonCompatibilityError(msg)
    for case in loaded[1:]:
        if case.shape != first.shape or not np.allclose(case.coordinates, first.coordinates) or set(case.residuals) != required:
            msg = "Spatial residual aggregation requires identical full grids and array semantics."
            raise dataframe.ComparisonCompatibilityError(msg)
    momentum = np.mean(
        np.stack([np.sqrt(case.residuals["Rx"] ** 2 + case.residuals["Ry"] ** 2) for case in loaded]),
        axis=0,
    )
    div_velocity = np.mean(
        np.stack([np.abs(case.residuals["div_u"]) for case in loaded]),
        axis=0,
    )
    div_eps_velocity = np.mean(
        np.stack([np.abs(case.residuals["div_eps_u"]) for case in loaded]),
        axis=0,
    )
    return momentum, div_velocity, div_eps_velocity, first


def _image(
    axis: Axes,
    values: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    title: str,
    unit: str,
    coordinate_units: tuple[str, str],
) -> None:
    """Render one non-negative full-grid residual map."""
    limit = float(max(np.quantile(values, 0.99), np.finfo(float).eps))
    image = axis.imshow(
        np.clip(values, 0.0, limit),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=limit,
    )
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(f"x [{coordinate_units[0]}]")
    axis.set_ylabel(f"y [{coordinate_units[1]}]")
    axis.figure.colorbar(image, ax=axis, label=unit, fraction=0.046)


def plot_spatial_residuals(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot full-grid mean momentum and both continuity residual magnitudes.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible steady-flow frames whose cases share full-grid coordinates.

    Returns
    -------
    matplotlib.figure.Figure
        One row per frame with mean momentum magnitude, mean ``abs(div(u))``, and
        mean ``abs(div(eps*u))`` in their declared physical units.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If physics provenance, NPZ residual arrays, or within-frame grids fail.

    Notes
    -----
    Maps use complete saved membership and full-grid arrays; they do not apply the
    scalar diagnostic's provenance-declared interior crop.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    figure, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(14, max(3.8 * len(datasets), 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (label, frame) in enumerate(datasets.items()):
        momentum, div_velocity, div_eps_velocity, first = _mean_residual_maps(frame)
        extent = cases.grid_extent(first)
        prefix = f"{label} ({dataframe.dataset_role(frame)}), n={len(frame)}"
        _image(
            axes[row_index, 0],
            momentum,
            extent=extent,
            title=f"{prefix}\nmean momentum magnitude",
            unit="Pa/m",
            coordinate_units=first.coordinate_units,
        )
        _image(
            axes[row_index, 1],
            div_velocity,
            extent=extent,
            title=f"{prefix}\nmean abs(div(u))",
            unit="1/s",
            coordinate_units=first.coordinate_units,
        )
        _image(
            axes[row_index, 2],
            div_eps_velocity,
            extent=extent,
            title=f"{prefix}\nmean abs(div(eps*u))",
            unit="1/s",
            coordinate_units=first.coordinate_units,
        )
    figure.suptitle("Full-grid spatial residual diagnostics")
    return figure


def _pressure_drop(case: cases.EvaluationCase) -> tuple[float, float, float]:
    """
    Compute declared and predicted pressure drop plus absolute mismatch in Pa.

    Inlet/outlet masks are the y-coordinate minima/maxima within half a median
    grid spacing. The declared drop comes from ``p_bc`` and the predicted drop
    from learned pressure; missing fields or unresolved masks fail explicitly.
    """
    if "p" not in case.fields or case.pressure_boundary is None:
        msg = "Pressure-drop analysis requires output field p and p_bc."
        raise dataframe.ComparisonCompatibilityError(msg)
    pressure = case.prediction[case.fields.index("p")]
    boundary = case.pressure_boundary[0]
    y_values = case.coordinates[1]
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    spacing = cases.grid_spacing(case)[1]
    inlet = np.isclose(y_values, y_min, rtol=0.0, atol=0.51 * spacing)
    outlet = np.isclose(y_values, y_max, rtol=0.0, atol=0.51 * spacing)
    if not inlet.any() or not outlet.any():
        msg = "Pressure-drop analysis could not resolve inlet/outlet coordinate masks."
        raise ValueError(msg)
    declared = float(np.mean(boundary[inlet]) - np.mean(boundary[outlet]))
    predicted = float(np.mean(pressure[inlet]) - np.mean(pressure[outlet]))
    return declared, predicted, abs(predicted - declared)


def build_pressure_boundary_summary(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build pressure-boundary components and independently calculated drop error.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible steady-flow frames with pressure fields and boundary inputs.

    Returns
    -------
    pandas.DataFrame
        One row per labelled frame containing role/count, medians of the three
        declared Pa-squared diagnostics, and median absolute pressure-drop error
        in pascals.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If physics admission, case pressure context, or coordinate masks fail.

    Notes
    -----
    ``pressure_outlet_mean_square`` is each sample's squared outlet-mean pressure,
    not pointwise outlet MSE. Drop error is calculated separately from NPZ fields.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        drops = [_pressure_drop(case) for case in cases.iter_cases(frame)]
        rows.append(
            {
                "label": label,
                "dataset_role": dataframe.dataset_role(frame),
                "sample_count": len(frame),
                "pressure_inlet_mse_median [Pa^2]": float(np.median(frame["pressure_inlet_mse"])),
                "pressure_outlet_mean_square_median [Pa^2]": float(np.median(frame["pressure_outlet_mean_square"])),
                "pressure_boundary_mse_median [Pa^2]": float(np.median(frame["pressure_boundary_mse"])),
                "pressure_drop_absolute_error_median [Pa]": float(np.median([item[2] for item in drops])),
            }
        )
    return pd.DataFrame(rows).set_index("label")


def plot_pressure_boundary_summary(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot declared pressure-boundary diagnostics and pressure-drop consistency.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Compatible steady-flow frames with pressure and boundary-input fields.

    Returns
    -------
    matplotlib.figure.Figure
        Median Pa-squared boundary components beside per-case predicted versus
        boundary-input pressure drops in pascals.

    Raises
    ------
    ComparisonCompatibilityError, FileNotFoundError, TypeError, ValueError
        If physics provenance, scalar columns, pressure cases, or boundary masks
        are invalid.

    Notes
    -----
    The identity line applies only to the drop panel; boundary components retain
    their distinct declared meanings and are not combined into a ranking score.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    boundary_metrics = dataframe.PRESSURE_BOUNDARY_METRICS
    x_positions = np.arange(len(boundary_metrics), dtype=float)
    width = 0.8 / max(len(datasets), 1)
    for index, (label, frame) in enumerate(datasets.items()):
        values = [float(np.median(pd.to_numeric(frame[metric], errors="raise"))) for metric in boundary_metrics]
        axes[0].bar(
            x_positions + (index - (len(datasets) - 1) / 2) * width,
            values,
            width=width,
            label=f"{label} ({dataframe.dataset_role(frame)}), n={len(frame)}",
        )
        declared = []
        predicted = []
        for case in cases.iter_cases(frame):
            declared_value, predicted_value, _error = _pressure_drop(case)
            declared.append(declared_value)
            predicted.append(predicted_value)
        axes[1].scatter(declared, predicted, s=24, alpha=0.75, label=f"{label} ({dataframe.dataset_role(frame)}), n={len(frame)}")

    axes[0].set_xticks(x_positions, ["inlet mismatch", "squared outlet mean", "total declared boundary"])
    axes[0].set_ylabel("median diagnostic [Pa^2]")
    axes[0].set_title("Pressure-boundary components")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    limits = [
        value for collection in axes[1].collections for value in np.concatenate((collection.get_offsets()[:, 0], collection.get_offsets()[:, 1]))
    ]
    low, high = min(limits), max(limits)
    axes[1].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("boundary-input pressure drop [Pa]")
    axes[1].set_ylabel("predicted pressure drop [Pa]")
    axes[1].set_title("Pressure-drop consistency")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Pressure boundary and pressure-drop diagnostics")
    return figure
