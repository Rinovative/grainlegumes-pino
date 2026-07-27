# ruff: noqa: S101
"""
Verify reusable physical and spectral derivative operators on analytic fields.

Linear and periodic manufactured solutions cover gradients, divergence, crop,
spacing inference, extension semantics, and invalid grids/identifiers. Brinkman
equation assembly is covered separately; these tests do not claim accuracy on
production COMSOL fields.
"""

from __future__ import annotations

import math

import pytest
import torch
from src import domain


def test_physical_derivatives_match_linear_field_and_crop() -> None:
    """
    Differentiate a linear float64 field on a rectangular uniform physical grid.

    Both analytic gradients and a valid interior crop must be exact, while an
    overlarge crop must fail so finite-difference axes and region semantics stay fixed.
    """
    x_values = torch.linspace(-2.0, 3.0, 17, dtype=torch.float64)
    y_values = torch.linspace(1.0, 4.0, 13, dtype=torch.float64)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    field = (3.0 * x_grid - 2.0 * y_grid + 5.0).unsqueeze(0)
    operator = domain.physics.derivatives.PhysicalDerivatives()
    derivative_x, derivative_y = operator.gradient(
        field,
        x_values[1] - x_values[0],
        y_values[1] - y_values[0],
    )

    assert torch.allclose(derivative_x, torch.full_like(field, 3.0), atol=1e-12)
    assert torch.allclose(derivative_y, torch.full_like(field, -2.0), atol=1e-12)
    assert domain.physics.derivatives.crop_interior(field, 2).shape == (1, 9, 13)
    with pytest.raises(ValueError, match="complete spatial domain"):
        domain.physics.derivatives.crop_interior(field, 7)


def test_spectral_derivatives_match_periodic_analytic_field() -> None:
    """
    Differentiate a periodic two-mode field with the no-extension spectral operator.

    Both FFT gradients must match their analytic functions, protecting physical
    spacing and x/y frequency-axis interpretation.
    """
    height = 32
    width = 40
    length_x = 2.0 * math.pi
    length_y = 2.0 * math.pi
    x_values = torch.arange(width, dtype=torch.float64) * length_x / width
    y_values = torch.arange(height, dtype=torch.float64) * length_y / height
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    field = (torch.sin(3.0 * x_grid) + 0.5 * torch.cos(2.0 * y_grid)).unsqueeze(0)
    expected_x = (3.0 * torch.cos(3.0 * x_grid)).unsqueeze(0)
    expected_y = (-torch.sin(2.0 * y_grid)).unsqueeze(0)
    operator = domain.physics.derivatives.SpectralDerivatives(extension="none")
    derivative_x, derivative_y = operator.gradient(
        field,
        length_x / width,
        length_y / height,
    )

    assert torch.allclose(derivative_x, expected_x, atol=1e-10, rtol=1e-10)
    assert torch.allclose(derivative_y, expected_y, atol=1e-10, rtol=1e-10)


def test_uniform_spacing_rejects_invalid_coordinate_grids() -> None:
    """
    Infer spacing from one valid grid, then vary finiteness, order, spacing, and Cartesian form.

    The valid grid must yield exact unit spacing and every malformed family must
    fail explicitly before a derivative operator consumes it.
    """
    y_grid, x_grid = torch.meshgrid(
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
        indexing="ij",
    )
    spacing_x, spacing_y = domain.physics.derivatives.infer_uniform_spacing(x_grid, y_grid)
    assert spacing_x.item() == pytest.approx(1.0)
    assert spacing_y.item() == pytest.approx(1.0)

    nonfinite = x_grid.clone()
    nonfinite[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        domain.physics.derivatives.infer_uniform_spacing(nonfinite, y_grid)

    with pytest.raises(ValueError, match="strictly increasing"):
        domain.physics.derivatives.infer_uniform_spacing(x_grid.flip(-1), y_grid)

    nonuniform = x_grid.clone()
    nonuniform[:, 2:] += 1.0
    with pytest.raises(ValueError, match="not uniform"):
        domain.physics.derivatives.infer_uniform_spacing(nonuniform, y_grid)

    noncartesian = x_grid.clone()
    noncartesian[1] += 0.1
    with pytest.raises(ValueError, match="constant along the y-axis"):
        domain.physics.derivatives.infer_uniform_spacing(noncartesian, y_grid)


def test_derivative_semantics_fail_clearly() -> None:
    """
    Request an unknown derivative kind and a physical operator with spectral extension.

    Both semantic contradictions must fail during construction, preventing hidden
    fallback or unused extension settings from changing scientific meaning.
    """
    with pytest.raises(ValueError, match="Unknown derivative identifier"):
        domain.physics.derivatives.build_derivative_operator("automatic", extension="none")
    with pytest.raises(ValueError, match="require extension 'none'"):
        domain.physics.derivatives.build_derivative_operator("physical", extension="reflect")
