# ruff: noqa: S101
"""
Verify manufactured Darcy-Brinkman momentum, continuity, and boundary equations.

Constant and linear physical fields establish analytic residual values, both
continuity selections, per-sample outlet reduction, and name-based channel
binding. Numerical derivative backends are tested in
``test_physics_derivatives``; training weights and warmup belong to loss tests.
"""

from __future__ import annotations

import pytest
import torch
from src import domain


def _grid(height: int = 9, width: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build batched float64 coordinate planes on x in [0, 2] and y in [0, 1].

    Each field has shape ``(1, height, width)`` with ``ij`` indexing, making
    the physical grid spacing explicit for analytic residual assertions.
    """
    y_values = torch.linspace(0.0, 1.0, height, dtype=torch.float64)
    x_values = torch.linspace(0.0, 2.0, width, dtype=torch.float64)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    return x_grid.unsqueeze(0), y_grid.unsqueeze(0)


def _steady_tensors() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    """
    Build one zero-pressure, zero-velocity state in steady-flow channel order.

    Inputs contain the physical grid, zero stored permeability channels, porosity
    one-half, and a zero pressure boundary. Returned names bind the BCHW channels
    explicitly so the manufactured baseline must produce zero residuals.
    """
    x_grid, y_grid = _grid()
    zeros = torch.zeros_like(x_grid)
    inputs = torch.stack(
        (
            x_grid,
            y_grid,
            zeros,
            zeros,
            zeros,
            torch.full_like(x_grid, 0.5),
            zeros,
        ),
        dim=1,
    )
    outputs = torch.stack((zeros, zeros, zeros), dim=1)
    return (
        inputs,
        outputs,
        ("x", "y", "kxx", "kxy", "kyy", "eps", "p_bc"),
        ("p", "u", "v"),
    )


def test_zero_velocity_constant_pressure_has_zero_residual() -> None:
    """
    Evaluate a task-bound constant state with zero pressure and velocity.

    Momentum, selected continuity, and boundary reductions must all be exactly
    zero with full-grid shapes, establishing the residual baseline.
    """
    inputs, outputs, input_fields, output_fields = _steady_tensors()
    diagnostics = domain.physics.brinkman.evaluate_steady_2d_brinkman(
        inputs,
        outputs,
        input_fields=input_fields,
        output_fields=output_fields,
        derivatives=domain.physics.derivatives.PhysicalDerivatives(),
        continuity="div_eps_velocity",
        boundary="pressure_inlet_zero_pressure_outlet",
        interior_crop=1,
    )

    assert diagnostics.momentum.x.shape == outputs[:, 0].shape
    assert diagnostics.momentum.y.shape == outputs[:, 0].shape
    assert diagnostics.continuity.selected.shape == outputs[:, 0].shape
    assert diagnostics.momentum_mse.item() == pytest.approx(0.0, abs=1e-24)
    assert diagnostics.continuity_mse.item() == pytest.approx(0.0, abs=1e-24)
    assert diagnostics.boundary_mse.item() == pytest.approx(0.0, abs=1e-24)


def test_linear_state_matches_analytic_brinkman_residuals() -> None:
    """
    Evaluate Brinkman diagnostics on linear pressure and velocity fields.

    The computed momentum residuals must equal the analytic pressure-gradient,
    viscous, and permeability-drag terms, protecting signs, units, and tensor axes.
    """
    inputs, outputs, input_fields, output_fields = _steady_tensors()
    x_grid = inputs[:, input_fields.index("x")]
    y_grid = inputs[:, input_fields.index("y")]
    velocity_scale = 1e-5
    outputs[:, output_fields.index("p")] = x_grid + 2.0 * y_grid
    outputs[:, output_fields.index("u")] = velocity_scale * x_grid
    outputs[:, output_fields.index("v")] = -velocity_scale * y_grid
    diagnostics = domain.physics.brinkman.evaluate_steady_2d_brinkman(
        inputs,
        outputs,
        input_fields=input_fields,
        output_fields=output_fields,
        derivatives=domain.physics.derivatives.PhysicalDerivatives(),
        continuity="div_velocity",
        boundary="pressure_inlet_zero_pressure_outlet",
        interior_crop=1,
    )
    viscosity = domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY
    expected_x = -torch.ones_like(x_grid) - viscosity * velocity_scale * x_grid
    expected_y = -2.0 * torch.ones_like(y_grid) + viscosity * velocity_scale * y_grid

    assert torch.allclose(diagnostics.momentum.x, expected_x, atol=1e-12, rtol=1e-12)
    assert torch.allclose(diagnostics.momentum.y, expected_y, atol=1e-12, rtol=1e-12)
    assert torch.allclose(diagnostics.continuity.selected, torch.zeros_like(x_grid), atol=1e-12)


def test_both_continuity_formulations_are_semantically_selected() -> None:
    """
    Evaluate a linear velocity under constant non-unit porosity with both continuity IDs.

    Plain divergence must equal one, conservative divergence must equal porosity,
    and an unknown ID must fail so configuration selection remains scientifically explicit.
    """
    x_grid, _ = _grid()
    zeros = torch.zeros_like(x_grid)
    porosity = torch.full_like(x_grid, 0.25)
    operator = domain.physics.derivatives.PhysicalDerivatives()
    dx = torch.tensor(0.2, dtype=torch.float64)
    dy = torch.tensor(0.125, dtype=torch.float64)
    conservative = domain.physics.brinkman.continuity_residuals(
        x_grid,
        zeros,
        porosity,
        operator,
        dx,
        dy,
        kind="div_eps_velocity",
    )
    plain = domain.physics.brinkman.continuity_residuals(
        x_grid,
        zeros,
        porosity,
        operator,
        dx,
        dy,
        kind="div_velocity",
    )

    assert torch.allclose(conservative.selected, torch.full_like(x_grid, 0.25), atol=1e-12)
    assert torch.allclose(plain.selected, torch.ones_like(x_grid), atol=1e-12)
    with pytest.raises(ValueError, match="Unknown continuity identifier"):
        domain.physics.contracts.validate_continuity_kind("unsupported")


def test_outlet_pressure_gauge_is_reduced_per_sample() -> None:
    """
    Give two samples equal-magnitude, opposite-sign outlet pressure gauges.

    Per-sample means must remain distinct and their squared reduction nonzero,
    preventing cross-sample cancellation from hiding boundary error.
    """
    _, y_grid = _grid()
    y_grid = y_grid.repeat(2, 1, 1)
    pressure = torch.zeros_like(y_grid)
    outlet = y_grid == y_grid.amax(dim=(-2, -1), keepdim=True)
    pressure[0][outlet[0]] = 1.0
    pressure[1][outlet[1]] = -1.0
    residuals = domain.physics.boundary.pressure_boundary_residuals(
        pressure,
        torch.zeros_like(pressure),
        y_grid,
        0.125,
    )

    assert residuals.outlet_sample_mean.tolist() == pytest.approx([1.0, -1.0])
    assert residuals.outlet_mean_square.item() == pytest.approx(1.0)
    assert residuals.mse.item() == pytest.approx(1.0)


def test_pressure_boundary_residual_and_field_map_invariance() -> None:
    """
    Satisfy the inlet gauge, then reorder every input and output channel with its label.

    Boundary error must remain zero and all residual arrays invariant, proving
    physics binds semantic field names rather than positional conventions.
    """
    inputs, outputs, input_fields, output_fields = _steady_tensors()
    y_grid = inputs[:, input_fields.index("y")]
    inlet = y_grid == y_grid.amin()
    inputs[:, input_fields.index("p_bc")][inlet] = 3.0
    outputs[:, output_fields.index("p")][inlet] = 3.0
    reference = domain.physics.brinkman.evaluate_steady_2d_brinkman(
        inputs,
        outputs,
        input_fields=input_fields,
        output_fields=output_fields,
        derivatives=domain.physics.derivatives.PhysicalDerivatives(),
        continuity="div_eps_velocity",
        boundary="pressure_inlet_zero_pressure_outlet",
    )

    input_order = (6, 0, 5, 2, 1, 4, 3)
    output_order = (2, 0, 1)
    reordered = domain.physics.brinkman.evaluate_steady_2d_brinkman(
        inputs[:, input_order],
        outputs[:, output_order],
        input_fields=tuple(input_fields[index] for index in input_order),
        output_fields=tuple(output_fields[index] for index in output_order),
        derivatives=domain.physics.derivatives.PhysicalDerivatives(),
        continuity="div_eps_velocity",
        boundary="pressure_inlet_zero_pressure_outlet",
    )

    assert reference.boundary_mse.item() == pytest.approx(0.0, abs=1e-24)
    assert torch.allclose(reference.momentum.x, reordered.momentum.x)
    assert torch.allclose(reference.momentum.y, reordered.momentum.y)
    assert torch.allclose(reference.continuity.selected, reordered.continuity.selected)
