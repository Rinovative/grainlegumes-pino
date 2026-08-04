# ruff: noqa: EM101, PLR2004, S101, TRY003
"""
Protect semantic supervised/physics loss composition and deterministic warmup.

Manufactured tensors cover named weighted components, selected continuity,
disabled-physics zeros, physical inverse normalization, invalid factory settings,
sample-weighted epoch telemetry, and monitor diagnostics. Domain equation values
and evaluation metric reductions are verified in separate modules.
"""

from __future__ import annotations

import pytest
import torch
from src import domain, experiments, learning
from support import configs
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

_MIDPOINT_EPOCH = 2
_TASK = domain.tasks.registry.get_task("steady_flow")


class _ListMappingDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Present caller-supplied tensor mappings to a DataLoader in their original order.

    The fake owns no storage, normalization, or task semantics. Tests choose batch
    sizes over its finite samples to expose partial-batch aggregation behavior.
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class IdentityNormalizer:
    """Return tensors unchanged for synthetic physical-space tests."""

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged."""
        return tensor


def _physics_config(
    *,
    model_kind: str = "fno",
    continuity: str = "div_eps_velocity",
) -> dict[str, object]:
    """
    Resolve a PI recipe with one continuity, four-epoch linear warmups, and crop one.

    Residual and boundary targets are fixed at two and three respectively, making
    their midpoint weights observable while retaining the selected FNO or UNO recipe.
    """
    raw = configs.direct_config(
        model_kind=model_kind,
        physics_enabled=True,
    )
    raw["loss"]["physics"]["continuity"] = continuity  # type: ignore[index]
    raw["loss"]["physics"]["residual_weight"] = {  # type: ignore[index]
        "target": 2.0,
        "warmup": {"kind": "linear", "epochs": 4},
    }
    raw["loss"]["physics"]["boundary_weight"] = {  # type: ignore[index]
        "target": 3.0,
        "warmup": {"kind": "linear", "epochs": 4},
    }
    raw["loss"]["physics"]["interior_crop"] = 1  # type: ignore[index]
    return experiments.config.loader.resolve_config(raw)


def _manufactured_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one TaskSpec-ordered manufactured steady-flow batch."""
    height = 9
    width = 11
    y_values = torch.linspace(0.0, 1.0, height)
    x_values = torch.linspace(0.0, 2.0, width)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    zeros = torch.zeros_like(x_grid)
    p_bc = zeros.clone()
    p_bc[:, 0, :] = 1.0

    coordinate_values = iter((x_grid, y_grid))
    input_by_field: dict[str, torch.Tensor] = {}
    for field in _TASK.inputs:
        if field.role == "coordinate":
            input_by_field[field.name] = next(coordinate_values)
        elif field.role == "permeability":
            input_by_field[field.name] = zeros
        elif field.role == "porosity":
            input_by_field[field.name] = torch.full_like(x_grid, 0.5)
        elif field.role == "boundary":
            input_by_field[field.name] = p_bc
        else:
            msg = f"unsupported steady-flow fixture role: {field.role}"
            raise AssertionError(msg)
    inputs = torch.stack([input_by_field[field] for field in _TASK.input_names], dim=1)

    velocity = next(group for group in _TASK.output_groups if group.id == "velocity")
    prediction_by_field = dict.fromkeys(_TASK.output_names, zeros)
    prediction_by_field[velocity.fields[0]] = 1e-5 * x_grid
    prediction = torch.stack([prediction_by_field[field] for field in _TASK.output_names], dim=1)
    target = torch.zeros_like(prediction)
    return inputs, prediction, target


def test_disabled_physics_exposes_named_zero_components() -> None:
    """
    Compute a supervised loss from nonzero predictions with physics disabled.

    Named physics components must remain present as exact zeros and total must
    equal data loss, keeping telemetry schema stable without fabricating residuals.
    """
    supervised = experiments.config.loader.resolve_config(
        configs.direct_config(model_kind="fno", physics_enabled=False),
    )
    supervised_loss = learning.losses.factory.build_training_loss(supervised, device=torch.device("cpu"))
    task = domain.tasks.registry.get_task(str(supervised["task"]))
    pred = torch.ones((2, task.out_channels, 8, 8))
    target = torch.zeros_like(pred)
    components = supervised_loss.compute_components(pred, x=None, y=target)

    assert tuple(components) == (
        "total",
        "data",
        "momentum",
        "boundary",
        "continuity_div_eps_velocity",
    )
    assert tuple(components) == supervised_loss.component_names
    assert components["total"] == components["data"]
    assert components["momentum"].item() == 0.0
    assert components["boundary"].item() == 0.0
    assert components["continuity_div_eps_velocity"].item() == 0.0
    assert supervised_loss.telemetry_state() == {}


def test_physics_composition_weighting_and_linear_warmup() -> None:
    """
    Evaluate one manufactured PI batch at warmup epochs zero, two, and four.

    Applied weights and named components must scale linearly and sum exactly,
    while explicit epoch state persists for deterministic resume.
    """
    loss = learning.losses.factory.build_training_loss(_physics_config(), device=torch.device("cpu"))
    normalizer = IdentityNormalizer()
    loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
    inputs, pred, target = _manufactured_batch()

    start = loss.compute_components(pred, x=inputs, y=target, epoch=0)
    midpoint = loss.compute_components(pred, x=inputs, y=target, epoch=2)
    end = loss.compute_components(pred, x=inputs, y=target, epoch=4)
    continuity_name = loss.continuity_component_name

    assert continuity_name == "continuity_div_eps_velocity"
    assert start["momentum"].item() == 0.0
    assert start["boundary"].item() == 0.0
    assert start[continuity_name].item() == 0.0
    assert midpoint["momentum"] == pytest.approx(0.5 * end["momentum"])
    assert midpoint["boundary"] == pytest.approx(0.5 * end["boundary"])
    assert midpoint[continuity_name] == pytest.approx(0.5 * end[continuity_name])
    for components in (start, midpoint, end):
        assert components["total"] == pytest.approx(
            components["data"] + components["momentum"] + components["boundary"] + components[continuity_name]
        )
    assert loss.component_weights(epoch=2) == {
        "data": 1.0,
        "momentum": 1.0,
        "boundary": 1.5,
        continuity_name: 1.0,
    }
    assert loss.telemetry_state(epoch=2) == {
        "weight_physics": 1.0,
        "weight_boundary": 1.5,
        "warmup_physics_fraction": 0.5,
        "warmup_boundary_fraction": 0.5,
    }

    loss.set_epoch(_MIDPOINT_EPOCH)
    forward_total = loss(pred, x=inputs, y=target)
    assert forward_total == loss.last_components["total"]
    assert forward_total == pytest.approx(midpoint["total"])
    assert int(loss.state_dict()["current_epoch"].item()) == _MIDPOINT_EPOCH
    with pytest.raises(ValueError, match="non-negative integer"):
        loss.set_epoch(-1)


def test_pi_model_families_share_formulation_selected_loss_path() -> None:
    """
    Cross both PI model families with both allowed continuity formulations.

    Model kind may vary, but the semantic loss class and selected residual mapping
    stay fixed. The opposite continuity key must never enter optimization components.
    """
    for model_kind in ("fno", "uno"):
        for continuity in ("div_velocity", "div_eps_velocity"):
            config = _physics_config(model_kind=model_kind, continuity=continuity)
            loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"))
            normalizer = IdentityNormalizer()
            loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
            inputs, pred, target = _manufactured_batch()

            diagnostics = loss.compute_physics_diagnostics(pred, x=inputs)
            components = loss.compute_components(pred, x=inputs, y=target, epoch=4)
            continuity_name = f"continuity_{continuity}"
            selected = (
                diagnostics.continuity.divergence_velocity if continuity == "div_velocity" else diagnostics.continuity.divergence_porosity_velocity
            )
            selected_interior = selected[..., 1:-1, 1:-1]
            expected = loss.component_weights(epoch=4)[continuity_name] * selected_interior.square().mean()
            opposite = "continuity_div_eps_velocity" if continuity == "div_velocity" else "continuity_div_velocity"

            assert config["model"]["kind"] == model_kind  # type: ignore[index]
            assert type(loss) is learning.losses.pino.SemanticComposedLoss
            assert loss.continuity == continuity
            assert loss.component_names == tuple(components)
            assert torch.allclose(diagnostics.continuity.selected, selected)
            assert torch.allclose(components[continuity_name], expected)
            assert opposite not in components


def test_invalid_physics_loss_settings_fail_through_public_factory() -> None:
    """
    Replace one valid derivative kind and one warmup kind with unsupported values.

    The public loss factory must reject each semantic setting before training,
    preventing implicit numerical or schedule fallbacks.
    """
    derivative_config = _physics_config()
    derivative_config["loss"]["physics"]["derivatives"]["kind"] = "unsupported"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown derivative identifier"):
        learning.losses.factory.build_training_loss(derivative_config, device=torch.device("cpu"))

    warmup_config = _physics_config()
    warmup_config["loss"]["physics"]["residual_weight"]["warmup"]["kind"] = "cosine"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown warmup identifier"):
        learning.losses.factory.build_training_loss(warmup_config, device=torch.device("cpu"))


class _IdentityScaleModel(torch.nn.Module):
    """
    Expose one trainable scalar for epoch-aggregation tests.

    Callers use zero learning rate so predictions reveal only batch weighting.
    this helper is not a model-accuracy fixture.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale a synthetic batch while retaining an autograd path."""
        return self.scale * value


class _SyntheticComponentLoss(torch.nn.Module):
    """Return batch-mean components whose unequal final batch is observable."""

    physics_enabled = False

    def __init__(self) -> None:
        super().__init__()
        self.last_components: dict[str, torch.Tensor] = {}

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch mean and expose detached total/data components."""
        del x
        if y is None:
            raise ValueError("target required")
        value = (pred - y).mean()
        self.last_components = {"total": value.detach(), "data": value.detach()}
        return value


class _SyntheticPhysicsComponentLoss(_SyntheticComponentLoss):
    """Expose exactly one continuity and fixed applied-weight telemetry."""

    physics_enabled = True
    continuity = "div_velocity"

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-ratio named physics components from one batch mean."""
        del x
        if y is None:
            raise ValueError("target required")
        data = (pred - y).mean()
        momentum = 2.0 * data
        boundary = 3.0 * data
        continuity = 4.0 * data
        total = data + momentum + boundary + continuity
        self.last_components = {
            "total": total.detach(),
            "data": data.detach(),
            "momentum": momentum.detach(),
            "boundary": boundary.detach(),
            "continuity_div_velocity": continuity.detach(),
        }
        return total

    def telemetry_state(self) -> dict[str, float]:
        """Return fixed applied-weight telemetry for epoch aggregation."""
        return {
            "weight_physics": 0.25,
            "weight_boundary": 0.5,
            "warmup_physics_fraction": 0.75,
            "warmup_boundary_fraction": 1.0,
        }


def test_epoch_loss_components_are_weighted_by_actual_sample_count() -> None:
    """
    Train over a two-sample batch followed by one unequal-valued singleton.

    Every named component and fixed physics telemetry must use actual sample count,
    preventing a partial batch from receiving equal weight to a full batch.
    """
    samples = [{"x": torch.full((1, 1, 1), value), "y": torch.zeros((1, 1, 1))} for value in (1.0, 3.0, 9.0)]
    loader = DataLoader(_ListMappingDataset(samples), batch_size=2, shuffle=False)
    model = _IdentityScaleModel()
    optimizer = SGD(model.parameters(), lr=0.0)

    values = learning.training.loop.train_one_epoch(
        model,
        loader,
        optimizer,
        _SyntheticComponentLoss(),
        torch.device("cpu"),
    )

    assert values["train/loss_total"] == pytest.approx(13.0 / 3.0)
    assert values["train/loss_data"] == pytest.approx(13.0 / 3.0)
    assert values["train/loss_total"] != pytest.approx((2.0 + 9.0) / 2.0)
    assert values["optimizer_steps"] == 2.0
    assert values["system/train_duration_seconds"] > 0.0
    assert values["system/train_samples_per_second"] > 0.0
    assert not any("momentum" in key or "continuity" in key for key in values)

    physics_model = _IdentityScaleModel()
    physics_values = learning.training.loop.train_one_epoch(
        physics_model,
        loader,
        SGD(physics_model.parameters(), lr=0.0),
        _SyntheticPhysicsComponentLoss(),
        torch.device("cpu"),
    )
    assert set(physics_values) == {
        "train/loss_total",
        "train/loss_data",
        "physics/train/loss_momentum",
        "physics/train/loss_boundary",
        "physics/train/loss_continuity_div_velocity",
        "physics/train/residual_weight",
        "physics/train/boundary_weight",
        "system/train_duration_seconds",
        "system/train_samples_per_second",
        "optimizer_steps",
    }
    assert physics_values["physics/train/residual_weight"] == 0.25
    assert "physics/train/loss_continuity_div_eps_velocity" not in physics_values


class _ManufacturedMonitorModel(torch.nn.Module):
    """Return a deterministic physical-as-normalized velocity field."""

    def __init__(self) -> None:
        super().__init__()
        self.samples_seen = 0
        self.inference_modes: list[bool] = []
        self.grad_modes: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return manufactured outputs and record bounded inference isolation."""
        self.samples_seen += int(inputs.shape[0])
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.grad_modes.append(torch.is_grad_enabled())
        coordinate = next(field.name for field in _TASK.inputs if field.role == "coordinate")
        coordinate_values = inputs[:, _TASK.input_names.index(coordinate)]
        zeros = torch.zeros_like(coordinate_values)
        velocity = next(group for group in _TASK.output_groups if group.id == "velocity")
        prediction_by_field = dict.fromkeys(_TASK.output_names, zeros)
        prediction_by_field[velocity.fields[0]] = 1e-5 * coordinate_values
        return torch.stack([prediction_by_field[field] for field in _TASK.output_names], dim=1)


def test_physics_monitor_is_bounded_and_reports_both_continuities() -> None:
    """
    Evaluate the shared domain diagnostics on only a fixed evaluation prefix.

    The monitor must consume exactly the bound and report both continuity metrics
    plus momentum/boundary values, without introducing a second physics formula.
    """
    inputs, _pred, target = _manufactured_batch()
    samples = [{"x": inputs[0].clone(), "y": target[0].clone()} for _ in range(3)]
    loader = DataLoader(_ListMappingDataset(samples), batch_size=3, shuffle=False)

    for model_kind in ("fno", "uno"):
        for physics_enabled in (False, True):
            config = experiments.config.loader.resolve_config(
                configs.direct_config(
                    model_kind=model_kind,
                    physics_enabled=physics_enabled,
                ),
            )
            loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"))
            normalizer = IdentityNormalizer()
            loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
            model = _ManufacturedMonitorModel()

            values = learning.training.loop.evaluate_physics_monitor(
                model,
                loader,
                loss,
                torch.device("cpu"),
                data_processor=None,
                max_cases=2,
            )

            assert model.samples_seen == 2
            assert model.inference_modes == [True]
            assert model.grad_modes == [False]
            assert set(values) == {
                "physics/id/momentum_residual_mse",
                "physics/id/continuity_div_velocity_mse",
                "physics/id/continuity_div_eps_velocity_mse",
                "physics/id/pressure_boundary_mse",
            }
            assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
            assert values["physics/id/continuity_div_velocity_mse"] != values["physics/id/continuity_div_eps_velocity_mse"]


class _FailingMonitorLoss(torch.nn.Module):
    """Inject the demonstrated spacing owner below the monitor boundary."""

    def compute_physics_diagnostics(self, _pred: torch.Tensor, *, x: torch.Tensor) -> None:
        del x
        msg = "x-coordinate spacing is not uniform within tolerance"
        raise ValueError(msg)


def test_physics_monitor_wraps_numerics_as_scientific_evaluation_with_cause() -> None:
    """Keep numerical monitor failures scientific and restore model state."""
    inputs, _pred, target = _manufactured_batch()
    loader = DataLoader(
        _ListMappingDataset([{"x": inputs[0], "y": target[0]}]),
        batch_size=1,
        shuffle=False,
    )
    model = _ManufacturedMonitorModel()
    model.train()
    with pytest.raises(learning.training.loop.PhysicsMonitorEvaluationError) as captured:
        learning.training.loop.evaluate_physics_monitor(
            model,
            loader,
            _FailingMonitorLoss(),
            torch.device("cpu"),
            data_processor=None,
            max_cases=1,
        )
    assert isinstance(captured.value.__cause__, ValueError)
    assert "x-coordinate spacing" in str(captured.value.__cause__)
    assert model.training
