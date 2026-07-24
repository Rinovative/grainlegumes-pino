# ruff: noqa: NPY002, S101, PLR2004
"""
Verify exact epoch-boundary resume and strict checkpoint publication.

The synthetic loop compares uninterrupted and resumed state while focused
failure cases enforce checkpoint identity, role, and completion rules.
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from src import experiments, learning
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


class _MappingDataset(Dataset[dict[str, torch.Tensor]]):
    """Small deterministic regression dataset."""

    def __init__(self) -> None:
        self.x = torch.linspace(-1.0, 1.0, 24).reshape(12, 2)
        self.y = torch.stack((self.x[:, 0] - self.x[:, 1], self.x[:, 0] + 0.5 * self.x[:, 1]), dim=1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


class _BatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Wrap prescribed tensor batches in PyTorch's typed dataset contract."""

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class _DatasetMSE:
    """Dataset-accumulated normalized MSE metric."""

    id = "mse"
    space = "normalized"

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        del batch_index
        assert space == self.space
        difference = pred - target
        self.total += float(torch.sum(difference * difference))
        self.count += difference.numel()

    def compute(self) -> float:
        return self.total / self.count


class _NonFiniteMetric(_DatasetMSE):
    """Metric used to prove non-finite objectives fail closed."""

    def compute(self) -> float:
        return math.nan


class _ObjectiveStreamMetric(_DatasetMSE):
    """Return a prescribed finite objective sequence across evaluations."""

    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def reset(self) -> None:
        pass

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        del pred, target, batch_index
        assert space == self.space

    def compute(self) -> float:
        return next(self.values)


class _StatefulLoss(nn.Module):
    """MSE with explicit epoch state that exact resume must preserve."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("epoch_index", torch.tensor(-1, dtype=torch.long))

    def set_epoch(self, epoch_index: int) -> None:
        self.epoch_index.fill_(epoch_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target).square())


class _NonFiniteTrainingLoss(nn.Module):
    """Return one prescribed non-finite value while retaining an autograd path."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return pred.sum() * 0.0 + self.value


class _PredictionSumLoss(nn.Module):
    """Return a finite prediction sum that can induce a scaled-gradient overflow."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return pred.sum()


def _objective(*, direction: str = "minimize") -> dict[str, Any]:
    """Return one complete synthetic resolved objective."""
    return {
        "id": "mse",
        "kind": "mse",
        "space": "normalized",
        "fields": ["first", "second"],
        "reduction": "element_mean",
        "direction": direction,
    }


def _config(epochs: int, *, direction: str = "minimize", evaluation_interval: int = 2) -> dict[str, Any]:
    """Return the minimal resolved training contract used by loop tests."""
    return {
        "run": {"device": "cpu"},
        "training": {"epochs": epochs, "evaluation_interval": evaluation_interval},
        "evaluation": {"objective": _objective(direction=direction)},
    }


def _identity(run: str = "synthetic", *, direction: str = "minimize") -> dict[str, Any]:
    """Return one exact synthetic checkpoint identity."""
    return {
        "task": run,
        "task_contract_digest": f"{run}-task-digest",
        "effective_config_digest": f"{run}-config-digest",
        "resume_contract_digest": f"{run}-resume-digest",
        "dataset_fingerprints": {"train": f"{run}-train", "ood": f"{run}-ood"},
        "split_membership_digests": {
            "train": f"{run}-train-membership",
            "eval": f"{run}-eval-membership",
            "ood": f"{run}-ood-membership",
        },
        "objective": _objective(direction=direction),
    }


def _components(seed: int) -> tuple[nn.Module, Optimizer, Any, nn.Module, DataLoader[Any], DataLoader[Any]]:
    """Construct deterministically seeded model, state, and data loaders."""
    experiments.run.seed_process(seed)
    dataset = _MappingDataset()
    generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=generator)
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Dropout(0.2), nn.Linear(8, 2))
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    return model, optimizer, scheduler, _StatefulLoss(), train_loader, eval_loader


def _loader_generator(loader: DataLoader[Any]) -> torch.Generator:
    """Return the generator explicitly configured on a training loader."""
    generator = loader.generator
    assert generator is not None
    return generator


def _run(
    run_dir: Path,
    *,
    epochs: int,
    seed: int,
    resume_from: Path | None = None,
) -> tuple[dict[str, Any], nn.Module, Optimizer, Any, nn.Module, DataLoader[Any]]:
    """Execute one synthetic training segment, optionally from last checkpoint."""
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(seed)
    result = learning.training.loop.train_loop(
        config=_config(epochs),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        resume_from=resume_from,
        checkpoint_identity=_identity(),
    )
    return result, model, optimizer, scheduler, loss, train_loader


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Compare nested tensor, NumPy, collection, and scalar state exactly."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_uninterrupted_and_resumed_training_are_state_identical(tmp_path: Path) -> None:
    """Resume from last reproduces uninterrupted model, optimizer, RNG, and history."""
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    full_dir.mkdir()
    resumed_dir.mkdir()

    full_result, full_model, full_optimizer, full_scheduler, full_loss, full_loader = _run(full_dir, epochs=4, seed=151)
    _run(resumed_dir, epochs=2, seed=151)
    resumed_result, resumed_model, resumed_optimizer, resumed_scheduler, resumed_loss, resumed_loader = _run(
        resumed_dir,
        epochs=4,
        seed=999,
        resume_from=resumed_dir / "last_checkpoint.pt",
    )

    assert resumed_result == full_result | {
        "checkpoint_path": str(resumed_dir / "best_checkpoint.pt"),
        "best_checkpoint_path": str(resumed_dir / "best_checkpoint.pt"),
        "last_checkpoint_path": str(resumed_dir / "last_checkpoint.pt"),
    }
    _assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(full_scheduler.state_dict(), resumed_scheduler.state_dict())
    _assert_nested_equal(full_loss.state_dict(), resumed_loss.state_dict())
    assert torch.equal(_loader_generator(full_loader).get_state(), _loader_generator(resumed_loader).get_state())

    full_last = torch.load(full_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    resumed_last = torch.load(resumed_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    for key in (
        "completed_epoch",
        "next_epoch",
        "global_step",
        "best_metric",
        "best_epoch",
        "objective_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "train_loader_generator_state",
    ):
        _assert_nested_equal(full_last[key], resumed_last[key])


def test_best_and_last_have_distinct_enforced_roles(tmp_path: Path) -> None:
    """Best is selection/inference state while last is continuation state."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=3, seed=31)
    identity = _identity()

    best = learning.training.checkpoint.load_checkpoint(
        run_dir / "best_checkpoint.pt",
        expected_identity=identity,
        expected_role="best",
        scheduler_expected=True,
        amp_expected=False,
        require_best=True,
    )
    last = learning.training.checkpoint.load_checkpoint(
        run_dir / "last_checkpoint.pt",
        expected_identity=identity,
        expected_role="last",
        scheduler_expected=True,
        amp_expected=False,
        require_best=True,
    )

    assert best["checkpoint_role"] == "best"
    assert last["checkpoint_role"] == "last"
    assert last["completed_epoch"] == 3
    with pytest.raises(ValueError, match="role mismatch"):
        learning.training.checkpoint.load_checkpoint(
            run_dir / "last_checkpoint.pt",
            expected_identity=identity,
            expected_role="best",
            scheduler_expected=True,
            amp_expected=False,
            require_best=True,
        )


def test_schema_or_identity_failure_precedes_runtime_mutation(tmp_path: Path) -> None:
    """An invalid last checkpoint cannot partially mutate the model."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=2, seed=7)
    payload = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    payload["identity"] = _identity("other")

    model, optimizer, scheduler, loss, train_loader, _ = _components(99)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="identity is incompatible"):
        learning.training.checkpoint.restore_checkpoint(
            payload,
            expected_identity=_identity(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            amp_enabled=False,
            loss=loss,
            train_loader=train_loader,
        )
    _assert_nested_equal(before, model.state_dict())


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf],
    ids=("nan", "positive-inf", "negative-inf"),
)
def test_non_finite_training_loss_precedes_backward(value: float) -> None:
    """NaN and infinity leave model parameters and optimizer state untouched."""
    model, optimizer, _scheduler, _loss, train_loader, _eval_loader = _components(73)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(FloatingPointError, match="non-finite before backward"):
        learning.training.loop.train_one_epoch(
            model,
            train_loader,
            optimizer,
            _NonFiniteTrainingLoss(value),
            torch.device("cpu"),
        )

    _assert_nested_equal(model_before, model.state_dict())
    _assert_nested_equal(optimizer_before, optimizer.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_grad_scaler_overflow_counts_only_real_steps_and_resumes() -> None:
    """A skipped CPU-scaled update does not advance persisted global progress."""
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu", init_scale=65536.0)
    overflow_loader = DataLoader(
        _BatchDataset([{"x": torch.tensor([1e35]), "y": torch.zeros(1)}]),
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(11),
    )
    weight_before = model.weight.detach().clone()

    overflow_metrics = learning.training.loop.train_one_epoch(
        model,
        overflow_loader,
        optimizer,
        _PredictionSumLoss(),
        torch.device("cpu"),
        scaler=scaler,
        use_amp=True,
    )

    assert overflow_metrics["optimizer_steps"] == 0.0
    assert torch.equal(model.weight, weight_before)
    assert scaler.get_scale() == 32768.0

    safe_loader = DataLoader(
        _BatchDataset([{"x": torch.ones(1), "y": torch.zeros(1)}]),
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(12),
    )
    safe_metrics = learning.training.loop.train_one_epoch(
        model,
        safe_loader,
        optimizer,
        _PredictionSumLoss(),
        torch.device("cpu"),
        scaler=scaler,
        use_amp=True,
    )
    global_step = int(overflow_metrics["optimizer_steps"] + safe_metrics["optimizer_steps"])
    assert global_step == 1

    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        identity=_identity("overflow"),
        completed_epoch=1,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        amp_enabled=True,
        loss=_PredictionSumLoss(),
        best_metric=0.5,
        best_epoch=1,
        objective_history=[{"epoch": 1, "objective_id": "mse", "value": 0.5}],
        train_loader=safe_loader,
        runtime_device="cpu",
    )
    restored_model = nn.Linear(1, 1, bias=False)
    restored_optimizer = SGD(restored_model.parameters(), lr=0.1)
    restored_scaler = GradScaler("cpu", init_scale=2.0)
    restored_loader = DataLoader(
        _BatchDataset([{"x": torch.ones(1), "y": torch.zeros(1)}]),
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(99),
    )
    restored = learning.training.checkpoint.restore_checkpoint(
        payload,
        expected_identity=_identity("overflow"),
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=None,
        scaler=restored_scaler,
        amp_enabled=True,
        loss=_PredictionSumLoss(),
        train_loader=restored_loader,
    )

    assert restored["global_step"] == 1
    _assert_nested_equal(restored_model.state_dict(), model.state_dict())
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert restored_scaler.state_dict() == scaler.state_dict()


def test_cpu_checkpoint_ignores_unrelated_host_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU runtime never captures CUDA state merely because CUDA is present."""
    model, optimizer, scheduler, loss, train_loader, _ = _components(81)

    def unexpected_cuda_capture(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        msg = "CPU checkpoint queried CUDA RNG"
        raise AssertionError(msg)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected_cuda_capture)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        identity=_identity("cpu-portable"),
        completed_epoch=1,
        global_step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        amp_enabled=False,
        loss=loss,
        best_metric=None,
        best_epoch=None,
        objective_history=[],
        train_loader=train_loader,
        runtime_device="cpu",
    )

    assert payload["torch_cuda_rng_states"] == []


def test_cpu_restore_is_portable_when_checkpoint_contains_cuda_rng() -> None:
    """Saved CUDA RNG does not prevent a deliberate CPU-device restore."""
    source_model, source_optimizer, source_scheduler, source_loss, source_loader, _ = _components(82)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        identity=_identity("device-portable"),
        completed_epoch=1,
        global_step=0,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        scaler=None,
        amp_enabled=False,
        loss=source_loss,
        best_metric=None,
        best_epoch=None,
        objective_history=[],
        train_loader=source_loader,
        runtime_device="cpu",
    )
    payload["torch_cuda_rng_states"] = [torch.get_rng_state().clone()]

    model, optimizer, scheduler, loss, train_loader, _ = _components(83)
    restored = learning.training.checkpoint.restore_checkpoint(
        payload,
        expected_identity=_identity("device-portable"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        amp_enabled=False,
        loss=loss,
        train_loader=train_loader,
    )

    assert restored["global_step"] == 0
    _assert_nested_equal(model.state_dict(), source_model.state_dict())


def test_late_restore_failure_rolls_back_every_runtime_state() -> None:
    """A final-component failure restores objects, loader state, and process RNG."""
    source_model, source_optimizer, source_scheduler, source_loss, source_loader, _ = _components(84)
    source_batch = next(iter(source_loader))
    source_optimizer.zero_grad()
    source_loss(source_model(source_batch["x"]), source_batch["y"]).backward()
    source_optimizer.step()
    source_scheduler.step(0.25)
    source_loss.set_epoch(3)
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        identity=_identity("transactional"),
        completed_epoch=1,
        global_step=1,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        scaler=None,
        amp_enabled=False,
        loss=source_loss,
        best_metric=None,
        best_epoch=None,
        objective_history=[],
        train_loader=source_loader,
        runtime_device="cpu",
    )
    payload["loss_state_dict"] = {"unexpected": torch.tensor(1.0)}

    model, optimizer, scheduler, loss, train_loader, _ = _components(85)
    destination_batch = next(iter(train_loader))
    optimizer.zero_grad()
    loss(model(destination_batch["x"]), destination_batch["y"]).backward()
    optimizer.step()
    scheduler.step(2.0)
    loss.set_epoch(7)
    before = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "loss": copy.deepcopy(loss.state_dict()),
        "loader": _loader_generator(train_loader).get_state().clone(),
        "python_rng": random.getstate(),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
        "torch_rng": torch.get_rng_state().clone(),
    }

    with pytest.raises(RuntimeError, match="Unexpected key"):
        learning.training.checkpoint.restore_checkpoint(
            payload,
            expected_identity=_identity("transactional"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            amp_enabled=False,
            loss=loss,
            train_loader=train_loader,
        )

    _assert_nested_equal(before["model"], model.state_dict())
    _assert_nested_equal(before["optimizer"], optimizer.state_dict())
    _assert_nested_equal(before["scheduler"], scheduler.state_dict())
    _assert_nested_equal(before["loss"], loss.state_dict())
    assert torch.equal(before["loader"], _loader_generator(train_loader).get_state())
    _assert_nested_equal(before["python_rng"], random.getstate())
    _assert_nested_equal(before["numpy_rng"], np.random.get_state())
    assert torch.equal(before["torch_rng"], torch.get_rng_state())


def test_non_finite_objective_never_creates_a_loadable_best(tmp_path: Path) -> None:
    """NaN objectives fail instead of becoming summaries or best checkpoints."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(5)

    with pytest.raises(FloatingPointError, match="non-finite"):
        learning.training.loop.train_loop(
            config={
                "run": {"device": "cpu"},
                "training": {"epochs": 1, "evaluation_interval": 1},
                "evaluation": {"objective": _objective()},
            },
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_loss=loss,
            eval_metrics={"mse": _NonFiniteMetric()},
            scheduler=scheduler,
            save_dir=run_dir,
            checkpoint_identity=_identity(),
        )

    assert not (run_dir / "best_checkpoint.pt").exists()
    assert not (run_dir / "last_checkpoint.pt").exists()


def test_maximize_objective_selects_and_persists_the_largest_value(tmp_path: Path) -> None:
    """Training and checkpoint validation honor a maximizing objective."""
    run_dir = tmp_path / "maximize"
    run_dir.mkdir()
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(17)
    objective = _objective(direction="maximize")

    result = learning.training.loop.train_loop(
        config=_config(2, direction="maximize", evaluation_interval=1),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _ObjectiveStreamMetric([1.0, 2.0])},
        save_dir=run_dir,
        checkpoint_identity=_identity("maximize", direction="maximize"),
    )

    assert result["objective"] == objective
    assert result["best_epoch"] == 2
    assert result["best_metric"] == 2.0
    payload = learning.training.checkpoint.load_checkpoint(
        run_dir / "last_checkpoint.pt",
        expected_identity=_identity("maximize", direction="maximize"),
        expected_role="last",
        scheduler_expected=False,
        amp_expected=False,
        require_best=True,
    )
    assert payload["identity"]["objective"] == objective


def test_every_missing_checkpoint_field_is_rejected(tmp_path: Path) -> None:
    """Enforce every required checkpoint field independently."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run(run_dir, epochs=2, seed=41)
    payload = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    identity = _identity()

    for missing_key in tuple(payload):
        incomplete = dict(payload)
        incomplete.pop(missing_key)
        with pytest.raises(ValueError, match="schema mismatch"):
            learning.training.checkpoint.validate_checkpoint(
                incomplete,
                expected_identity=identity,
                expected_role="last",
                scheduler_expected=True,
                amp_expected=False,
                require_best=True,
            )


def test_cpu_scaler_state_round_trips_through_checkpoint() -> None:
    """A non-empty scaler state is captured and restored with all components."""
    model, optimizer, scheduler, loss, train_loader, _ = _components(61)
    scaler = GradScaler("cpu", init_scale=64.0, growth_interval=1)
    batch = next(iter(train_loader))
    optimizer.zero_grad()
    scaled_loss = loss(model(batch["x"]), batch["y"])
    scaler.scale(scaled_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    source_scaler_state = copy.deepcopy(scaler.state_dict())
    payload = learning.training.checkpoint.make_checkpoint(
        role="last",
        identity=_identity("scaled"),
        completed_epoch=1,
        global_step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        amp_enabled=True,
        loss=loss,
        best_metric=0.5,
        best_epoch=1,
        objective_history=[{"epoch": 1, "objective_id": "mse", "value": 0.5}],
        train_loader=train_loader,
    )

    restored_model, restored_optimizer, restored_scheduler, restored_loss, restored_loader, _ = _components(999)
    restored_scaler = GradScaler("cpu", init_scale=2.0)
    learning.training.checkpoint.restore_checkpoint(
        payload,
        expected_identity=_identity("scaled"),
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        amp_enabled=True,
        loss=restored_loss,
        train_loader=restored_loader,
    )

    assert restored_scaler.state_dict() == source_scaler_state
    _assert_nested_equal(restored_model.state_dict(), model.state_dict())
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    _assert_nested_equal(restored_scheduler.state_dict(), scheduler.state_dict())
    _assert_nested_equal(restored_loss.state_dict(), loss.state_dict())


def test_final_epoch_is_evaluated_when_interval_is_larger(tmp_path: Path) -> None:
    """A one-epoch run with interval five still publishes a valid best."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(12)
    config = {
        "run": {"device": "cpu"},
        "training": {"epochs": 1, "evaluation_interval": 5},
        "evaluation": {"objective": _objective()},
    }

    result = learning.training.loop.train_loop(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        checkpoint_identity=_identity("final-eval"),
    )

    assert result["best_epoch"] == 1
    assert math.isfinite(result["best_metric"])
    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "last_checkpoint.pt").is_file()


def test_missing_best_prevents_successful_loop_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion reload-validation rejects a missing advertised best file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(14)
    real_save = learning.training.checkpoint.save_checkpoint

    def drop_best(payload: dict[str, Any], checkpoint_path: Path) -> Path:
        if payload["checkpoint_role"] == "best":
            return Path(checkpoint_path)
        return real_save(payload, checkpoint_path)

    monkeypatch.setattr(learning.training.checkpoint, "save_checkpoint", drop_best)
    with pytest.raises(FileNotFoundError, match="Required best checkpoint"):
        learning.training.loop.train_loop(
            config={
                "run": {"device": "cpu"},
                "training": {"epochs": 1, "evaluation_interval": 1},
                "evaluation": {"objective": _objective()},
            },
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_loss=loss,
            eval_metrics={"mse": _DatasetMSE()},
            scheduler=scheduler,
            save_dir=run_dir,
            checkpoint_identity=_identity("missing-best"),
        )

    assert not (run_dir / "best_checkpoint.pt").exists()
    assert (run_dir / "last_checkpoint.pt").is_file()
