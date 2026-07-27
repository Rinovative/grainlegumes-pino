# ruff: noqa: NPY002, S101, PLR2004
"""
Protect exact epoch-boundary resume and strict best/last checkpoint publication.

Deterministic synthetic loaders/models cover optimizer, scheduler, scaler, RNG,
history, objective roles, schema validation-before-mutation, non-finite loss,
writer failure, and concurrent run behavior. Run-directory allocation and high-level
inference admission are covered by their dedicated modules.
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
    """
    Provide a fixed twelve-sample regression dataset for exact-resume tests.

    Samples are generated in memory with no normalization or storage behavior;
    deterministic shuffle state is owned by the surrounding DataLoader.
    """

    def __init__(self) -> None:
        self.x = torch.linspace(-1.0, 1.0, 24).reshape(12, 2)
        self.y = torch.stack((self.x[:, 0] - self.x[:, 1], self.x[:, 0] + 0.5 * self.x[:, 1]), dim=1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


class _BatchDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Wrap prescribed tensor samples in PyTorch's typed dataset contract.

    Parameters
    ----------
    samples : list[dict[str, torch.Tensor]]
        Caller-owned synthetic samples retained in order without copying.

    Notes
    -----
    The helper deliberately models no task fields, identity, or persistence.

    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class _DatasetMSE:
    """
    Accumulate normalized MSE from global squared-error/count evidence.

    The helper mirrors the evaluation metric protocol but omits field selection,
    units, and distributed state so checkpoint tests isolate lifecycle behavior.
    """

    id = "mse"
    space = "normalized"

    def reset(self) -> None:
        """Clear accumulated squared error and element count."""
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
        """Accumulate one normalized batch using elementwise SSE and count."""
        del batch_index
        assert space == self.space
        difference = pred - target
        self.total += float(torch.sum(difference * difference))
        self.count += difference.numel()

    def compute(self) -> float:
        """Return the global element-mean squared error."""
        return self.total / self.count


class _NonFiniteMetric(_DatasetMSE):
    """Metric used to prove non-finite objectives fail closed."""

    def compute(self) -> float:
        """Return NaN to exercise non-finite objective admission."""
        return math.nan


class _ObjectiveStreamMetric(_DatasetMSE):
    """
    Return a prescribed finite objective sequence across evaluations.

    Parameters
    ----------
    values : list[float]
        One value consumed by each ``compute`` call in evaluation order.

    """

    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def reset(self) -> None:
        """Preserve the prescribed stream across evaluation resets."""

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        space: str,
        batch_index: int,
    ) -> None:
        """Validate normalized routing without accumulating batch values."""
        del pred, target, batch_index
        assert space == self.space

    def compute(self) -> float:
        """Consume and return the next prescribed selection objective."""
        return next(self.values)


class _StatefulLoss(nn.Module):
    """MSE with explicit epoch state that exact resume must preserve."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("epoch_index", torch.tensor(-1, dtype=torch.long))

    def set_epoch(self, epoch_index: int) -> None:
        """Persist the completed training epoch in module state."""
        self.epoch_index.fill_(epoch_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return mean squared error while retaining serializable epoch state."""
        return torch.mean((pred - target).square())


class _NonFiniteTrainingLoss(nn.Module):
    """
    Return one prescribed non-finite value while retaining an autograd path.

    Parameters
    ----------
    value : float
        NaN or signed infinity injected before backward.

    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Attach the prescribed non-finite scalar to the prediction graph."""
        del target
        return pred.sum() * 0.0 + self.value


class _PredictionSumLoss(nn.Module):
    """Return a finite prediction sum that can induce a scaled-gradient overflow."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Sum predictions to expose gradient overflow under artificial scaling."""
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
    """
    Build a complete synthetic checkpoint identity namespaced by one run label.

    Task, config, resume, dataset, and ordered split digests vary together, while
    objective direction stays aligned with selection tests to avoid unrelated failures.
    """
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
    """
    Construct deterministically seeded model, optimizer, scheduler, loss, and loaders.

    The training loader owns an explicit generator whose state is part of exact
    checkpoint continuation; all components remain CPU-local and synthetic.
    """
    experiments.run.seed_process(seed, device=torch.device("cpu"))
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
    """
    Execute one synthetic training segment, optionally from ``last_checkpoint.pt``.

    Checkpoints are published below ``run_dir`` and the returned live components
    expose state for exact comparison with an uninterrupted segment.
    """
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(seed)
    result = learning.training.loop.train_loop(
        config=_config(epochs),
        device=torch.device("cpu"),
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
    """
    Compare one uninterrupted four-epoch run with a two-plus-two exact resume.

    Model, optimizer, scheduler, loss warmup, loader epoch, RNG, objective, and
    history state must match, proving ``last_checkpoint.pt`` is sufficient continuation.
    """
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
    """
    Train three epochs and load both published checkpoint roles explicitly.

    Each file must validate only for its declared role; treating ``last`` as
    ``best`` must fail so inference selection and continuation remain distinct.
    """
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
    """
    Replace the saved checkpoint identity before restoring into fresh components.

    Validation must fail before any model parameter changes, protecting runtime
    state from partial application of an incompatible continuation payload.
    """
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
    """
    Vary training loss across NaN and both signed infinities before backward.

    Every non-finite family must fail with parameters, gradients, and optimizer
    state untouched, preventing invalid arithmetic from entering an update.
    """
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


def test_cpu_training_rejects_amp_before_iteration() -> None:
    """
    Pass an enabled scaler to a one-batch CPU training call.

    Device validation must fail before iteration or weight mutation because the
    maintained mixed-precision path requires a concrete CUDA runtime.
    """
    model = nn.Linear(1, 1, bias=False)
    optimizer = SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu", init_scale=64.0)
    loader = DataLoader(
        _BatchDataset([{"x": torch.ones(1), "y": torch.zeros(1)}]),
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(11),
    )
    before = model.weight.detach().clone()

    with pytest.raises(ValueError, match="concrete CUDA device"):
        learning.training.loop.train_one_epoch(
            model,
            loader,
            optimizer,
            _PredictionSumLoss(),
            torch.device("cpu"),
            scaler=scaler,
            use_amp=True,
        )

    assert torch.equal(model.weight, before)


def test_cpu_checkpoint_ignores_unrelated_host_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Mock CUDA as available while forbidding its RNG query during CPU checkpoint creation.

    The payload must contain an empty CUDA RNG list, keeping checkpoint contents
    determined by the resolved runtime device rather than unrelated host capability.
    """
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
        runtime_device=torch.device("cpu"),
    )

    assert payload["torch_cuda_rng_states"] == []


def test_cpu_restore_is_portable_when_checkpoint_contains_cuda_rng() -> None:
    """
    Add synthetic CUDA RNG evidence to an otherwise CPU-created checkpoint.

    CPU restoration must ignore that optional state and recover model/global-step
    values, preserving portability when checkpoints move away from GPU hosts.
    """
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
        runtime_device=torch.device("cpu"),
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
    """
    Inject an invalid loss state after all earlier checkpoint components validate.

    Restore must roll back model, optimizer, scheduler, loss, loader generator,
    and every process RNG exactly, proving continuation is transactional.
    """
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
        runtime_device=torch.device("cpu"),
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
    """
    Evaluate one training epoch with a metric that returns NaN.

    The loop must raise and publish neither best nor last checkpoint, preventing
    a non-finite selection objective from creating a loadable completed state.
    """
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
            device=torch.device("cpu"),
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


@pytest.mark.parametrize(
    ("direction", "values", "expected_epoch", "expected_metric"),
    [
        ("minimize", [1.0, 1.0], 1, 1.0),
        ("maximize", [1.0, 2.0], 2, 2.0),
    ],
)
def test_objective_direction_and_strict_ties_control_persisted_best(
    direction: str,
    values: list[float],
    expected_epoch: int,
    expected_metric: float,
    tmp_path: Path,
) -> None:
    """
    Vary objective direction and a two-evaluation value stream including a minimize tie.

    Selection must follow the declared direction, retain the earlier strict tie,
    and persist the same complete objective in checkpoint identity.
    """
    run_dir = tmp_path / direction
    run_dir.mkdir()
    model, optimizer, _scheduler, loss, train_loader, eval_loader = _components(17)
    objective = _objective(direction=direction)
    identity = _identity(f"selection-{direction}", direction=direction)
    observed_learning_rates: list[float] = []

    result = learning.training.loop.train_loop(
        config=_config(2, direction=direction, evaluation_interval=1),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _ObjectiveStreamMetric(values)},
        save_dir=run_dir,
        epoch_end_callback=lambda _epoch, metrics: observed_learning_rates.append(metrics["train/learning_rate"]),
        checkpoint_identity=identity,
    )

    assert observed_learning_rates == [0.05, 0.05]
    assert result["objective"] == objective
    assert result["best_epoch"] == expected_epoch
    assert result["best_metric"] == expected_metric
    payload = learning.training.checkpoint.load_checkpoint(
        run_dir / "last_checkpoint.pt",
        expected_identity=identity,
        expected_role="last",
        scheduler_expected=False,
        amp_expected=False,
        require_best=True,
    )
    assert payload["identity"]["objective"] == objective


def test_every_missing_checkpoint_field_is_rejected(tmp_path: Path) -> None:
    """
    Remove each current checkpoint key in turn while holding all other fields valid.

    Every omission must fail the exact schema check, proving no required continuation
    state is accidentally optional merely because another test covers it.
    """
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


def test_cpu_checkpoint_rejects_amp_scaler_state() -> None:
    """
    Create checkpoint state with AMP enabled, an active scaler, and concrete CPU runtime.

    Publication must reject the contradiction so saved device facts cannot claim
    an unsupported mixed-precision continuation path.
    """
    model, optimizer, scheduler, loss, train_loader, _ = _components(61)
    scaler = GradScaler("cpu", init_scale=64.0)

    with pytest.raises(ValueError, match="concrete CUDA runtime device"):
        learning.training.checkpoint.make_checkpoint(
            role="last",
            identity=_identity("scaled"),
            completed_epoch=1,
            global_step=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=True,
            loss=loss,
            best_metric=None,
            best_epoch=None,
            objective_history=[],
            train_loader=train_loader,
            runtime_device=torch.device("cpu"),
        )


def test_completed_epoch_callback_preserves_evaluation_cadence(
    tmp_path: Path,
) -> None:
    """
    Train three epochs with evaluation interval two and inspect each callback payload.

    Callbacks must follow safe checkpoint publication every epoch, include training
    state always, and expose objective values only at actual evaluation epochs.
    """
    run_dir = tmp_path / "callback-cadence"
    run_dir.mkdir()
    model, optimizer, scheduler, loss, train_loader, eval_loader = _components(28)
    observed: list[tuple[int, dict[str, float]]] = []

    def capture(completed_epoch: int, values: dict[str, float]) -> None:
        payload = torch.load(
            run_dir / "last_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert payload["completed_epoch"] == completed_epoch
        observed.append((completed_epoch, dict(values)))

    result = learning.training.loop.train_loop(
        config=_config(3, evaluation_interval=2),
        device=torch.device("cpu"),
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_loss=loss,
        eval_metrics={"mse": _DatasetMSE()},
        scheduler=scheduler,
        save_dir=run_dir,
        epoch_end_callback=capture,
        checkpoint_identity=_identity("callback-cadence"),
    )

    assert [epoch for epoch, _values in observed] == [1, 2, 3]
    assert "mse" not in observed[0][1]
    assert "mse" in observed[1][1]
    assert "mse" in observed[2][1]
    assert all("train/loss_total" in values for _epoch, values in observed)
    assert all("global_step" in values for _epoch, values in observed)
    assert [entry["epoch"] for entry in result["objective_history"]] == [2, 3]


def test_final_epoch_is_evaluated_when_interval_is_larger(tmp_path: Path) -> None:
    """
    Train one epoch with a nominal evaluation interval larger than total duration.

    The terminal epoch must still evaluate and publish finite best/last checkpoints,
    ensuring a short valid run cannot finish without selection state.
    """
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
        device=torch.device("cpu"),
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
    """
    Stub best-checkpoint publication away while allowing the last checkpoint to save.

    Final reload validation must fail rather than report success, keeping a run
    without its advertised selection artifact from becoming completed.
    """
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
            device=torch.device("cpu"),
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
