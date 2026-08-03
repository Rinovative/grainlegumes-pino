# ruff: noqa: S101, PLR2004, SLF001
"""Protect optimizer-phase timing and actual-sample training throughput."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from src import learning
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Iterator

loop = learning.training.loop


class _MappingDataset(Dataset[dict[str, torch.Tensor]]):
    """Provide a configurable number of scalar regression samples."""

    def __init__(self, samples: int) -> None:
        self.inputs = torch.arange(float(samples)).reshape(samples, 1)
        self.targets = 2.0 * self.inputs

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.inputs[index], "y": self.targets[index]}


class _RecordingLoader:
    """Yield prescribed batches while exposing iterator construction order."""

    def __init__(self, batches: list[dict[str, torch.Tensor]], events: list[str]) -> None:
        self.batches = batches
        self.events = events

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        self.events.append("loader_iteration")
        yield from self.batches


class _FakeScaler:
    """Exercise the maintained AMP optimizer path without physical CUDA."""

    def get_scale(self) -> float:
        return 16.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: SGD) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _model_and_optimizer() -> tuple[nn.Module, SGD]:
    """Return a minimal differentiable CPU model and optimizer."""
    model = nn.Linear(1, 1, bias=False)
    return model, SGD(model.parameters(), lr=0.0)


def _fixed_clock(monkeypatch: pytest.MonkeyPatch, start: float, stop: float) -> None:
    """Install one two-reading monotonic clock."""
    ticks = iter((start, stop))
    monkeypatch.setattr(loop.time, "perf_counter", lambda: next(ticks))


def test_cpu_timer_wraps_loader_iteration_and_final_optimizer_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start before loader iteration, stop after optimizer work, and never touch CUDA."""
    events: list[str] = []
    batches = [
        {"x": torch.ones(2, 1), "y": torch.zeros(2, 1)},
        {"x": torch.ones(3, 1), "y": torch.zeros(3, 1)},
    ]
    loader = cast("DataLoader[Any]", _RecordingLoader(batches, events))
    model, optimizer = _model_and_optimizer()
    original_step = optimizer.step

    def record_step(*args: Any, **kwargs: Any) -> Any:
        events.append("optimizer_step")
        return original_step(*args, **kwargs)

    def clock() -> float:
        events.append("clock")
        return 10.0 if events.count("clock") == 1 else 12.0

    def forbidden_sync(_device: torch.device) -> None:
        pytest.fail("CPU training must not synchronize CUDA")

    monkeypatch.setattr(optimizer, "step", record_step)
    monkeypatch.setattr(loop.time, "perf_counter", clock)
    monkeypatch.setattr(loop.torch.cuda, "synchronize", forbidden_sync)

    values = loop.train_one_epoch(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
    )

    assert events[0:2] == ["clock", "loader_iteration"]
    assert events[-1] == "clock"
    assert max(index for index, event in enumerate(events) if event == "optimizer_step") < len(events) - 1
    assert values["system/train_duration_seconds"] == 2.0
    assert values["system/train_samples_per_second"] == 2.5


@pytest.mark.parametrize(
    ("batch_size", "drop_last", "processed_samples"),
    [
        (2, False, 5),
        (4, False, 5),
        (2, True, 4),
        (4, True, 4),
    ],
)
def test_throughput_counts_actual_full_partial_and_retained_samples(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    drop_last: bool,
    processed_samples: int,
) -> None:
    """Use observed batch dimensions across batch sizes and drop-last policies."""
    loader = DataLoader(
        _MappingDataset(5),
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
    )
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 20.0, 22.0)

    values = loop.train_one_epoch(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
    )

    assert values["system/train_duration_seconds"] == 2.0
    assert values["system/train_samples_per_second"] == processed_samples / 2.0
    assert values["system/train_samples_per_second"] != len(loader) * batch_size / 2.0 or processed_samples == len(loader) * batch_size


@pytest.mark.parametrize("stop", [5.0, math.nan, math.inf])
def test_invalid_training_duration_fails_without_publishing_throughput(
    monkeypatch: pytest.MonkeyPatch,
    stop: float,
) -> None:
    """Reject zero and non-finite durations rather than emit misleading values."""
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 5.0, stop)
    loader = DataLoader(_MappingDataset(2), batch_size=2)

    with pytest.raises(RuntimeError, match="invalid duration"):
        loop.train_one_epoch(
            model,
            loader,
            optimizer,
            nn.MSELoss(),
            torch.device("cpu"),
        )


@pytest.mark.parametrize("use_amp", [False, True])
def test_cuda_synchronizes_once_before_and_after_training(
    monkeypatch: pytest.MonkeyPatch,
    use_amp: bool,
) -> None:
    """Use one synchronization pair for CUDA in AMP and non-AMP paths."""
    device = torch.device("cuda:3")
    calls: list[torch.device] = []
    batch = {"x": torch.ones(2, 1), "y": torch.zeros(2, 1)}
    loader = cast("DataLoader[Any]", [batch])
    model, optimizer = _model_and_optimizer()
    _fixed_clock(monkeypatch, 30.0, 31.0)
    monkeypatch.setattr(loop.torch.cuda, "synchronize", calls.append)
    monkeypatch.setattr(
        loop,
        "_prepare_batch",
        lambda raw_batch, _device, _processor, **_kwargs: raw_batch,
    )
    if use_amp:
        monkeypatch.setattr(loop.torch, "autocast", lambda **_kwargs: nullcontext())

    values = loop.train_one_epoch(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        device,
        scaler=_FakeScaler() if use_amp else None,
        use_amp=use_amp,
    )

    assert calls == [device, device]
    assert values["system/train_duration_seconds"] == 1.0
    assert values["system/train_samples_per_second"] == 2.0


@pytest.mark.parametrize(
    "batch",
    [
        {"x": torch.tensor(1.0), "y": torch.tensor(1.0)},
        {"x": torch.ones(2, 1), "y": torch.ones(1, 1)},
        {"x": torch.ones(0, 1), "y": torch.ones(0, 1)},
    ],
)
def test_batch_sample_dimension_must_be_unambiguous_and_positive(
    batch: dict[str, torch.Tensor],
) -> None:
    """Reject scalar, inconsistent, and empty batch sample dimensions."""
    with pytest.raises(ValueError, match="sample"):
        loop._training_batch_size(batch)
