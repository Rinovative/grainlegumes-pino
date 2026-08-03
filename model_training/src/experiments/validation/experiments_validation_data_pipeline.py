"""
===============================================================================
experiments_validation_data_pipeline.py
===============================================================================
Validate the complete production data pipeline before an experiment is run.

Responsibilities:
  - Admit complete task-bound ID and OOD datasets and metadata artifacts
  - Verify deterministic split membership, leakage, and loader coverage
  - Independently check train-fitted and normalized per-channel statistics
  - Verify finite transforms, inverse transforms, and sampler-state restoration
  - Return typed scientific records without notebook presentation dependencies

Design principles:
  - Production dataset, split, normalizer, and loader services remain authoritative
  - Independent checks fail closed and never persist validation state
  - The module is imported only for explicit full-data acceptance

This module does NOT:
  - Construct models, losses, optimizers, schedulers, or training runs
  - Initialize W&B, allocate paths, write checkpoints, or generate artifacts
  - Build pandas tables or own notebook display behavior
===============================================================================
"""

from __future__ import annotations

import gc
import resource
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import RandomSampler, SequentialSampler, Subset

from src import datasets
from src.experiments import experiments_run as run_service
from src.experiments.config import experiments_config_loader as config_loader

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec

_FULL_VALIDATION_RTOL = 3e-4
_FULL_VALIDATION_ATOL = 3e-5
_NORMALIZED_MEAN_ATOL = 8e-4
_NORMALIZED_SCALE_RTOL = 8e-4
_NORMALIZED_SCALE_ATOL = 8e-4
_TENSOR_RANK = 4
ResultLabel = Literal["PASS", "INFO"]


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """Record one overall validation result."""

    check: str
    evidence: str
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class DatasetMembershipRecord:
    """Record identity and membership evidence for one dataset role."""

    role: str
    dataset_id: str
    full_samples: int
    expected: int
    observed: int
    duplicates: int
    missing: int
    shape: str
    dtype: str
    fingerprint: str
    task_digest: str
    finite: bool
    policy: str
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class ChannelStatisticsRecord:
    """Record independent fitted and normalized statistics for one channel."""

    tensor_role: str
    channel: str
    fitted_mean: float
    fitted_scale: float
    normalized_mean: float
    normalized_scale: float
    finite: bool
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class LoaderCoverageRecord:
    """Record complete traversal and preprocessing evidence for one loader."""

    loader: str
    sampler: str
    batches: int
    batch_size: int | None
    final_batch: int
    drop_last: bool
    inverse_checked: bool
    finite: bool
    result: ResultLabel


@dataclass(frozen=True, slots=True)
class FullDataValidationResult:
    """Return compact typed evidence from complete read-only validation."""

    overall: tuple[ValidationCheck, ...]
    dataset_membership: tuple[DatasetMembershipRecord, ...]
    channels: tuple[ChannelStatisticsRecord, ...]
    coverage: tuple[LoaderCoverageRecord, ...]
    elapsed_seconds: float
    peak_gib: float


@dataclass(slots=True)
class _ChannelMoments:
    """Accumulate float64 per-channel moments without retaining batches."""

    count: int = 0
    total: Tensor | None = None
    total_sq: Tensor | None = None
    finite: bool = True


@dataclass(frozen=True, slots=True)
class _DatasetContract:
    """Retain one fully admitted source dataset and its identity evidence."""

    source: Any
    identity: Any
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    finite: bool


@dataclass(frozen=True, slots=True)
class _LoaderPass:
    """Retain one complete loader traversal result."""

    observed_ids: tuple[str, ...]
    final_batch_size: int
    inverse_checked: bool


def _require(condition: bool, message: str) -> None:
    """Fail one scientific invariant with an actionable validation error."""
    if not condition:
        detail = f"Full data-pipeline validation failed: {message}"
        raise RuntimeError(detail)


def _update_channel_moments(state: _ChannelMoments, tensor: Tensor) -> None:
    """Accumulate BCHW channel moments without retaining normalized batches."""
    _require(
        tensor.ndim == _TENSOR_RANK,
        f"expected BCHW tensor, got shape {tuple(tensor.shape)}",
    )
    state.finite = bool(state.finite and torch.isfinite(tensor).all().item())
    values = tensor.detach().to(dtype=torch.float64)
    reduced_sum = values.sum(dim=(0, 2, 3)).cpu()
    reduced_sum_sq = values.square().sum(dim=(0, 2, 3)).cpu()
    state.total = reduced_sum if state.total is None else state.total + reduced_sum
    state.total_sq = reduced_sum_sq if state.total_sq is None else state.total_sq + reduced_sum_sq
    state.count += int(values.shape[0] * values.shape[2] * values.shape[3])


def _finalize_channel_moments(state: _ChannelMoments) -> tuple[Tensor, Tensor]:
    """Return mean and Torch-compatible sample standard deviation."""
    _require(state.count > 1, "channel statistics require at least two elements")
    _require(state.total is not None and state.total_sq is not None, "channel moment totals are absent")
    total = cast("Tensor", state.total)
    total_sq = cast("Tensor", state.total_sq)
    mean = total / state.count
    variance = (total_sq - total.square() / state.count) / (state.count - 1)
    return mean, variance.clamp_min(0).sqrt()


def _batch_case_ids(batch: Mapping[str, Any]) -> list[str]:
    """Extract compact admitted case identities from one production batch."""
    metadata = batch.get("meta")
    _require(isinstance(metadata, Mapping), "batch metadata must be a mapping")
    metadata_mapping = cast("Mapping[str, Any]", metadata)
    raw_ids = metadata_mapping.get("case_id")
    if isinstance(raw_ids, Tensor):
        values: Iterable[Any] = raw_ids.tolist()
    elif isinstance(raw_ids, (list, tuple)):
        values = raw_ids
    else:
        values = (raw_ids,)
    case_ids = [str(value) for value in values]
    _require(all(case_ids), "batch case IDs must be non-empty")
    return case_ids


def _full_source_dataset(loader: Any) -> Any:
    """Return the production dataset beneath one admitted Subset."""
    selected = loader.dataset
    _require(isinstance(selected, Subset), "production loader dataset must be a torch Subset")
    return selected.dataset


def _validate_tensor_contract(
    *,
    loader: Any,
    role: str,
    task: TaskSpec,
) -> _DatasetContract:
    """Validate a complete mounted dataset in bounded tensor chunks."""
    source = _full_source_dataset(loader)
    identity = source.identity
    inputs = source.data["inputs"]
    outputs = source.data["outputs"]
    expected_input_shape = (len(source), task.in_channels, *identity.spatial_shape)
    expected_output_shape = (len(source), task.out_channels, *identity.spatial_shape)
    _require(tuple(inputs.shape) == expected_input_shape, f"{role} input shape disagrees with TaskSpec")
    _require(tuple(outputs.shape) == expected_output_shape, f"{role} output shape disagrees with TaskSpec")
    _require(
        inputs.dtype == torch.float32 and outputs.dtype == torch.float32,
        f"{role} tensors must be float32",
    )
    _require(source.input_fields == list(task.input_names), f"{role} input field order changed")
    _require(source.output_fields == list(task.output_names), f"{role} output field order changed")
    _require(identity.task == task.id, f"{role} task identity changed")
    _require(
        identity.task_contract_digest == task.contract_digest,
        f"{role} TaskSpec digest changed",
    )
    _require(
        len(identity.sample_ids) == len(source),
        f"{role} sample IDs do not match sample count",
    )
    _require(
        len(set(identity.sample_ids)) == len(identity.sample_ids),
        f"{role} sample IDs contain duplicates",
    )
    metadata_ids = tuple(str(item["case_id"]) for item in source.data["source_metadata"])
    _require(metadata_ids == identity.sample_ids, f"{role} metadata case IDs are misaligned")
    finite = True
    for start in range(0, len(source), 16):
        finite = finite and bool(torch.isfinite(inputs[start : start + 16]).all().item())
        finite = finite and bool(torch.isfinite(outputs[start : start + 16]).all().item())
    _require(finite, f"{role} complete tensors contain NaN or infinity")
    return _DatasetContract(
        source=source,
        identity=identity,
        input_shape=tuple(inputs.shape),
        output_shape=tuple(outputs.shape),
        dtype=str(inputs.dtype).removeprefix("torch."),
        finite=finite,
    )


def _validate_loader_pass(
    *,
    loader: Any,
    role: str,
    expected_ids: tuple[str, ...],
    processor: Any,
    raw_moments: dict[str, _ChannelMoments] | None = None,
    normalized_moments: dict[str, _ChannelMoments] | None = None,
) -> _LoaderPass:
    """Traverse one complete production membership and validate preprocessing."""
    observed: list[str] = []
    final_batch_size = 0
    inverse_checked = False
    for batch in loader:
        x = batch["x"]
        y = batch["y"]
        _require(
            x.ndim == _TENSOR_RANK and y.ndim == _TENSOR_RANK,
            f"{role} batch is not BCHW",
        )
        _require(
            x.dtype == torch.float32 and y.dtype == torch.float32,
            f"{role} batch dtype changed",
        )
        _require(
            x.shape[1] == len(processor.in_normalizer.mean.reshape(-1)),
            f"{role} input channels changed",
        )
        _require(
            y.shape[1] == len(processor.out_normalizer.mean.reshape(-1)),
            f"{role} output channels changed",
        )
        x_normalized = processor.in_normalizer.transform(x)
        y_normalized = processor.out_normalizer.transform(y)
        _require(
            bool(torch.isfinite(x_normalized).all().item()),
            f"{role} normalized inputs are non-finite",
        )
        _require(
            bool(torch.isfinite(y_normalized).all().item()),
            f"{role} normalized outputs are non-finite",
        )
        if raw_moments is not None:
            _update_channel_moments(raw_moments["input"], x)
            _update_channel_moments(raw_moments["output"], y)
        if normalized_moments is not None:
            _update_channel_moments(normalized_moments["input"], x_normalized)
            _update_channel_moments(normalized_moments["output"], y_normalized)
        if not inverse_checked:
            for tensor_name, original, normalized, normalizer in (
                ("input", x, x_normalized, processor.in_normalizer),
                ("output", y, y_normalized, processor.out_normalizer),
            ):
                reconstructed = normalizer.inverse_transform(normalized)
                rounding_scale = original.abs() + normalizer.mean.abs() + normalizer.std.abs() + 1.0
                rounding_bound = 32 * torch.finfo(original.dtype).eps * rounding_scale
                _require(
                    bool(torch.all((reconstructed - original).abs() <= rounding_bound).item()),
                    f"{role} {tensor_name} inverse transform exceeded the float32 rounding bound",
                )
            inverse_checked = True
        observed.extend(_batch_case_ids(batch))
        final_batch_size = int(x.shape[0])
    _require(len(observed) == len(expected_ids), f"{role} loader count changed")
    _require(len(set(observed)) == len(observed), f"{role} loader produced duplicate case IDs")
    _require(set(observed) == set(expected_ids), f"{role} loader omitted or added case IDs")
    return _LoaderPass(
        observed_ids=tuple(observed),
        final_batch_size=final_batch_size,
        inverse_checked=inverse_checked,
    )


def _membership_record(
    *,
    role: str,
    contract: _DatasetContract,
    expected_ids: tuple[str, ...],
    observed_ids: tuple[str, ...],
    shape: str,
    policy: str,
) -> DatasetMembershipRecord:
    """Build one exact membership record after validation succeeds."""
    duplicates = len(observed_ids) - len(set(observed_ids))
    missing = len(set(expected_ids).difference(observed_ids))
    return DatasetMembershipRecord(
        role=role,
        dataset_id=contract.identity.dataset_id,
        full_samples=len(contract.source),
        expected=len(expected_ids),
        observed=len(observed_ids),
        duplicates=duplicates,
        missing=missing,
        shape=shape,
        dtype=contract.dtype,
        fingerprint=contract.identity.fingerprint[:16],
        task_digest=contract.identity.task_contract_digest[:16],
        finite=contract.finite,
        policy=policy,
        result="PASS",
    )


def validate_full_data_pipeline(config: Mapping[str, Any]) -> FullDataValidationResult:
    """Execute complete read-only production data admission and validation."""
    started = time.perf_counter()
    peak_before_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    effective = config_loader.validate_resolved_config(config)
    task = config_loader.validate_resolved_task_contract(effective)
    _require(
        task.tensor_layout == ("batch", "channel", "y", "x"),
        "TaskSpec layout must be BCHW",
    )
    _require(
        task.normalization_axes == (0, 2, 3),
        "normalization axes must preserve channels",
    )
    _require(
        task.preprocessing.fit_split == "train",
        "normalizer ownership must be train-only",
    )
    seed_plan = run_service.build_seed_plan(int(effective["run"]["seed"]))
    built = config_loader.create_dataloaders_from_config(
        effective,
        seed_plan=seed_plan,
    )
    train_loader = built["train"]
    eval_loader = built["eval"]
    ood_loader = built["ood"]
    processor = built["data_processor"]
    split = built["split_indices"]
    normalizer_state = {key: value.detach().cpu().clone() for key, value in processor.state_dict().items()}

    id_contract = _validate_tensor_contract(loader=train_loader, role="ID", task=task)
    ood_contract = _validate_tensor_contract(loader=ood_loader, role="OOD", task=task)
    metadata_root = Path(effective["paths"]["training_meta_root"])
    for contract in (id_contract, ood_contract):
        datasets.metadata.load_dataset_metadata(
            contract.identity.dataset_id,
            dataset_identity=contract.identity,
            metadata_root=metadata_root,
            dataset_path=contract.source.path,
        )

    validated = datasets.base.validate_split_info(
        split,
        train_identity=id_contract.identity,
        ood_identity=ood_contract.identity,
        expected_train_ratio=effective["data"]["train_ratio"],
        expected_ood_fraction=effective["data"]["ood_fraction"],
        expected_split_seed=seed_plan["split"],
    )
    train_indices = validated["train_indices"]
    eval_indices = validated["eval_indices"]
    ood_indices = validated["ood_indices"]
    _require(
        not bool(torch.isin(train_indices, eval_indices).any().item()),
        "ID train/evaluation overlap",
    )
    id_membership = torch.cat((train_indices, eval_indices))
    _require(
        torch.equal(
            torch.sort(id_membership).values,
            torch.arange(len(id_contract.source)),
        ),
        "ID membership is incomplete",
    )
    _require(
        torch.unique(ood_indices).numel() == ood_indices.numel(),
        "OOD membership contains duplicates",
    )

    id_ids = id_contract.identity.sample_ids
    ood_ids = ood_contract.identity.sample_ids
    expected = {
        "train": tuple(id_ids[int(index)] for index in train_indices.tolist()),
        "eval": tuple(id_ids[int(index)] for index in eval_indices.tolist()),
        "ood": tuple(ood_ids[int(index)] for index in ood_indices.tolist()),
    }

    generator = train_loader.generator
    _require(
        isinstance(generator, torch.Generator),
        "training loader has no explicit generator",
    )
    _require(
        isinstance(train_loader.sampler, RandomSampler),
        "training loader must use RandomSampler",
    )
    _require(
        isinstance(eval_loader.sampler, SequentialSampler),
        "evaluation loader must be stable",
    )
    _require(
        isinstance(ood_loader.sampler, SequentialSampler),
        "OOD loader must be stable",
    )
    sampler_state = generator.get_state().clone()
    first_epoch_batches = tuple(tuple(batch) for batch in train_loader.batch_sampler)
    sampler_after = generator.get_state().clone()
    flat_positions = [position for batch in first_epoch_batches for position in batch]
    _require(
        sorted(flat_positions) == list(range(len(train_loader.dataset))),
        "training sampler epoch is not an exact permutation",
    )
    generator.set_state(sampler_state)
    replay_batches = tuple(tuple(batch) for batch in train_loader.batch_sampler)
    _require(
        replay_batches == first_epoch_batches,
        "training sampler generator restore is not deterministic",
    )
    _require(
        torch.equal(generator.get_state(), sampler_after),
        "restored sampler continuation state changed",
    )
    generator.set_state(sampler_state)

    raw_moments = {
        "input": _ChannelMoments(),
        "output": _ChannelMoments(),
    }
    normalized_moments = {
        "input": _ChannelMoments(),
        "output": _ChannelMoments(),
    }
    train_pass = _validate_loader_pass(
        loader=train_loader,
        role="ID train",
        expected_ids=expected["train"],
        processor=processor,
        raw_moments=raw_moments,
        normalized_moments=normalized_moments,
    )
    eval_pass = _validate_loader_pass(
        loader=eval_loader,
        role="ID evaluation",
        expected_ids=expected["eval"],
        processor=processor,
    )
    ood_pass = _validate_loader_pass(
        loader=ood_loader,
        role="OOD",
        expected_ids=expected["ood"],
        processor=processor,
    )

    channel_records: list[ChannelStatisticsRecord] = []
    for tensor_role, names, prefix in (
        ("Input", task.input_names, "in_normalizer"),
        ("Output", task.output_names, "out_normalizer"),
    ):
        key = tensor_role.lower()
        raw_mean, raw_std = _finalize_channel_moments(raw_moments[key])
        normalized_mean, normalized_std = _finalize_channel_moments(normalized_moments[key])
        fitted_mean = normalizer_state[f"{prefix}.mean"].reshape(-1).to(torch.float64)
        fitted_std = normalizer_state[f"{prefix}.std"].reshape(-1).to(torch.float64)
        _require(
            torch.allclose(
                raw_mean,
                fitted_mean,
                rtol=_FULL_VALIDATION_RTOL,
                atol=_FULL_VALIDATION_ATOL,
            ),
            f"{tensor_role} train-only fitted means disagree with independent pass",
        )
        _require(
            torch.allclose(
                raw_std,
                fitted_std,
                rtol=_FULL_VALIDATION_RTOL,
                atol=_FULL_VALIDATION_ATOL,
            ),
            f"{tensor_role} train-only fitted scales disagree with independent pass",
        )
        normalizer = processor.in_normalizer if prefix == "in_normalizer" else processor.out_normalizer
        eps = float(normalizer.eps)
        expected_scale = fitted_std / (fitted_std + eps)
        nondegenerate = fitted_std > eps
        _require(
            bool(torch.all(torch.abs(normalized_mean) <= _NORMALIZED_MEAN_ATOL).item()),
            f"{tensor_role} normalized means are not near zero",
        )
        _require(
            torch.allclose(
                normalized_std[nondegenerate],
                expected_scale[nondegenerate],
                rtol=_NORMALIZED_SCALE_RTOL,
                atol=_NORMALIZED_SCALE_ATOL,
            ),
            f"{tensor_role} normalized scales are not near the maintained target",
        )
        for index, name in enumerate(names):
            finite = raw_moments[key].finite and normalized_moments[key].finite
            channel_records.append(
                ChannelStatisticsRecord(
                    tensor_role=tensor_role,
                    channel=name,
                    fitted_mean=float(fitted_mean[index]),
                    fitted_scale=float(fitted_std[index]),
                    normalized_mean=float(normalized_mean[index]),
                    normalized_scale=float(normalized_std[index]),
                    finite=finite,
                    result="PASS",
                )
            )

    restored_processor = datasets.base.data_processor_from_state(
        normalizer_state,
        device="cpu",
    )
    _require(
        all(torch.equal(restored_processor.state_dict()[key], value) for key, value in normalizer_state.items()),
        "persisted normalizer reconstruction changed state",
    )
    zero_variance_state = {key: value.clone() for key, value in normalizer_state.items()}
    zero_variance_state["in_normalizer.std"].zero_()
    zero_variance_state["out_normalizer.std"].zero_()
    zero_variance_processor = datasets.base.data_processor_from_state(
        zero_variance_state,
        device="cpu",
    )
    zero_input_normalizer = zero_variance_processor.in_normalizer
    zero_output_normalizer = zero_variance_processor.out_normalizer
    if zero_input_normalizer is None or zero_output_normalizer is None:
        msg = "Reconstructed zero-variance processor must retain both normalizers."
        raise RuntimeError(msg)
    zero_input = zero_input_normalizer.transform(zero_variance_state["in_normalizer.mean"])
    zero_output = zero_output_normalizer.transform(zero_variance_state["out_normalizer.mean"])
    _require(
        bool(torch.isfinite(zero_input).all().item()) and torch.equal(zero_input, torch.zeros_like(zero_input)),
        "input zero-variance epsilon policy failed",
    )
    _require(
        bool(torch.isfinite(zero_output).all().item()) and torch.equal(zero_output, torch.zeros_like(zero_output)),
        "output zero-variance epsilon policy failed",
    )

    coverage = (
        LoaderCoverageRecord(
            loader="ID train",
            sampler=type(train_loader.sampler).__name__,
            batches=len(train_loader),
            batch_size=train_loader.batch_size,
            final_batch=train_pass.final_batch_size,
            drop_last=train_loader.drop_last,
            inverse_checked=train_pass.inverse_checked,
            finite=True,
            result="PASS",
        ),
        LoaderCoverageRecord(
            loader="ID evaluation",
            sampler=type(eval_loader.sampler).__name__,
            batches=len(eval_loader),
            batch_size=eval_loader.batch_size,
            final_batch=eval_pass.final_batch_size,
            drop_last=eval_loader.drop_last,
            inverse_checked=eval_pass.inverse_checked,
            finite=True,
            result="PASS",
        ),
        LoaderCoverageRecord(
            loader="OOD",
            sampler=type(ood_loader.sampler).__name__,
            batches=len(ood_loader),
            batch_size=ood_loader.batch_size,
            final_batch=ood_pass.final_batch_size,
            drop_last=ood_loader.drop_last,
            inverse_checked=ood_pass.inverse_checked,
            finite=True,
            result="PASS",
        ),
    )
    dataset_membership = (
        _membership_record(
            role="ID full",
            contract=id_contract,
            expected_ids=id_contract.identity.sample_ids,
            observed_ids=id_contract.identity.sample_ids,
            shape=f"x{id_contract.input_shape}, y{id_contract.output_shape}",
            policy="complete identity and metadata artifact binding",
        ),
        _membership_record(
            role="ID train",
            contract=id_contract,
            expected_ids=expected["train"],
            observed_ids=train_pass.observed_ids,
            shape="BCHW",
            policy="RandomSampler permutation",
        ),
        _membership_record(
            role="ID evaluation",
            contract=id_contract,
            expected_ids=expected["eval"],
            observed_ids=eval_pass.observed_ids,
            shape="BCHW",
            policy="SequentialSampler",
        ),
        _membership_record(
            role="OOD full",
            contract=ood_contract,
            expected_ids=ood_contract.identity.sample_ids,
            observed_ids=ood_contract.identity.sample_ids,
            shape=f"x{ood_contract.input_shape}, y{ood_contract.output_shape}",
            policy="complete identity and metadata artifact binding",
        ),
        _membership_record(
            role="OOD selected",
            contract=ood_contract,
            expected_ids=expected["ood"],
            observed_ids=ood_pass.observed_ids,
            shape="BCHW",
            policy="SequentialSampler",
        ),
    )

    first_split = {key: split[key].clone() for key in ("train_indices", "eval_indices", "ood_indices")}
    del built, train_loader, eval_loader, ood_loader, processor, id_contract, ood_contract
    gc.collect()
    repeated = config_loader.create_dataloaders_from_config(
        effective,
        seed_plan=seed_plan,
    )
    for key, values in first_split.items():
        _require(
            torch.equal(repeated["split_indices"][key], values),
            f"repeated production split changed {key}",
        )
    del repeated
    gc.collect()

    elapsed = time.perf_counter() - started
    peak_after_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    overall = (
        ValidationCheck(
            "Production APIs",
            "dataset admission, split, normalizer, loaders, preprocessing",
            "PASS",
        ),
        ValidationCheck(
            "Metadata binding",
            "complete artifact hashes and metadata snapshots admitted",
            "PASS",
        ),
        ValidationCheck(
            "Train-only normalizer",
            "independent float64 complete-pass agreement",
            "PASS",
        ),
        ValidationCheck(
            "Persisted state",
            "split schema and normalizer reconstruction admitted",
            "PASS",
        ),
        ValidationCheck(
            "Zero-variance policy",
            "production epsilon floor returns finite zeros",
            "PASS",
        ),
        ValidationCheck(
            "Sampler restore",
            "captured, advanced, restored, and replayed",
            "PASS",
        ),
        ValidationCheck(
            "Read-only boundary",
            "no run, W&B, checkpoint, or artifact operation",
            "PASS",
        ),
        ValidationCheck("Elapsed seconds", f"{elapsed:.1f}", "INFO"),
        ValidationCheck(
            "Observed process peak GiB",
            f"{peak_after_gib:.2f} (baseline {peak_before_gib:.2f})",
            "INFO",
        ),
    )
    return FullDataValidationResult(
        overall=overall,
        dataset_membership=dataset_membership,
        channels=tuple(channel_records),
        coverage=coverage,
        elapsed_seconds=elapsed,
        peak_gib=peak_after_gib,
    )
