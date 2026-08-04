# ruff: noqa: S101
"""
Protect shared requested-versus-resolved runtime device behavior at every boundary.

Mocked CUDA facts cover CPU silence, ``auto`` selection and fallback, strict
CUDA failure, safe metadata, mixed precision, cleanup, CLI forwarding, and
artifact resolution exactly once. Focused operation probes run only when CUDA is
available. Model numerics and queue selection are covered elsewhere.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import analysis, domain, experiments, learning
from support import configs

if TYPE_CHECKING:
    from pathlib import Path

_MOCK_CUDA_INDEX = 2


def _forbid_cuda_query(*_args: Any, **_kwargs: Any) -> Any:
    """Fail if a CPU-only path touches a CUDA availability/property API."""
    message = "CPU policy queried CUDA"
    raise AssertionError(message)


def _hide_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CUDA deterministically unavailable at the shared resolver seam."""
    monkeypatch.setattr(learning.device.torch.cuda, "is_available", lambda: False)


def _file_inventory(root: Path) -> dict[str, bytes]:
    """Return relative file contents below one mutation boundary."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _mock_cuda_resolution() -> learning.device.DeviceResolution:
    """Return an indexed CUDA runtime fact without querying physical hardware."""
    metadata = learning.device.DeviceRuntimeMetadata(
        requested_policy="cuda",
        resolved_device="cuda:2",
        device_type="cuda",
        pytorch_version=str(torch.__version__),
        cuda_index=_MOCK_CUDA_INDEX,
        cuda_device_name="Mock GPU 2",
    )
    return learning.device.DeviceResolution(
        requested_policy="cuda",
        device=torch.device("cuda:2"),
        device_type="cuda",
        cuda_index=_MOCK_CUDA_INDEX,
        metadata=metadata,
    )


def test_cpu_resolution_is_immutable_serializable_and_cuda_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve explicit CPU while making every CUDA availability/property probe fail.

    The immutable result must serialize requested/resolved CPU facts without CUDA
    fields, protecting genuinely CUDA-silent CPU operation.
    """
    for name in ("is_available", "current_device", "get_device_name"):
        monkeypatch.setattr(learning.device.torch.cuda, name, _forbid_cuda_query)

    resolution = learning.device.resolve_device("cpu", path="run.device")
    metadata = resolution.as_dict()

    assert resolution.requested_policy == "cpu"
    assert resolution.device == torch.device("cpu")
    assert resolution.device_type == "cpu"
    assert resolution.cuda_index is None
    assert metadata["requested_policy"] == "cpu"
    assert metadata["resolved_device"] == "cpu"
    assert not {"cuda_index", "cuda_device_name", "cuda_runtime_version"}.intersection(metadata)
    json.dumps(metadata)


def test_auto_resolution_falls_back_to_cpu_only_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve ``auto`` with CUDA hidden and property probes forbidden.

    Metadata must preserve the requested policy but record concrete CPU, separating
    user intent from the runtime fact used by training.
    """
    _hide_cuda(monkeypatch)
    monkeypatch.setattr(learning.device.torch.cuda, "current_device", _forbid_cuda_query)
    monkeypatch.setattr(learning.device.torch.cuda, "get_device_name", _forbid_cuda_query)

    resolution = learning.device.resolve_device("auto", path="run.device")

    assert resolution.requested_policy == "auto"
    assert resolution.device == torch.device("cpu")
    assert resolution.as_dict()["requested_policy"] == "auto"
    assert resolution.as_dict()["resolved_device"] == "cpu"


def test_auto_resolution_uses_mocked_usable_index_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resolve ``auto`` against mocked availability, current index, and device name.

    The result must be a concrete indexed CUDA device with truthful serializable
    metadata, never an ambiguous unindexed policy token.
    """
    monkeypatch.setattr(learning.device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(learning.device.torch.cuda, "current_device", lambda: _MOCK_CUDA_INDEX)
    monkeypatch.setattr(learning.device.torch.cuda, "get_device_name", lambda index: f"Mock GPU {index}")

    resolution = learning.device.resolve_device("auto", path="run.device")
    metadata = resolution.as_dict()

    assert resolution.requested_policy == "auto"
    assert resolution.device == torch.device("cuda:2")
    assert resolution.device_type == "cuda"
    assert resolution.cuda_index == _MOCK_CUDA_INDEX
    assert metadata["resolved_device"] == "cuda:2"
    assert metadata["cuda_index"] == _MOCK_CUDA_INDEX
    assert metadata["cuda_device_name"] == "Mock GPU 2"
    json.dumps(metadata)


def test_explicit_unavailable_cuda_raises_project_error_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Request strict CUDA while the shared resolver reports it unavailable.

    Resolution must raise the project error with semantic path context and never
    fall back to CPU, preserving the meaning of an explicit hardware requirement.
    """
    _hide_cuda(monkeypatch)

    with pytest.raises(
        learning.device.DeviceResolutionError,
        match=r"run\.device requested 'cuda'.*never falls back",
    ):
        learning.device.resolve_device("cuda", path="run.device")


def test_cuda_availability_query_failure_is_wrapped_or_safely_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Make the CUDA availability query raise a driver-style runtime error.

    Strict CUDA must wrap the unknown state as failure while ``auto`` may safely
    resolve CPU, distinguishing required hardware from fallback policy.
    """

    def fail_availability() -> bool:
        message = "driver probe failed"
        raise RuntimeError(message)

    monkeypatch.setattr(learning.device.torch.cuda, "is_available", fail_availability)

    with pytest.raises(learning.device.DeviceResolutionError, match="availability could not be established"):
        learning.device.resolve_device("cuda", path="run.device")
    assert learning.device.resolve_device("auto", path="run.device").device == torch.device("cpu")


def test_invalid_device_values_fail_at_the_exact_semantic_path() -> None:
    """
    Vary an unknown string, indexed device, boolean, number, and null.

    Every invalid family must fail at ``run.device`` while the base YAML remains
    fixed, proving config accepts only ``auto``, ``cuda``, or ``cpu`` strings.
    """
    invalid_values: tuple[Any, ...] = ("unsupported", "cuda:0", True, 0, None)
    for invalid in invalid_values:
        raw = configs.direct_config()
        raw["run"]["device"] = invalid
        with pytest.raises(
            experiments.config.loader.ConfigError,
            match=r"run\.device must be exactly one of: auto, cuda, cpu",
        ):
            experiments.config.loader.resolve_config(raw)


def test_cpu_mixed_precision_is_rejected_before_scaler_construction() -> None:
    """
    Validate mixed precision against one resolved CPU device.

    The boundary must reject it before scaler construction because the maintained
    training contract implements AMP only for CUDA.
    """
    resolution = learning.device.resolve_device("cpu")
    with pytest.raises(learning.device.DeviceResolutionError, match="CPU autocast is not supported"):
        learning.device.validate_mixed_precision_device(True, resolution)


def test_artifact_cpu_cleanup_does_not_touch_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run artifact cleanup on CPU while CUDA cache and IPC calls are forbidden.

    Cleanup must complete without touching CUDA runtime state, preserving CPU-only
    operation even after artifact generation.
    """
    monkeypatch.setattr(analysis.artifacts.service.torch.cuda, "empty_cache", _forbid_cuda_query)
    monkeypatch.setattr(analysis.artifacts.service.torch.cuda, "ipc_collect", _forbid_cuda_query)

    analysis.artifacts.service.cleanup_runtime(torch.device("cpu"))


def test_cuda_determinism_policy_is_operation_aware_and_never_mutates_config() -> None:
    """Reject known strict CUDA conflicts while admitting supported effective operations."""
    cuda = _mock_cuda_resolution()
    uno_raw = configs.direct_config(model_kind="uno", physics_enabled=False)
    uno_raw["run"]["deterministic"] = True
    uno = experiments.config.loader.resolve_config(uno_raw)
    with pytest.raises(
        learning.device.DeviceResolutionError,
        match=r"(?s)UNO 2D resampling.*bicubic.*run\.deterministic: false.*best-effort CUDA reproducibility",
    ):
        experiments.run.validate_deterministic_model_device_policy(uno, cuda)
    assert uno["run"]["deterministic"] is True

    reflected_raw = configs.direct_config(model_kind="fno", physics_enabled=True)
    reflected_raw["run"]["deterministic"] = True
    reflected_raw["loss"]["physics"]["derivatives"] = {
        "kind": "spectral",
        "extension": "reflect",
    }
    reflected = experiments.config.loader.resolve_config(reflected_raw)
    with pytest.raises(
        learning.device.DeviceResolutionError,
        match=r"(?s)spectral physics.*extension 'reflect'.*reflection_pad2d_backward_cuda.*will not change",
    ):
        experiments.run.validate_deterministic_model_device_policy(reflected, cuda)
    assert reflected["run"]["deterministic"] is True
    assert reflected["loss"]["physics"]["derivatives"] == {
        "kind": "spectral",
        "extension": "reflect",
    }

    for config in (uno, reflected):
        config["run"]["deterministic"] = False
        experiments.run.validate_deterministic_model_device_policy(config, cuda)
        config["run"]["deterministic"] = True
        experiments.run.validate_deterministic_model_device_policy(config, learning.device.resolve_device("cpu"))

    data_only = experiments.config.loader.resolve_config(
        configs.direct_config(model_kind="fno", physics_enabled=False),
    )
    physical = experiments.config.loader.resolve_config(
        configs.direct_config(model_kind="fno", physics_enabled=True),
    )
    experiments.run.validate_deterministic_model_device_policy(data_only, cuda)
    experiments.run.validate_deterministic_model_device_policy(physical, cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("reflected-spectral-physics", "reflection_pad2d_backward_cuda"),
        ("uno-bicubic-resampling", "upsample_bicubic2d_backward"),
    ],
)
def test_known_cuda_operations_reject_strict_deterministic_backward(
    operation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the project derivative and maintained UNO model CUDA paths."""
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    try:
        if operation == "reflected-spectral-physics":
            value = torch.randn(1, 2, 8, 10, device="cuda", requires_grad=True)
            derivative_x, derivative_y = domain.physics.derivatives.SpectralDerivatives(
                extension="reflect",
            ).gradient(value, 1.0, 1.0)
            objective = derivative_x.square().mean() + derivative_y.square().mean()
        else:
            config = experiments.config.loader.resolve_config(
                configs.direct_config(model_kind="uno", physics_enabled=False),
            )
            model = learning.models.factory.build_model(config, device=torch.device("cuda:0"))
            params = config["model"]["params"]
            in_channels = int(params["in_channels"])
            height = int(params["modes_y"])
            width = int(params["modes_x"])
            value = torch.randn(1, in_channels, height, width, device="cuda")
            objective = model(value).square().mean()
        with pytest.raises(RuntimeError, match=message):
            objective.backward()
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)


@pytest.mark.parametrize(
    ("model_kind", "reflected_spectral_physics"),
    [("uno", False), ("fno", True)],
    ids=("uno-bicubic", "reflected-spectral-physics"),
)
def test_known_cuda_determinism_conflicts_fail_before_allocation_or_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_kind: str,
    reflected_spectral_physics: bool,
) -> None:
    """Reject known conflicts before run locks, files, allocation, or W&B initialization."""
    raw = configs.direct_config(
        model_kind=model_kind,
        physics_enabled=True,
    )
    raw["run"]["device"] = "cuda"
    raw["run"]["deterministic"] = True
    if reflected_spectral_physics:
        raw["loss"]["physics"]["derivatives"] = {
            "kind": "spectral",
            "extension": "reflect",
        }
    allocation_attempted = False
    tracking_attempted = False

    def reject_allocation(_run_dir: Path | str) -> Path:
        nonlocal allocation_attempted
        allocation_attempted = True
        pytest.fail("unsupported deterministic request must not allocate")

    def reject_tracking(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal tracking_attempted
        tracking_attempted = True
        pytest.fail("unsupported deterministic request must not initialize W&B")

    monkeypatch.setattr(experiments.run.config_loader, "load_yaml", lambda _path: raw)
    monkeypatch.setattr(experiments.run.learning.device, "resolve_device", lambda *_args, **_kwargs: _mock_cuda_resolution())
    monkeypatch.setattr(experiments.run, "allocate_run_directory", reject_allocation)
    monkeypatch.setattr(experiments.run.tracking, "initialize_wandb", reject_tracking)
    output_root = tmp_path / "outputs"

    with pytest.raises(learning.device.DeviceResolutionError, match=r"Set run\.deterministic: false"):
        experiments.run.run_experiment(f"unsupported-{model_kind}.yaml", output_root=output_root)

    assert allocation_attempted is False
    assert tracking_attempted is False
    assert not output_root.exists()


def test_strict_cuda_fails_before_fresh_training_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Start a fresh run with strict CUDA hidden and an unused output root.

    Resolution must fail before allocating run or lock paths, preventing a hardware
    policy error from leaving partial lifecycle state.
    """
    _hide_cuda(monkeypatch)
    output_root = tmp_path / "outputs"

    config_path = configs.write_yaml(
        tmp_path / "request.yaml",
        configs.direct_config(),
    )
    with pytest.raises(learning.device.DeviceResolutionError):
        experiments.run.run_experiment(
            config_path,
            device="cuda",
            output_root=output_root,
        )

    assert not output_root.exists()


def test_strict_cuda_does_not_mutate_an_existing_resume_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Resume an existing marker-only run with strict CUDA unavailable.

    The byte inventory and lease directory must remain unchanged, proving device
    resolution precedes every mutation of an existing run.
    """
    _hide_cuda(monkeypatch)
    training_root = tmp_path / "training"
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(training_root))
    run_dir = tmp_path / "resume"
    run_dir.mkdir()
    (run_dir / "marker.bin").write_bytes(b"unchanged")
    before = _file_inventory(run_dir)

    config_path = configs.write_yaml(
        tmp_path / "request.yaml",
        configs.direct_config(),
    )
    with pytest.raises(learning.device.DeviceResolutionError):
        experiments.run.run_experiment(
            config_path,
            resume=run_dir,
            device="cuda",
        )

    assert _file_inventory(run_dir) == before
    assert not (training_root / ".state").exists()
    assert not list(run_dir.rglob("*.lock"))


def test_inference_strict_cuda_fails_before_saved_run_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Request strict CUDA inference for a deliberately absent run directory.

    Device failure must precede saved-run admission, establishing one predictable
    resolution boundary before filesystem or model reconstruction work.
    """
    _hide_cuda(monkeypatch)

    with pytest.raises(learning.device.DeviceResolutionError):
        learning.inference.context.load_inference_context(
            run_dir=tmp_path / "missing-run",
            device_policy="cuda",
        )


def test_artifact_strict_cuda_fails_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Request strict CUDA artifact rebuild around an existing marker-only root.

    Failure must leave every byte unchanged, ensuring destructive rebuild decisions
    occur only after a usable concrete device exists.
    """
    _hide_cuda(monkeypatch)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "marker.bin").write_bytes(b"unchanged")
    before = _file_inventory(runs_root)

    with pytest.raises(learning.device.DeviceResolutionError):
        analysis.artifacts.service.build_artifacts(
            runs_root=runs_root,
            dataset_root=tmp_path / "datasets",
            device_policy="cuda",
            rebuild=True,
        )

    assert _file_inventory(runs_root) == before


def test_artifact_boundary_reuses_one_resolution_for_both_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Build ID and OOD artifacts through stubs that record resolution object identity.

    Both splits must receive the same concrete resolution and cleanup its device
    once, preventing repeated policy resolution or cross-stage device drift.
    """
    run_dir = tmp_path / "run"
    plan = analysis.artifacts.service.RunArtifactPlan(
        run_dir=run_dir,
        id_dataset_name="id",
        ood_dataset_name="ood",
        lifecycle_status="completed",
        is_completed=True,
        scientific_run_name=run_dir.name,
    )
    resolutions: list[learning.device.DeviceResolution] = []
    cleanup_devices: list[torch.device] = []

    monkeypatch.setattr(analysis.artifacts.service, "iter_run_dirs", lambda *_args, **_kwargs: [run_dir])
    monkeypatch.setattr(analysis.artifacts.service, "load_run_artifact_plan", lambda _run_dir: plan)

    def capture_artifacts(**kwargs: Any) -> object:
        resolutions.append(kwargs["device_resolution"])
        return object()

    monkeypatch.setattr(analysis.artifacts.service, "run_or_load_artifacts", capture_artifacts)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", cleanup_devices.append)
    monkeypatch.setattr(
        analysis.artifacts.service,
        "_upload_published_artifacts",
        lambda **_kwargs: None,
    )

    result = analysis.artifacts.service.build_artifacts(
        runs_root=tmp_path,
        dataset_root=tmp_path / "datasets",
        device_policy="cpu",
    )

    assert set(result["run"]) == {"eval", "ood"}
    assert len(resolutions) == len(("eval", "ood"))
    assert resolutions[0] is resolutions[1]
    assert resolutions[0].device == torch.device("cpu")
    assert cleanup_devices == [torch.device("cpu")]
