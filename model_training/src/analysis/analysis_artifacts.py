"""
===============================================================================
analysis_artifacts.py
===============================================================================
Create persistent evaluation artifacts from trained neural-operator runs.

Responsibilities:
  - Run deterministic inference over explicit evaluation loaders
  - Write Parquet scalar metrics and NPZ field artifacts
  - Compute field, physics, boundary and metadata diagnostics
  - Keep artifact columns stable for downstream analysis modules

Design principles:
  - Artifacts are reproducible and split-aware
  - Physical units and the caller-provided task field order are explicit
  - Heavy inference work stays out of plotting modules
  - Saved-run detection uses the current run artifact contract

Boundaries:
  - Model reconstruction belongs to learning.inference.context
  - Interactive visualization belongs to analysis.evaluation and analysis.ui
  - Saved outputs follow the strict current artifact schema and provenance contract

Notes:
  Artifact contents:
    - Parquet stores case-level scalar metrics, artifact paths and JSON-safe metadata
    - NPZ stores predictions, targets, errors, kappa fields, raw tensors and residual fields
    - residual metrics use the same canonical evaluation crop for model comparability
  Schema:
    Artifacts
    ---------
    Parquet (global, per case):
        - case_index        : stable one-based source case id
        - source_index      : zero-based original merged-dataset index
        - split_local_index : zero-based position in the saved split
        - npz_path          : path to the corresponding NPZ artifact

        # Relative field errors (dimensionless, channel-balanced)
        - rel_l2            : channel-balanced mean relative L2 on [p, u, v] over the full domain
        - rel_h1            : channel-balanced mean relative H1 on [p, u, v] over the interior (cropped by EVAL_PAD)

        # Book-friendly metrics (physical units, intuitive)
        - rmse_p            : RMSE of pressure p over the full domain
        - rmse_u            : RMSE of x-velocity u over the full domain
        - rmse_v            : RMSE of y-velocity v over the full domain
        - rmse_U            : RMSE of speed magnitude U=sqrt(u^2+v^2) over the full domain

        # Physics residual metrics (interior-cropped by EVAL_PAD)
        - mom_mse           : MSE of Brinkman momentum residual
        - cont_mse          : MSE of the task-selected continuity residual computed with spectral reflect derivatives
        # Boundary condition metric (no crop, evaluated on inlet/outlet masks)
        - bc_mse            : pressure BC mismatch on inlet/outlet boundaries

        # Diagnostics
        - kappa_names       : list of available permeability tensor components
        - meta              : JSON-safe metadata dictionary (stored as JSON string)

    NPZ (local, full fields per case):
        - case_index   : stable one-based source case id
        - source_index : zero-based original merged-dataset index
        - split_local_index : zero-based position in the saved split
        - pred         : (C_artifact, H, W) prediction aligned with artifact_fields
        - gt           : (C_artifact, H, W) ground truth aligned with artifact_fields
        - err          : (C_artifact, H, W) prediction error (pred - gt)
        - artifact_fields : list[str] names aligned with pred/gt/err
        - artifact_units  : list[str] physical units aligned with artifact_fields

        - kappa_encoded: (C_kappa, H, W) task-stored permeability representations
        - kappa        : (C_kappa, H, W) physical permeability components
        - kappa_names  : list[str], same order as kappa channels
        - p_bc         : (1, H, W) pressure boundary condition

        # Declared inputs and targets retained for downstream analysis
        - x_raw        : (C_in, H, W) raw input tensor (physical units)
        - y_raw        : (C_out, H, W) raw target tensor (physical units)
        - input_fields : list[str] canonical input channel names
        - output_fields: list[str] canonical learned-output names aligned with y_raw
        - output_units : list[str] physical units aligned with output_fields

        # Physics diagnostic fields (full fields, not cropped)
        - Rx           : (H, W) x-momentum residual field
        - Ry           : (H, W) y-momentum residual field
        - Rc           : (H, W) task-selected continuity residual field
        - div_u        : (H, W) divergence field div(u)
        - div_eps_u    : (H, W) divergence field div(eps u)

        - meta         : JSON string with full metadata
===============================================================================

"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import common, domain

# ============================================================================
# Global constants
# ============================================================================

INTERNAL_KAPPA_NAMES = set(domain.permeability.INTERNAL_KAPPA_2D_ORDER) | set(domain.permeability.INTERNAL_KAPPA_3D_ORDER)
MU_AIR = 1.8139e-5  # must be consistent with training
ARTIFACT_SCHEMA_VERSION = 3
ARTIFACT_PROVENANCE_SCHEMA_VERSION = 2
ARTIFACT_PROVENANCE_FILENAME = "artifact_provenance.json"

# ----------------------------------------------------------------------------
# Canonical evaluation settings (for model-to-model comparability)
# ----------------------------------------------------------------------------
EVAL_PAD = 2  # crop for ALL gradient-based metrics (H1 + physics)

# =============================================================================
# JSON / type normalisation utilities
# =============================================================================


def meta_to_jsonable(obj: Any) -> Any:
    """
    Convert tensors, numpy values and nested structures into JSON-safe types.

    Rules
    -----
    - torch.Tensor      -> float (0-d) or list
    - numpy.ndarray     -> float (0-d) or list
    - numpy scalar      -> Python scalar
    - dict / list       -> recursively processed

    This function guarantees that the returned object can be safely
    serialised via `json.dumps`.
    """
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
        return float(arr) if arr.ndim == 0 else arr.tolist()

    if isinstance(obj, np.ndarray):
        return float(obj) if obj.ndim == 0 else obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


def ordered_indices_sha256(indices: Iterable[int]) -> str:
    """Return the canonical SHA-256 digest for ordered integer membership."""
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_provenance_path(save_root: str | Path) -> Path:
    """Return the versioned provenance sidecar path for an artifact root."""
    return Path(save_root) / ARTIFACT_PROVENANCE_FILENAME


def artifact_output_manifest(save_root: str | Path) -> dict[str, Any]:
    """Return the exact digest manifest for one complete artifact payload."""
    root = Path(save_root)
    parquet_files = sorted(root.glob("*.parquet"))
    npz_files = sorted((root / "npz").glob("*.npz"))
    if len(parquet_files) != 1:
        msg = f"Artifact payload must contain exactly one Parquet file, found {len(parquet_files)} in {root}."
        raise RuntimeError(msg)
    if not npz_files:
        msg = f"Artifact payload contains no NPZ files: {root / 'npz'}"
        raise RuntimeError(msg)

    def entry(payload_path: Path) -> dict[str, Any]:
        return {
            "path": payload_path.relative_to(root).as_posix(),
            "sha256": common.serialization.file_sha256(payload_path),
        }

    return {
        "parquet": entry(parquet_files[0]),
        "npz": [entry(payload_path) for payload_path in npz_files],
    }


def write_artifact_provenance(save_root: str | Path, provenance: Mapping[str, Any]) -> Path:
    """Atomically publish provenance and payload digests as the completion marker."""
    provenance_path = artifact_provenance_path(save_root)
    if provenance_path.exists():
        msg = f"Refusing to overwrite existing artifact provenance: {provenance_path}"
        raise FileExistsError(msg)

    payload = meta_to_jsonable(dict(provenance))
    if not isinstance(payload, dict):
        msg = "Artifact provenance must normalise to a JSON object."
        raise TypeError(msg)
    if "outputs" in payload:
        msg = "Caller-provided artifact provenance cannot define output digests."
        raise ValueError(msg)
    payload["outputs"] = artifact_output_manifest(save_root)

    return common.serialization.atomic_write_json(provenance_path, payload)


def _require_batch_scalar_int(batch: Mapping[str, Any], key: str) -> int:
    """Return one integer identity value from a batch-size-one collated field."""
    if key not in batch:
        msg = f"Artifact batches must provide top-level {key!r} identity."
        raise KeyError(msg)

    value = batch[key]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            msg = f"Artifact batch {key!r} must contain exactly one integer value; got {value!r}."
            raise TypeError(msg)
        return int(value.detach().cpu().item())

    if isinstance(value, np.ndarray):
        if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
            msg = f"Artifact batch {key!r} must contain exactly one integer value; got {value!r}."
            raise TypeError(msg)
        return int(value.reshape(-1)[0])

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            msg = f"Artifact batch {key!r} must contain exactly one integer value; got {value!r}."
            raise TypeError(msg)
        value = value[0]

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"Artifact batch {key!r} must be an integer; got {type(value).__name__}."
        raise TypeError(msg)
    return int(value)


def _require_finite_artifact_tensor(value: torch.Tensor, *, label: str) -> None:
    """Reject non-floating or non-finite tensors before artifact publication."""
    if not value.is_floating_point() or value.is_complex():
        msg = f"{label} must be a real floating-point tensor."
        raise TypeError(msg)
    if not torch.isfinite(value).all():
        msg = f"{label} contains non-finite values."
        raise FloatingPointError(msg)


def _artifact_effective_case_count(provenance: Mapping[str, Any]) -> int:
    """Return the required generated row count from validated provenance."""
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping):
        msg = "Artifact provenance must contain a selection mapping."
        raise TypeError(msg)

    count = selection.get("effective_case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        msg = "Artifact provenance selection.effective_case_count must be a positive integer."
        raise ValueError(msg)
    return count


def _validate_generated_source_indices(provenance: Mapping[str, Any], source_indices: list[int]) -> None:
    """Prove generated loader identity matches the provenance before completion."""
    selection = provenance.get("selection")
    if not isinstance(selection, Mapping):
        msg = "Artifact provenance must contain a selection mapping."
        raise TypeError(msg)
    expected_digest = selection.get("effective_ordered_source_indices_sha256")
    if not isinstance(expected_digest, str) or not expected_digest:
        msg = "Artifact provenance must contain an effective ordered source-index digest."
        raise TypeError(msg)
    actual_digest = ordered_indices_sha256(source_indices)
    if actual_digest != expected_digest:
        msg = f"Generated ordered source_index values do not match artifact provenance: expected digest {expected_digest}, got {actual_digest}."
        raise RuntimeError(msg)


def _ensure_artifact_targets_absent(save_root: Path, dataset_name: str) -> None:
    """Refuse to overwrite complete or interrupted artifact outputs."""
    npz_dir = save_root / "npz"
    candidates = [
        save_root / f"{dataset_name}.parquet",
        artifact_provenance_path(save_root),
        *sorted(npz_dir.glob("*.npz")),
        *sorted(save_root.glob(".*.tmp")),
        *sorted(npz_dir.glob(".*.tmp")),
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        msg = f"Refusing to overwrite existing or interrupted artifacts:\n{formatted}"
        raise FileExistsError(msg)


# =============================================================================
# Kappa field utilities (fields only, no scalar statistics)
# =============================================================================


def detect_kappa_channels_from_inputs(include_inputs: list[str]) -> list[str]:
    """
    Detect permeability-related input channels based on their names.

    Parameters
    ----------
    include_inputs : list[str]
        List of canonical input channel names.

    Returns
    -------
    list[str]
        Names of all channels that represent permeability components.

    """
    return [name for name in include_inputs if name in INTERNAL_KAPPA_NAMES]


def extract_kappa(
    x_tensor: torch.Tensor,
    *,
    input_fields: list[str],
    kappa_names: list[str],
) -> dict[str, torch.Tensor]:
    """
    Extract task-encoded and physical permeability fields from the input tensor.

    This function handles the case where no permeability channels are
    present by returning empty tensors with the correct shape.

    Parameters
    ----------
    x_tensor : torch.Tensor
        Input tensor of shape (B, C_in, H, W).
    input_fields : list[str]
        Canonical list of input channel names.
    kappa_names : list[str]
        Names of kappa components to extract.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
            - "kappa_encoded"
            - "kappa"

    """
    if not kappa_names:
        return {
            "kappa_encoded": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
            "kappa": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
        }

    index_map = {name: i for i, name in enumerate(input_fields)}
    kappa_indices = [index_map[name] for name in kappa_names]

    # Task-encoded permeability representations as stored in the dataset.
    kappa_encoded = x_tensor[:, kappa_indices, :, :]

    # Physical permeability reconstruction in square metres.
    kappa_phys = torch.zeros_like(kappa_encoded)
    name_to_pos = {name: i for i, name in enumerate(kappa_names)}

    # kxx, kyy (always log10-physical)
    kxx = torch.pow(10.0, kappa_encoded[:, name_to_pos["kxx"]])
    kyy = torch.pow(10.0, kappa_encoded[:, name_to_pos["kyy"]])
    kappa_phys[:, name_to_pos["kxx"]] = kxx
    kappa_phys[:, name_to_pos["kyy"]] = kyy

    # kxy is a dimensionless ratio to sqrt(kxx * kyy).
    if "kxy" in name_to_pos:
        kxy_ratio = kappa_encoded[:, name_to_pos["kxy"]]
        kappa_phys[:, name_to_pos["kxy"]] = kxy_ratio * torch.sqrt(kxx * kyy)

    return {
        "kappa_encoded": kappa_encoded,
        "kappa": kappa_phys,
    }


# =============================================================================
# Run-directory utilities
# =============================================================================


def infer_current_run_dir(save_root: Path) -> Path:
    """
    Infer the current-contract run directory from an artifact save root.

    This is used only for artifact metadata/logging. Artifact generation itself
    continues to consume the explicit model, loader, processor and device passed
    by callers.
    """
    candidate = Path(save_root)
    while candidate.parent != candidate:
        if common.paths.is_current_run_dir(candidate):
            return candidate
        candidate = candidate.parent
    return Path(save_root)


# =============================================================================
# Main artifact generator
# =============================================================================


def _generate_steady_flow_artifacts(  # noqa: PLR0915
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Run inference on all cases and generate persistent evaluation artifacts.

    For each case:
        - perform a forward pass with the trained model
        - compute channel-balanced relative L2/H1 and physical-unit per-field RMSE metrics
        - store full spatial fields in an NPZ file
        - store scalar metrics and metadata in a Parquet table

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated task contract owning exact input channel names and order.
    model : Any
        Trained neural operator model (FNO, PINO, etc.).
    loader : Iterable[Mapping[str, Any]]
        Deterministic iterable of evaluation batches.
    processor : Any
        Normalisation processor used during training.
    device : torch.device
        Device used for inference.
    save_root : str or Path
        Root directory for all generated artifacts.
    dataset_name : str
        Base name for the Parquet summary file.
    provenance : Mapping[str, Any]
        Exact versioned generation contract. Written only after all payloads
        complete successfully.
    max_cases : int or None, optional
        Maximum number of cases to process. If None, process all cases.

    Returns
    -------
    df : pandas.DataFrame
        Per-case summary table.
    parquet_path : pathlib.Path
        Path to the written Parquet file.

    """
    save_root = Path(save_root)
    expected_case_count = _artifact_effective_case_count(provenance)
    _ensure_artifact_targets_absent(save_root, dataset_name)
    model.eval()

    # Infer run_dir from save_root for logging/metadata only.
    run_dir = infer_current_run_dir(save_root)
    run_name = run_dir.name
    physics_variant = f"{task.physics.continuity}-spectral-reflect"

    print(
        "[ARTIFACTS]",
        f"save_root={save_root}",
        f"run_dir={run_dir}",
        f"run_name={run_name}",
        f"variant={physics_variant}",
        sep="\n  - ",
    )

    # --------------------------------------------------
    # Build domain-owned residual diagnostics
    # --------------------------------------------------
    physics_evaluator = domain.physics.brinkman.resolve_physics_evaluator(task.physics.kind)
    derivative_operator = domain.physics.derivatives.build_derivative_operator(
        "spectral",
        extension="reflect",
    )
    print(f"[ARTIFACTS] Using domain physics diagnostics kind={task.physics.kind} derivatives=spectral/reflect pad={EVAL_PAD}")

    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    generated_source_indices: list[int] = []

    # Detect available kappa channels from field contracts
    kappa_names = detect_kappa_channels_from_inputs(list(task.input_names))

    for idx, batch in enumerate(loader):
        if max_cases is not None and idx >= max_cases:
            break
        split_local_index = _require_batch_scalar_int(batch, "split_local_index")
        source_index = _require_batch_scalar_int(batch, "source_index")
        if split_local_index != idx:
            msg = f"Artifact loader order does not match saved split-local identity: iteration={idx}, split_local_index={split_local_index}."
            raise RuntimeError(msg)
        if source_index < 0:
            msg = f"Artifact source_index must be non-negative, got {source_index}."
            raise ValueError(msg)
        case_id = source_index + 1
        generated_source_indices.append(source_index)

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        _require_finite_artifact_tensor(x, label=f"Artifact case {case_id} inputs")
        _require_finite_artifact_tensor(y, label=f"Artifact case {case_id} targets")

        # Preserve generator metadata in a JSON-safe form.
        source_meta = meta_to_jsonable(batch.get("meta", {}))
        meta_clean = dict(source_meta) if isinstance(source_meta, dict) else {"source_meta": source_meta}
        reserved_meta_keys = {"case_index", "source_index", "split_local_index"}.intersection(meta_clean)
        if reserved_meta_keys:
            msg = f"Source metadata contains reserved artifact identity keys and cannot be preserved unambiguously: {sorted(reserved_meta_keys)}."
            raise KeyError(msg)
        # Pressure boundary condition (stored for diagnostics)
        p_bc_idx = task.input_names.index("p_bc")
        p_bc = x[:, p_bc_idx : p_bc_idx + 1].detach().cpu()

        # Permeability fields (no scalar stats here)
        kappa_info = extract_kappa(
            x,
            input_fields=list(task.input_names),
            kappa_names=kappa_names,
        )

        # --------------------------------------------------
        # Forward pass (model operates in normalized space)
        # --------------------------------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda" and start_time is not None:
            start_time.record(torch.cuda.current_stream())

        with torch.no_grad():
            x_norm = processor.in_normalizer.transform(x)
            y_hat_norm = model(x_norm)
            y_hat = processor.out_normalizer.inverse_transform(y_hat_norm)

        if device.type == "cuda" and end_time is not None:
            end_time.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            inference_time_ms = start_time.elapsed_time(end_time) if start_time is not None else None
        else:
            inference_time_ms = None

        _require_finite_artifact_tensor(x_norm, label=f"Artifact case {case_id} normalized inputs")
        _require_finite_artifact_tensor(y_hat_norm, label=f"Artifact case {case_id} normalized prediction")
        _require_finite_artifact_tensor(y_hat, label=f"Artifact case {case_id} physical prediction")
        if y_hat.shape != y.shape:
            msg = f"Artifact prediction/target shape mismatch: {tuple(y_hat.shape)} != {tuple(y.shape)}."
            raise RuntimeError(msg)

        # --------------------------------------------------
        # Physics diagnostics (exact training-consistent implementation)
        # --------------------------------------------------
        with torch.no_grad():
            diag = physics_evaluator(
                x,
                y_hat,
                input_fields=task.input_names,
                output_fields=task.output_names,
                derivatives=derivative_operator,
                continuity=task.physics.continuity,
                boundary=task.physics.boundary,
                interior_crop=EVAL_PAD,
            ).as_dict()
        for diagnostic_name, diagnostic_value in diag.items():
            _require_finite_artifact_tensor(
                diagnostic_value,
                label=f"Artifact case {case_id} physics diagnostic {diagnostic_name}",
            )

        mom_mse = float(diag["mom_mse"].detach().cpu().item())
        bc_mse = float(diag["bc_mse"].detach().cpu().item())

        cont_mse = float(diag["cont_mse"].detach().cpu().item())
        Rx_np = diag["Rx"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Ry_np = diag["Ry"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Rc_np = diag["Rc"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divu_np = diag["div_u"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divepsu_np = diag["div_eps_u"].detach().cpu().squeeze(0).squeeze(0).numpy()

        # --------------------------------------------------
        # Outputs (de-normalised, physical units)
        # --------------------------------------------------
        p, u, v = y_hat[:, 0:1], y_hat[:, 1:2], y_hat[:, 2:3]
        p_gt, u_gt, v_gt = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        U = torch.sqrt(u**2 + v**2)
        U_gt = torch.sqrt(u_gt**2 + v_gt**2)

        # Book-friendly metrics (physical units)
        rmse_p = torch.sqrt(torch.mean((p - p_gt) ** 2)).item()
        rmse_u = torch.sqrt(torch.mean((u - u_gt) ** 2)).item()
        rmse_v = torch.sqrt(torch.mean((v - v_gt) ** 2)).item()
        rmse_U = torch.sqrt(torch.mean((U - U_gt) ** 2)).item()

        # Full tensors for NPZ export / plotting (includes U)
        y_hat_ext = torch.cat([p, u, v, U], dim=1)
        y_ext = torch.cat([p_gt, u_gt, v_gt, U_gt], dim=1)
        err_ext = y_hat_ext - y_ext

        # Main metric tensors (ONLY p,u,v)
        y_hat_main = torch.cat([p, u, v], dim=1)
        y_main = torch.cat([p_gt, u_gt, v_gt], dim=1)
        err_main = y_hat_main - y_main

        # Grid spacing from coordinate fields (physical)
        idx_x = task.input_names.index("x")
        idx_y = task.input_names.index("y")
        dx = float((x[0, idx_x, 0, 1] - x[0, idx_x, 0, 0]).abs().detach().cpu().item())
        dy = float((x[0, idx_y, 1, 0] - x[0, idx_y, 0, 0]).abs().detach().cpu().item())

        metric_denominator_floor = 1e-12

        # ------------------------------------------------------------------
        # Dimensionless relative L2/H1, normalized independently per field
        # ------------------------------------------------------------------
        rel_l2_per_channel: list[float] = []
        rel_h1_per_channel: list[float] = []

        for c in range(y_main.shape[1]):  # c in {p,u,v}
            e_c = err_main[:, c : c + 1]
            r_c = y_main[:, c : c + 1]

            # Relative L2 per channel (global norm ratio)
            l2_e_c = torch.linalg.norm(e_c)
            l2_r_c = torch.linalg.norm(r_c)
            rel_l2_c = (l2_e_c / (l2_r_c + metric_denominator_floor)).item()
            rel_l2_per_channel.append(float(rel_l2_c))

            # Relative H1 per channel (interior, with gradients)
            de_dy_c, de_dx_c = torch.gradient(e_c, spacing=(dy, dx), dim=(2, 3))
            dr_dy_c, dr_dx_c = torch.gradient(r_c, spacing=(dy, dx), dim=(2, 3))

            if EVAL_PAD > 0:
                e_i = e_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                de_dx_i = de_dx_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                de_dy_i = de_dy_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]

                r_i = r_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                dr_dx_i = dr_dx_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
                dr_dy_i = dr_dy_c[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
            else:
                e_i, de_dx_i, de_dy_i = e_c, de_dx_c, de_dy_c
                r_i, dr_dx_i, dr_dy_i = r_c, dr_dx_c, dr_dy_c

            h1_e_c = torch.sqrt((e_i.pow(2) + de_dx_i.pow(2) + de_dy_i.pow(2)).mean())
            h1_r_c = torch.sqrt((r_i.pow(2) + dr_dx_i.pow(2) + dr_dy_i.pow(2)).mean())

            rel_h1_c = (h1_e_c / (h1_r_c + metric_denominator_floor)).item()
            rel_h1_per_channel.append(float(rel_h1_c))

        rel_l2 = float(np.mean(rel_l2_per_channel))
        rel_h1 = float(np.mean(rel_h1_per_channel))
        scalar_metrics = (rmse_p, rmse_u, rmse_v, rmse_U, rel_l2, rel_h1, mom_mse, cont_mse, bc_mse)
        if not np.isfinite(np.asarray(scalar_metrics, dtype=float)).all():
            msg = f"Artifact case {case_id} produced non-finite scalar metrics."
            raise FloatingPointError(msg)

        # --------------------------------------------------
        # Write NPZ artifact
        # --------------------------------------------------
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        if npz_path.exists():
            msg = f"Refusing to overwrite an existing NPZ artifact: {npz_path}"
            raise FileExistsError(msg)
        x_raw = x.squeeze(0).detach().cpu().numpy()  # (C_in,H,W)
        y_raw = y.squeeze(0).detach().cpu().numpy()  # (C_out,H,W)

        artifact_fields = (*task.output_names, "U")
        artifact_units = (*(field.unit for field in task.outputs), "m/s")
        npz_payload = {
            "case_index": np.int64(case_id),
            "source_index": np.int64(source_index),
            "split_local_index": np.int64(split_local_index),
            "pred": y_hat_ext.squeeze(0).cpu().numpy(),
            "gt": y_ext.squeeze(0).cpu().numpy(),
            "err": err_ext.squeeze(0).cpu().numpy(),
            "artifact_fields": np.asarray(artifact_fields),
            "artifact_units": np.asarray(artifact_units),
            "kappa_encoded": kappa_info["kappa_encoded"].squeeze(0).cpu().numpy(),
            "kappa": kappa_info["kappa"].squeeze(0).cpu().numpy(),
            "kappa_names": np.asarray(kappa_names),
            "p_bc": p_bc.squeeze(0).numpy(),
            "meta": json.dumps(meta_clean),
            "x_raw": x_raw,
            "y_raw": y_raw,
            "input_fields": np.asarray(task.input_names),
            "output_fields": np.asarray(task.output_names),
            "output_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "Rx": Rx_np,
            "Ry": Ry_np,
            "Rc": Rc_np,
            "div_u": divu_np,
            "div_eps_u": divepsu_np,
        }

        def write_npz(temp_path: Path, content: dict[str, Any] = npz_payload) -> None:
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, **content)

        common.serialization.atomic_path_write(npz_path, write_npz)

        # --------------------------------------------------
        # Parquet row (scalar metrics + metadata only)
        # --------------------------------------------------
        rows.append(
            {
                "inference_time_ms": inference_time_ms,
                "case_index": case_id,
                "source_index": source_index,
                "split_local_index": split_local_index,
                "npz_path": str(npz_path),
                "rel_l2": rel_l2,
                "rel_h1": rel_h1,
                "rmse_p": rmse_p,
                "rmse_u": rmse_u,
                "rmse_v": rmse_v,
                "rmse_U": rmse_U,
                "kappa_names": kappa_names,
                "mom_mse": mom_mse,
                "cont_mse": cont_mse,
                "bc_mse": bc_mse,
                "meta": json.dumps(meta_clean),
            }
        )

    df = pd.DataFrame(rows)
    _validate_generated_source_indices(provenance, generated_source_indices)
    if len(df) != expected_case_count:
        msg = f"Artifact generation produced {len(df)} cases, expected {expected_case_count} from provenance."
        raise RuntimeError(msg)

    parquet_path = save_root / f"{dataset_name}.parquet"
    common.serialization.atomic_path_write(
        parquet_path,
        lambda temp_path: df.to_parquet(temp_path, index=False),
    )
    write_artifact_provenance(save_root, provenance)

    return df, parquet_path


def _generate_generic_artifacts(
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Store task-declared fields without assuming any concrete field names."""
    root = Path(save_root)
    expected_case_count = _artifact_effective_case_count(provenance)
    _ensure_artifact_targets_absent(root, dataset_name)
    model.eval()
    npz_dir = root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    generated_source_indices: list[int] = []

    for iteration, batch in enumerate(loader):
        if max_cases is not None and iteration >= max_cases:
            break
        split_local_index = _require_batch_scalar_int(batch, "split_local_index")
        source_index = _require_batch_scalar_int(batch, "source_index")
        if split_local_index != iteration:
            msg = f"Artifact loader order does not match saved split-local identity: iteration={iteration}, split_local_index={split_local_index}."
            raise RuntimeError(msg)
        if source_index < 0:
            msg = f"Artifact source_index must be non-negative, got {source_index}."
            raise ValueError(msg)
        case_index = source_index + 1
        generated_source_indices.append(source_index)

        source_meta = meta_to_jsonable(batch.get("meta", {}))
        metadata = dict(source_meta) if isinstance(source_meta, dict) else {"source_meta": source_meta}
        reserved = {"case_index", "source_index", "split_local_index"}.intersection(metadata)
        if reserved:
            msg = f"Source metadata contains reserved artifact identity keys: {sorted(reserved)}."
            raise KeyError(msg)

        inputs = batch["x"].to(device)
        targets = batch["y"].to(device)
        _require_finite_artifact_tensor(inputs, label=f"Artifact case {case_index} inputs")
        _require_finite_artifact_tensor(targets, label=f"Artifact case {case_index} targets")
        with torch.no_grad():
            normalized_inputs = processor.in_normalizer.transform(inputs)
            normalized_prediction = model(normalized_inputs)
            prediction = processor.out_normalizer.inverse_transform(normalized_prediction)
        _require_finite_artifact_tensor(
            normalized_inputs,
            label=f"Artifact case {case_index} normalized inputs",
        )
        _require_finite_artifact_tensor(
            normalized_prediction,
            label=f"Artifact case {case_index} normalized prediction",
        )
        _require_finite_artifact_tensor(
            prediction,
            label=f"Artifact case {case_index} physical prediction",
        )
        if prediction.shape != targets.shape:
            msg = f"Prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(targets.shape)}."
            raise RuntimeError(msg)
        if prediction.shape[1] != len(task.output_names):
            msg = f"Artifact output channel count {prediction.shape[1]} does not match task fields {list(task.output_names)}."
            raise RuntimeError(msg)

        prediction_cpu = prediction.squeeze(0).detach().cpu()
        target_cpu = targets.squeeze(0).detach().cpu()
        error_cpu = prediction_cpu - target_cpu
        npz_path = npz_dir / f"case_{case_index:04d}.npz"
        payload = {
            "case_index": np.int64(case_index),
            "source_index": np.int64(source_index),
            "split_local_index": np.int64(split_local_index),
            "pred": prediction_cpu.numpy(),
            "gt": target_cpu.numpy(),
            "err": error_cpu.numpy(),
            "artifact_fields": np.asarray(task.output_names),
            "artifact_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "x_raw": inputs.squeeze(0).detach().cpu().numpy(),
            "y_raw": target_cpu.numpy(),
            "input_fields": np.asarray(task.input_names),
            "output_fields": np.asarray(task.output_names),
            "output_units": np.asarray(tuple(field.unit for field in task.outputs)),
            "meta": json.dumps(metadata),
        }

        def write_npz(temp_path: Path, content: dict[str, Any] = payload) -> None:
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, **content)

        common.serialization.atomic_path_write(npz_path, write_npz)
        row: dict[str, Any] = {
            "inference_time_ms": None,
            "case_index": case_index,
            "source_index": source_index,
            "split_local_index": split_local_index,
            "npz_path": str(npz_path),
            "meta": json.dumps(metadata),
        }
        for channel, field in enumerate(task.outputs):
            row[f"rmse_{field.name}"] = float(torch.sqrt(torch.mean(error_cpu[channel].square())).item())
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.columns.is_unique:
        msg = "Generic artifact table contains duplicate columns."
        raise RuntimeError(msg)
    _validate_generated_source_indices(provenance, generated_source_indices)
    if len(frame) != expected_case_count:
        msg = f"Artifact generation produced {len(frame)} cases, expected {expected_case_count}."
        raise RuntimeError(msg)
    parquet_path = root / f"{dataset_name}.parquet"
    common.serialization.atomic_path_write(
        parquet_path,
        lambda temp_path: frame.to_parquet(temp_path, index=False),
    )
    write_artifact_provenance(root, provenance)
    return frame, parquet_path


def generate_artifacts(
    *,
    task: domain.tasks.spec.TaskSpec,
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    provenance: Mapping[str, Any],
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Generate artifacts through a task-extensible storage contract.

    Parameters
    ----------
    task : domain.tasks.spec.TaskSpec
        Validated task owning ordered input/output fields and units.
    model : Any
        Reconstructed best-checkpoint model.
    loader : Iterable[Mapping[str, Any]]
        Deterministic batch-size-one saved-split batches.
    processor : Any
        Restored training normalizer processor.
    device : torch.device
        Inference device.
    save_root : str | Path
        Exact artifact target directory.
    dataset_name : str
        Logical dataset name used for the Parquet filename.
    provenance : Mapping[str, Any]
        Exact cache identity published only after payload completion.
    max_cases : int | None, optional
        Positive effective saved-split case limit.

    Returns
    -------
    tuple[pandas.DataFrame, Path]
        Generated table and atomically published Parquet path.

    Notes
    -----
    The maintained steady-flow task retains its task-specific diagnostic
    adapter. Every other valid TaskSpec uses generic named field/unit storage,
    so adding a future task does not require lifecycle changes.

    """
    dataset_name = common.paths.validate_logical_name(dataset_name, label="dataset_name")
    if task.id == domain.tasks.steady_flow.STEADY_FLOW.id:
        return _generate_steady_flow_artifacts(
            task=task,
            model=model,
            loader=loader,
            processor=processor,
            device=device,
            save_root=save_root,
            dataset_name=dataset_name,
            provenance=provenance,
            max_cases=max_cases,
        )
    return _generate_generic_artifacts(
        task=task,
        model=model,
        loader=loader,
        processor=processor,
        device=device,
        save_root=save_root,
        dataset_name=dataset_name,
        provenance=provenance,
        max_cases=max_cases,
    )
