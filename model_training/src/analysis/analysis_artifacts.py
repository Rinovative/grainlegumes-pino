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
  - Physical units and domain field order are explicit
  - Heavy inference work stays out of plotting modules

Boundaries:
  - Model reconstruction belongs to learning.inference.context
  - Interactive visualization belongs to analysis.evaluation and analysis.ui

Notes:
  Artifact contents:
    - Parquet stores case-level scalar metrics, artifact paths and JSON-safe metadata
    - NPZ stores predictions, targets, errors, kappa fields, raw tensors and residual fields
    - residual metrics use the same canonical evaluation crop for model comparability
  Schema:
    Artifacts
    ---------
    Parquet (global, per case):
        - case_index        : integer case id
        - npz_path          : path to the corresponding NPZ artifact

        # Absolute field errors (physical units, full domain unless noted)
        - l2                : absolute RMSE on [p, u, v] over the full domain
        - h1                : absolute H1-like error on [p, u, v] over the interior (cropped by EVAL_PAD)

        # Relative field errors (dimensionless, channel-balanced)
        - rel_l2            : channel-balanced mean relative L2 on [p, u, v] over the full domain
        - rel_h1            : channel-balanced mean relative H1 on [p, u, v] over the interior (cropped by EVAL_PAD)

        # Book-friendly metrics (physical units, intuitive)
        - rmse_p            : RMSE of pressure p over the full domain
        - rmse_U            : RMSE of speed magnitude U=sqrt(u^2+v^2) over the full domain

        # Physics residual metrics (interior-cropped by EVAL_PAD)
        - mom_mse           : MSE of Brinkman momentum residual
        - cont_mse          : MSE of continuity residual div(u) computed with SP (reflect-FFT)
        - phys_mse          : mom_mse + cont_mse (convenience scalar)

        # Boundary condition metric (no crop, evaluated on inlet/outlet masks)
        - bc_mse            : pressure BC mismatch on inlet/outlet boundaries

        # Diagnostics
        - cont_mse_divepsu  : MSE of div(eps u) on interior (cropped by EVAL_PAD), diagnostic even if cont_mse is div(u)
        - kappa_names       : list of available permeability tensor components
        - meta              : JSON-safe metadata dictionary (stored as JSON string)

    NPZ (local, full fields per case):
        - pred         : (4, H, W) prediction [p, u, v, U]
        - gt           : (4, H, W) ground truth [p, u, v, U]
        - err          : (4, H, W) prediction error (pred - gt) for [p, u, v, U]

        - kappa_log    : (C_kappa, H, W) log10-permeability components
        - kappa        : (C_kappa, H, W) physical permeability components
        - kappa_names  : list[str], same order as kappa channels
        - p_bc         : (1, H, W) pressure boundary condition

        # Raw inputs/targets for compatibility with downstream tools
        - x_raw        : (C_in, H, W) raw input tensor (physical units)
        - y_raw        : (C_out, H, W) raw target tensor (physical units)
        - input_fields : list[str] canonical input channel names

        # Physics diagnostic fields (full fields, not cropped)
        - Rx           : (H, W) x-momentum residual field
        - Ry           : (H, W) y-momentum residual field
        - Rc           : (H, W) continuity residual field (here: div(u))
        - div_u        : (H, W) divergence field div(u)
        - div_eps_u    : (H, W) divergence field div(eps u)

        - meta         : JSON string with full metadata
===============================================================================

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src import domain, learning

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


# ============================================================================
# Global constants
# ============================================================================

INTERNAL_KAPPA_NAMES = set(domain.permeability.INTERNAL_KAPPA_2D_ORDER) | set(domain.permeability.INTERNAL_KAPPA_3D_ORDER)
MU_AIR = 1.8139e-5  # must be consistent with training

# ----------------------------------------------------------------------------
# Canonical evaluation settings (for model-to-model comparability)
# ----------------------------------------------------------------------------
EVAL_PAD = 2  # crop for ALL gradient-based metrics (H1 + physics)

# =============================================================================
# JSON / type normalisation utilities
# =============================================================================

NUMPY_INT_TYPES = (
    np.int_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)

NUMPY_FLOAT_TYPES = (
    np.float16,
    np.float32,
    np.float64,
)


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

    if isinstance(obj, NUMPY_INT_TYPES):  # pyright: ignore[reportArgumentType]
        return int(obj)

    if isinstance(obj, NUMPY_FLOAT_TYPES):  # pyright: ignore[reportArgumentType]
        return float(obj)

    if isinstance(obj, dict):
        return {k: meta_to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [meta_to_jsonable(v) for v in obj]

    return obj


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
    Extract log-kappa and physical kappa fields from the input tensor.

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
            - "kappa_log"
            - "kappa"

    """
    if not kappa_names:
        return {
            "kappa_log": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
            "kappa": x_tensor.new_empty((x_tensor.shape[0], 0, *x_tensor.shape[2:])),
        }

    index_map = {name: i for i, name in enumerate(input_fields)}
    kappa_indices = [index_map[name] for name in kappa_names]

    # log-kappa (as stored in dataset)
    kappa_log = x_tensor[:, kappa_indices, :, :]

    # physical kappa reconstruction
    kappa_phys = torch.zeros_like(kappa_log)
    name_to_pos = {name: i for i, name in enumerate(kappa_names)}

    # kxx, kyy (always log10-physical)
    kxx = torch.pow(10.0, kappa_log[:, name_to_pos["kxx"]])
    kyy = torch.pow(10.0, kappa_log[:, name_to_pos["kyy"]])
    kappa_phys[:, name_to_pos["kxx"]] = kxx
    kappa_phys[:, name_to_pos["kyy"]] = kyy

    # kxy is stored as kxy_hat (dimensionless). Keep it as is.
    if "kxy" in name_to_pos:
        kxy_hat = kappa_log[:, name_to_pos["kxy"]]
        kappa_phys[:, name_to_pos["kxy"]] = kxy_hat

    return {
        "kappa_log": kappa_log,
        "kappa": kappa_phys,
    }


# =============================================================================
# Main artifact generator
# =============================================================================


def generate_artifacts(  # noqa: PLR0915
    *,
    model: Any,
    loader: DataLoader,
    processor: Any,
    device: torch.device,
    save_root: str | Path,
    dataset_name: str,
    max_cases: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Run inference on all cases and generate persistent evaluation artifacts.

    For each case:
        - perform a forward pass with the trained model
        - compute global error metrics (l2/rel_l2 full-domain; h1 + physics terms on interior cropped by EVAL_PAD)
        - store full spatial fields in an NPZ file
        - store scalar metrics and metadata in a Parquet table

    Parameters
    ----------
    model : Any
        Trained neural operator model (FNO, PINO, etc.).
    loader : DataLoader
        Deterministic evaluation DataLoader.
    processor : Any
        Normalisation processor used during training.
    device : torch.device
        Device used for inference.
    save_root : str or Path
        Root directory for all generated artifacts.
    dataset_name : str
        Base name for the Parquet summary file.
    max_cases : int or None, optional
        Maximum number of cases to process. If None, process all cases.

    Returns
    -------
    df : pandas.DataFrame
        Per-case summary table.
    parquet_path : pathlib.Path
        Path to the written Parquet file.

    """
    model.eval()

    save_root = Path(save_root)

    # Infer run_dir from save_root (.../processed/<run>/analysis/...)
    run_dir = save_root
    while run_dir.name not in {"processed", ""} and not (run_dir / "best_model_state_dict.pt").exists():
        if run_dir.parent == run_dir:
            break
        run_dir = run_dir.parent

    run_name = run_dir.name
    physics_variant = "divu-SP"

    print(
        "[ARTIFACTS]",
        f"save_root={save_root}",
        f"run_dir={run_dir}",
        f"run_name={run_name}",
        f"variant={physics_variant}",
        sep="\n  - ",
    )

    # --------------------------------------------------
    # Build exact residual calculator (same as training loss)
    # --------------------------------------------------
    def _zero_data_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        return pred.new_zeros(())

    # Canonical eval physics: spectral derivatives (reflect-FFT) + div(u), fixed interior crop
    loss_obj = learning.losses.pino.PINOSpectralLossDiv(
        data_loss=_zero_data_loss,
        lambda_phys=1.0,
        lambda_p=0.0,
        in_normalizer=processor.in_normalizer,
        out_normalizer=processor.out_normalizer,
        grad_mode="fft_reflect",
        interior_pad=EVAL_PAD,
        log_every=10_000_000,
    ).to(device)
    loss_obj.eval()
    print(f"[ARTIFACTS] Using EVAL loss_cls={loss_obj.__class__.__name__} (pad={EVAL_PAD})")

    npz_dir = save_root / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    # Normalisation tensors
    in_mean = processor.in_normalizer.mean.to(device)
    in_std = processor.in_normalizer.std.to(device)
    out_mean = processor.out_normalizer.mean.to(device)
    out_std = processor.out_normalizer.std.to(device)

    rows: list[dict[str, Any]] = []

    # Detect available kappa channels from field contracts
    kappa_names = detect_kappa_channels_from_inputs(domain.field_sets.DEFAULT_INPUTS_2D)

    for idx, batch in enumerate(loader):
        if max_cases is not None and idx >= max_cases:
            break
        case_id = idx + 1

        x = batch["x"].to(device)
        y = batch["y"].to(device)

        # Metadata (generator-side)
        meta_clean = meta_to_jsonable(batch.get("meta", {}))

        # Pressure boundary condition (stored for diagnostics)
        p_bc_idx = domain.field_sets.DEFAULT_INPUTS_2D.index("p_bc")
        p_bc = x[:, p_bc_idx : p_bc_idx + 1].detach().cpu()

        # Permeability fields (no scalar stats here)
        kappa_info = extract_kappa(
            x,
            input_fields=domain.field_sets.DEFAULT_INPUTS_2D,
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
            x_norm = (x - in_mean) / (in_std + 1e-12)
            y_hat_norm = model(x_norm)
            y_hat = y_hat_norm * (out_std + 1e-12) + out_mean

        # --------------------------------------------------
        # Physics diagnostics (exact training-consistent implementation)
        # --------------------------------------------------
        with torch.no_grad():
            diag = loss_obj.compute_diagnostics(y_hat_norm, x=x_norm)

        mom_mse = float(diag["mom_mse"].detach().cpu().item())
        bc_mse = float(diag["bc_mse"].detach().cpu().item())

        # Rc from this eval loss is div(u)
        cont_mse_divu = float(diag["cont_mse"].detach().cpu().item())
        phys_mse = float(mom_mse + cont_mse_divu)

        Rx_np = diag["Rx"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Ry_np = diag["Ry"].detach().cpu().squeeze(0).squeeze(0).numpy()
        Rc_np = diag["Rc"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divu_np = diag["div_u"].detach().cpu().squeeze(0).squeeze(0).numpy()
        divepsu_np = diag["div_eps_u"].detach().cpu().squeeze(0).squeeze(0).numpy()

        # diagnostic: div(eps u) MSE with same crop
        div_eps_u_i = divepsu_np[EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD] if EVAL_PAD > 0 else divepsu_np
        cont_mse_divepsu = float(np.mean(div_eps_u_i**2))

        if device.type == "cuda" and end_time is not None:
            end_time.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            inference_time_ms = start_time.elapsed_time(end_time) if start_time is not None else None
        else:
            inference_time_ms = None

        # --------------------------------------------------
        # Outputs (de-normalised, physical units)
        # --------------------------------------------------
        p, u, v = y_hat[:, 0:1], y_hat[:, 1:2], y_hat[:, 2:3]
        p_gt, u_gt, v_gt = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        U = torch.sqrt(u**2 + v**2)
        U_gt = torch.sqrt(u_gt**2 + v_gt**2)

        # Book-friendly metrics (physical units)
        rmse_p = torch.sqrt(torch.mean((p - p_gt) ** 2)).item()
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
        idx_x = domain.field_sets.DEFAULT_INPUTS_2D.index("x")
        idx_y = domain.field_sets.DEFAULT_INPUTS_2D.index("y")
        dx = float((x[0, idx_x, 0, 1] - x[0, idx_x, 0, 0]).abs().detach().cpu().item())
        dy = float((x[0, idx_y, 1, 0] - x[0, idx_y, 0, 0]).abs().detach().cpu().item())

        metric_eps = 1e-12

        # ------------------------------------------------------------------
        # Absolute + relative L2/H1 on [p, u, v]
        # ------------------------------------------------------------------
        rel_l2_per_channel: list[float] = []
        rel_h1_per_channel: list[float] = []

        # Absolute "L2" here is RMSE over all entries of [p,u,v]
        l2 = torch.sqrt(torch.mean(err_main.pow(2))).item()

        # Absolute H1-like: RMSE of (field error + gradient error) on interior
        derr_dy_all, derr_dx_all = torch.gradient(err_main, spacing=(dy, dx), dim=(2, 3))
        if EVAL_PAD > 0:
            err_i = err_main[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
            derr_dx_i = derr_dx_all[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
            derr_dy_i = derr_dy_all[..., EVAL_PAD:-EVAL_PAD, EVAL_PAD:-EVAL_PAD]
        else:
            err_i, derr_dx_i, derr_dy_i = err_main, derr_dx_all, derr_dy_all

        h1 = torch.sqrt(torch.mean(err_i.pow(2) + derr_dx_i.pow(2) + derr_dy_i.pow(2))).item()

        for c in range(y_main.shape[1]):  # c in {p,u,v}
            e_c = err_main[:, c : c + 1]
            r_c = y_main[:, c : c + 1]

            # Relative L2 per channel (global norm ratio)
            l2_e_c = torch.linalg.norm(e_c)
            l2_r_c = torch.linalg.norm(r_c)
            rel_l2_c = (l2_e_c / (l2_r_c + metric_eps)).item()
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

            rel_h1_c = (h1_e_c / (h1_r_c + metric_eps)).item()
            rel_h1_per_channel.append(float(rel_h1_c))

        rel_l2 = float(np.mean(rel_l2_per_channel))
        rel_h1 = float(np.mean(rel_h1_per_channel))

        # --------------------------------------------------
        # Write NPZ artifact
        # --------------------------------------------------
        npz_path = npz_dir / f"case_{case_id:04d}.npz"
        x_raw = x.squeeze(0).detach().cpu().numpy()  # (C_in,H,W)
        y_raw = y.squeeze(0).detach().cpu().numpy()  # (C_out,H,W)

        np.savez_compressed(
            npz_path,
            pred=y_hat_ext.squeeze(0).cpu().numpy(),
            gt=y_ext.squeeze(0).cpu().numpy(),
            err=err_ext.squeeze(0).cpu().numpy(),
            kappa_log=kappa_info["kappa_log"].squeeze(0).cpu().numpy(),
            kappa=kappa_info["kappa"].squeeze(0).cpu().numpy(),
            kappa_names=np.array(kappa_names, dtype=object),
            p_bc=p_bc.squeeze(0).numpy(),
            meta=json.dumps(meta_clean),
            x_raw=x_raw,
            y_raw=y_raw,
            input_fields=np.array(domain.field_sets.DEFAULT_INPUTS_2D, dtype=object),
            Rx=Rx_np,
            Ry=Ry_np,
            Rc=Rc_np,
            div_u=divu_np,
            div_eps_u=divepsu_np,
        )

        # --------------------------------------------------
        # Parquet row (scalar metrics + metadata only)
        # --------------------------------------------------
        rows.append(
            {
                "inference_time_ms": inference_time_ms,
                "case_index": case_id,
                "npz_path": str(npz_path),
                "l2": l2,
                "h1": h1,
                "rel_l2": rel_l2,
                "rel_h1": rel_h1,
                "rmse_p": rmse_p,
                "rmse_U": rmse_U,
                "kappa_names": kappa_names,
                "mom_mse": mom_mse,
                "cont_mse": cont_mse_divu,
                "phys_mse": phys_mse,
                "bc_mse": bc_mse,
                "cont_mse_divepsu": cont_mse_divepsu,
                "meta": json.dumps(meta_clean),
            }
        )

    df = pd.DataFrame(rows)
    parquet_path = save_root / f"{dataset_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    return df, parquet_path
