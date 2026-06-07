"""
===============================================================================
 learning_losses_spectral.py
===============================================================================
Spectral-space derivative operators for PINO residual computation via FFT.

Responsibilities:
  - Compute gradients using Fast Fourier Transform (FFT-based differentiation)
  - Support Reflect-Extension boundary handling to reduce edge artifacts
  - Support plain FFT with periodic boundary assumption
  - Compute divergences using spectral derivatives
  - Support Brinkman and continuity residual evaluation

Design principles:
  - FFT-based differentiation for spectral accuracy
  - Reflect-Extension padding reduces edge artifacts (default mode)
  - Support 2D spatial domains with optional interior cropping
  - Numerically stable with half-precision (float16, bfloat16) support

Supported modes:
  - "fft": Plain FFT with periodic boundary assumption
  - "fft_reflect": Reflect-Extension before FFT (reduces boundary artifacts)

This module does NOT:
  - Define loss functions or training objectives
  - Manage model architecture or checkpointing
  - Handle physical-space derivatives (see learning_losses_physical.py)
  - Perform gradient scaling or normalization beyond physical units

Physics context:
  - Used for evaluating Brinkman momentum: -∇p + ∇·τ - μ K^{-1} u = 0
  - Used for evaluating continuity: ∇·(ε u) = 0 or ∇·u = 0
  - Consistent with COMSOL's deviatoric stress formulation

Tensor conventions:
  - inputs  x : (B, C_in, H, W)
  - outputs y : (B, C_out, H, W)
  - predictions pred : (B, C_out, H, W)
  - Spatial derivatives along last two dimensions (H, W)
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import wandb
from torch import nn

from src import domain

if TYPE_CHECKING:
    from collections.abc import Callable

MU_AIR = 1.8139e-5  # Pa*s


# ================================================================
# Spectral utilities
# ================================================================
def _fftfreq_2d(H: int, W: int, dx: float, dy: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D FFT frequencies.

    Parameters
    ----------
    H: int
        Height of the grid
    W: int
        Width of the grid
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in y-direction
    device: torch.device
        Device for the output tensors
    dtype: torch.dtype
        Data type for the output tensors

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ky, kx frequency tensors

    """
    kx = 2.0 * torch.pi * torch.fft.rfftfreq(W, d=dx, device=device, dtype=dtype)  # (W//2+1,)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(H, d=dy, device=device, dtype=dtype)  # (H,)
    return ky, kx


def _spectral_grad_fft(f: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spectral gradient via FFT.

    Parameters
    ----------
    f: torch.Tensor
        Input tensor of shape (B, H, W)
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in y-direction

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gradients (dfdx, dfdy) each of shape (B, H, W)

    """
    _B, H, W = f.shape
    dtype = torch.float32 if f.dtype in (torch.float16, torch.bfloat16) else f.dtype
    ky, kx = _fftfreq_2d(H, W, dx, dy, f.device, dtype)

    Ff = torch.fft.rfft2(f.to(dtype), dim=(-2, -1))
    dfdx = torch.fft.irfft2(1j * kx.view(1, 1, -1) * Ff, s=(H, W))
    dfdy = torch.fft.irfft2(1j * ky.view(1, -1, 1) * Ff, s=(H, W))
    return dfdx.to(f.dtype), dfdy.to(f.dtype)


def _spectral_grad_fft_reflect(f: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spectral gradient via FFT with Reflect-Extension (reduces edge artefacts).

    Parameters
    ----------
    f: torch.Tensor
        Input tensor of shape (B, H, W)
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in y-direction

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gradients (dfdx, dfdy) each of shape (B, H, W

    """
    _B, H, W = f.shape
    dtype = torch.float32 if f.dtype in (torch.float16, torch.bfloat16) else f.dtype

    # (left, right, top, bottom) for last two dims (W, H)
    f_pad = torch.nn.functional.pad(f, (0, W - 1, 0, H - 1), mode="reflect").to(dtype)
    Hp, Wp = f_pad.shape[-2], f_pad.shape[-1]

    ky, kx = _fftfreq_2d(Hp, Wp, dx, dy, f.device, dtype)

    Ff = torch.fft.rfft2(f_pad, dim=(-2, -1))
    dfdx_pad = torch.fft.irfft2(1j * kx.view(1, 1, -1) * Ff, s=(Hp, Wp))
    dfdy_pad = torch.fft.irfft2(1j * ky.view(1, -1, 1) * Ff, s=(Hp, Wp))

    dfdx = dfdx_pad[..., :H, :W].to(f.dtype)
    dfdy = dfdy_pad[..., :H, :W].to(f.dtype)
    return dfdx, dfdy


def spectral_grad(
    f: torch.Tensor,
    dx: float,
    dy: float,
    *,
    mode: Literal["fft", "fft_reflect"] = "fft_reflect",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spectral gradient via selected method.

    Parameters
    ----------
    f: torch.Tensor
        Input tensor of shape (B, H, W)
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in y-direction
    mode: Literal["fft", "fft_reflect"], optional
        Spectral gradient mode (default: "fft_reflect")

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gradients (dfdx, dfdy) each of shape (B, H, W)

    """
    if mode == "fft":
        return _spectral_grad_fft(f, dx, dy)
    if mode == "fft_reflect":
        return _spectral_grad_fft_reflect(f, dx, dy)
    msg = f"Unknown spectral grad mode: {mode}"
    raise ValueError(msg)


def spectral_div(
    fx: torch.Tensor,
    fy: torch.Tensor,
    dx: float,
    dy: float,
    *,
    mode: Literal["fft", "fft_reflect"] = "fft_reflect",
) -> torch.Tensor:
    """
    Spectral divergence via selected method.

    Parameters
    ----------
    fx: torch.Tensor
        x-component tensor of shape (B, H, W)
    fy: torch.Tensor
        y-component tensor of shape (B, H, W)
    dx: float
        Grid spacing in x-direction
    dy: float
        Grid spacing in y-direction
    mode: Literal["fft", "fft_reflect"], optional
        Spectral gradient mode (default: "fft_reflect")

    Returns
    -------
    torch.Tensor
        Divergence tensor of shape (B, H, W)

    """
    dfxdx, _ = spectral_grad(fx, dx, dy, mode=mode)
    _, dfydy = spectral_grad(fy, dx, dy, mode=mode)
    return dfxdx + dfydy


# ================================================================
# Small helpers
# ================================================================
def _interior(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Extract interior region by cropping 'pad' pixels from each boundary.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor of shape (..., H, W)
    pad: int
        Number of pixels to crop from each boundary

    Returns
    -------
    torch.Tensor
        Cropped tensor of shape (..., H-2*pad, W-2*pad)

    """
    if pad <= 0:
        return x
    return x[..., pad:-pad, pad:-pad]


# ================================================================
# PINO Brinkman Loss
# ================================================================
class PINOSpectralLoss(nn.Module):
    """
    PINO loss for 2D stationary incompressible Brinkman flow.

    Combines data loss, physics residual loss, and pressure-BC loss.

    Dataschema:
    - Inputs:
        - kxx, kyy: log10(Kxx), log10(Kyy)  (K in m^2)
        - kxy:      kxy_hat = Kxy / sqrt(Kxx*Kyy)  (dimensionless)
        - phi:      Porosity (0..1)
        - p_bc:     Pressure BC as volume field (0 except at inlet)
        - x, y:     Coordinate fields
    - Outputs:
        - p, u, v:  Physical fields (Pa, m/s, m/s) after inverse_transform
    Brinkman (strong, stationary, incompressible, conservative form):
    -∇p + ∇·tau - mu * K^{-1} u = 0
    ∇·(eps u) = 0
      tau = (mu/eps) * ( ∇u + ∇u^T - (2/3)(∇·u) I )  (deviatoric, COMSOL-like)

    Parameters
    ----------
    data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Data loss function in normalized space (e.g., L2 or H1)
    lambda_phys: float
        Weight for the physics residual loss
    lambda_p: float
        Weight for the pressure BC loss
    in_normalizer: Any
        Normalizer for input features (must have inverse_transform method)
    out_normalizer: Any
        Normalizer for output features (must have inverse_transform method)
    grad_mode: Literal["fft", "fft_reflect"], optional
        Spectral gradient mode (default: "fft_reflect")
    interior_pad: int, optional
        Number of grid points (discrete cells) cropped from each boundary
        when evaluating the physics loss (suppresses FFT edge artefacts;
        default: 2)
    log_every: int, optional
        Logging interval (in steps) for wandb (default: 10)

    Returns
    -------
    torch.Tensor
        Total loss (scalar)

    """

    def __init__(
        self,
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_phys: float,
        lambda_p: float,
        in_normalizer: Any | None = None,
        out_normalizer: Any | None = None,
        *,
        grad_mode: Literal["fft", "fft_reflect"] = "fft_reflect",
        interior_pad: int = 2,
        log_every: int = 10,
    ) -> None:
        """
        Initialize the PINOLoss.

        Parameters
        ----------
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Data loss function in normalized space (e.g., L2 or H1)
        lambda_phys: float
            Weight for the physics residual loss
        lambda_p: float
            Weight for the pressure BC loss
        in_normalizer: Any
            Normalizer for input features (must have inverse_transform method)
        out_normalizer: Any
            Normalizer for output features (must have inverse_transform method)
        grad_mode: Literal["fft", "fft_reflect"], optional
            Spectral gradient mode (default: "fft_reflect")
        interior_pad: int, optional
            Number of pixels to crop from each boundary for physics loss (default: 2)
        log_every: int, optional
            Logging interval (in steps) for wandb (default: 10)

        """
        super().__init__()

        self.data_loss = data_loss
        self.lambda_phys = float(lambda_phys)
        self.lambda_p = float(lambda_p)

        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer

        self.grad_mode: Literal["fft", "fft_reflect"] = grad_mode
        self.interior_pad = int(interior_pad)

        self.log_every = int(log_every)
        self.step = 0

        # numerical safeties
        self._eps_eps = 1e-6
        self._eps_det = 1e-30
        self._kxy_hat_clip = 0.999

        # field order from schema
        self.input_fields = domain.field_sets.DEFAULT_INPUTS_2D
        self.output_fields = domain.field_sets.DEFAULT_OUTPUTS_2D
        self.iidx = {n: i for i, n in enumerate(self.input_fields)}
        self.oidx = {n: i for i, n in enumerate(self.output_fields)}

    def forward(
        self,
        pred: torch.Tensor,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute the PINO loss.

        Parameters
        ----------
        pred: torch.Tensor
            Predicted outputs in normalized space (B, C_out, H, W)
        x: torch.Tensor
            Input features in normalized space (B, C_in, H, W)
        y: torch.Tensor
            Ground-truth outputs in normalized space (B, C_out, H, W)
        **_kwargs: Any
            Additional keyword arguments (not used)

        Returns
        -------
        torch.Tensor
            Total loss (scalar)

        """
        # ------------------------------------------------------------
        # 1) Data loss (normalized space)
        # ------------------------------------------------------------
        data = self.data_loss(pred, y)

        # ------------------------------------------------------------
        # 2) Denormalize to physical space
        # ------------------------------------------------------------
        if self.in_normalizer is None or self.out_normalizer is None:
            msg = "PINOSpectralLoss: normalizers not set. Call set_normalizers(in_normalizer=..., out_normalizer=...)."
            raise RuntimeError(msg)
        in_norm = self.in_normalizer
        out_norm = self.out_normalizer

        x_phys = in_norm.inverse_transform(x)
        pred_phys = out_norm.inverse_transform(pred)

        # ------------------------------------------------------------
        # 3) Outputs
        # ------------------------------------------------------------
        p = pred_phys[:, self.oidx["p"]]  # (B,H,W)
        u = pred_phys[:, self.oidx["u"]]
        v = pred_phys[:, self.oidx["v"]]

        # ------------------------------------------------------------
        # 4) Inputs
        # ------------------------------------------------------------
        eps = x_phys[:, self.iidx["phi"]].clamp_min(self._eps_eps)
        p_bc = x_phys[:, self.iidx["p_bc"]]

        Kxx = 10.0 ** x_phys[:, self.iidx["kxx"]]
        Kyy = 10.0 ** x_phys[:, self.iidx["kyy"]]

        kxy_hat = x_phys[:, self.iidx["kxy"]].clamp(-self._kxy_hat_clip, self._kxy_hat_clip)
        K0 = torch.sqrt((Kxx * Kyy).clamp_min(self._eps_det))
        _Kxy = kxy_hat * K0

        x_coord = x_phys[:, self.iidx["x"]]
        y_coord = x_phys[:, self.iidx["y"]]
        dx_t = (x_coord[:, 0, 1] - x_coord[:, 0, 0]).abs().mean().clamp_min(1e-12)
        dy_t = (y_coord[:, 1, 0] - y_coord[:, 0, 0]).abs().mean().clamp_min(1e-12)
        dx = float(dx_t.detach().cpu().item())
        dy = float(dy_t.detach().cpu().item())

        # ------------------------------------------------------------
        # 5) Spectral derivatives
        # ------------------------------------------------------------
        dpdx, dpdy = spectral_grad(p, dx, dy, mode=self.grad_mode)  # dpdx = ∂p/∂x , dpdy = ∂p/∂y
        dudx, dudy = spectral_grad(u, dx, dy, mode=self.grad_mode)  # dudx = ∂u/∂x , dudy = ∂u/∂y
        dvdx, dvdy = spectral_grad(v, dx, dy, mode=self.grad_mode)  # dvdx = ∂v/∂x , dvdy = ∂v/∂y

        div_u = dudx + dvdy  # ∇·u (used only inside viscous stress term)

        # ------------------------------------------------------------
        # 6) Brinkman viscous stress (COMSOL-like)
        # ------------------------------------------------------------
        coef = MU_AIR / eps

        tau_xx = coef * (2.0 * dudx - (2.0 / 3.0) * div_u)  # τ_xx = (μ/ε) [ 2 ∂u/∂x - (2/3)(∇·u) ]
        tau_yy = coef * (2.0 * dvdy - (2.0 / 3.0) * div_u)  # τ_yy = (μ/ε) [ 2 ∂v/∂y - (2/3)(∇·u) ]
        tau_xy = coef * (dudy + dvdx)  # τ_xy = (μ/ε) [ ∂u/∂y + ∂v/∂x ]

        div_tau_x = spectral_div(tau_xx, tau_xy, dx, dy, mode=self.grad_mode)  # ∇·τ_x = ∂τ_xx/∂x + ∂τ_xy/∂y
        div_tau_y = spectral_div(tau_xy, tau_yy, dx, dy, mode=self.grad_mode)  # ∇·τ_y = ∂τ_xy/∂x + ∂τ_yy/∂y

        # ------------------------------------------------------------
        # 7) Darcy drag
        # ------------------------------------------------------------
        det_hat = 1.0 - kxy_hat * kxy_hat
        det_hat_safe = det_hat.clamp_min(1e-4)

        invhat_xx = Kyy / K0 / det_hat_safe  # (K^{-1})_xx
        invhat_xy = -kxy_hat / det_hat_safe / K0  # (K^{-1})_xy
        invhat_yy = Kxx / K0 / det_hat_safe  # (K^{-1})_yy

        drag_x = MU_AIR * (invhat_xx * u + invhat_xy * v)  # μ (K^{-1} u)_x
        drag_y = MU_AIR * (invhat_xy * u + invhat_yy * v)  # μ (K^{-1} u)_y

        # ------------------------------------------------------------
        # 8) Residuals
        # ------------------------------------------------------------
        Rx = -dpdx + div_tau_x - drag_x
        Ry = -dpdy + div_tau_y - drag_y

        eps_u = eps * u
        eps_v = eps * v
        div_eps_u = spectral_div(eps_u, eps_v, dx, dy, mode=self.grad_mode)

        Rc = div_eps_u  # R_c = ∇·(eps u)

        # ------------------------------------------------------------
        # 9) Interior-only physics
        # ------------------------------------------------------------
        pad = self.interior_pad
        Rx_i = _interior(Rx, pad)
        Ry_i = _interior(Ry, pad)
        Rc_i = _interior(Rc, pad)

        phys = Rx_i.pow(2).mean() + Ry_i.pow(2).mean() + Rc_i.pow(2).mean()  # L_phys = ||R_x||^2 + ||R_y||^2 + ||R_c||^2

        # ------------------------------------------------------------
        # 10) Pressure BC loss (inlet + outlet, COMSOL-consistent)
        # ------------------------------------------------------------
        # Inlet: p = p_bc (pointwise, volume-field encoded)
        inlet = p_bc.abs() > 0.0
        p_inlet_loss = (
            (p[inlet] - p_bc[inlet]).pow(2).mean() if inlet.any() else torch.zeros((), device=p.device, dtype=p.dtype)
        )  # L_in = || p - p_bc ||^2_{Γ_in}

        # Outlet: mean pressure constraint (reference / gauge fix)
        y_max = y_coord.amax(dim=(-2, -1), keepdim=True)
        outlet = (y_coord - y_max).abs() <= 0.5 * dy_t

        p_outlet_loss = (
            p[outlet].mean().pow(2) if outlet.any() else torch.zeros((), device=p.device, dtype=p.dtype)
        )  # L_out = ( mean_{Γ_out}(p) )^2   (pressure gauge)

        p_bc_loss = p_inlet_loss + p_outlet_loss  # L_p = L_in + L_out

        # ------------------------------------------------------------
        # 11) Total loss
        # ------------------------------------------------------------
        total = data + self.lambda_phys * phys + self.lambda_p * p_bc_loss  # L = L_data + λ_phys L_phys + λ_p L_p

        # ------------------------------------------------------------
        # 12) Logging
        # ------------------------------------------------------------
        if wandb.run is not None and self.step % self.log_every == 0:
            total_val = total.item()

            wandb.log(
                {
                    # --- Total & loss components ---
                    "loss/total": total_val,
                    "loss/data": data.item(),
                    "loss/phys": phys.item(),
                    "loss/p_bc": p_bc_loss.item(),
                    # --- Relative contributions ---
                    "loss/frac_loss_data": data.item() / total_val,
                    "loss/frac_loss_phys": (self.lambda_phys * phys.item()) / total_val,
                    "loss/frac_loss_p_bc": (self.lambda_p * p_bc_loss.item()) / total_val,
                    # --- Physics diagnostics (L2 norms) ---
                    "physics/Rx_l2": Rx.pow(2).mean().sqrt().item(),
                    "physics/Ry_l2": Ry.pow(2).mean().sqrt().item(),
                    "physics/div_eps_u_l2": Rc.pow(2).mean().sqrt().item(),
                },
                commit=False,
            )

        self.step += 1
        return total

    def set_normalizers(
        self,
        *,
        in_normalizer: Any,
        out_normalizer: Any,
    ) -> None:
        """
        Attach input and output normalizers after loss construction.

        This method is called by the training pipeline once the dataset
        and its associated normalizers are available. It enables defining
        the PINO loss object independently of the data loading stage.

        Parameters
        ----------
        in_normalizer : Any
            Input normalizer with an `inverse_transform` method.
        out_normalizer : Any
            Output normalizer with an `inverse_transform` method.

        """
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
