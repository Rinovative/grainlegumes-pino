"""
===============================================================================
 learning_losses_pino.py
===============================================================================
Physics-informed neural operator (PINO) loss functions for Brinkman flow.

Responsibilities:
  - Compute supervised data loss (H1 + Lp norms on model predictions)
  - Compute Brinkman momentum and continuity residuals
  - Support both physical-space and spectral-space (FFT) derivative computation
  - Support both conservative div(ε u) and plain div(u) continuity formulations
  - Provide 4 public loss variants: Physical-Eps, Physical-Div, Spectral-Eps, Spectral-Div

Provided classes:
  - PINOPhysicalLossEps   : Physical derivatives + conservative continuity div(eps u)
  - PINOPhysicalLossDiv   : Physical derivatives + plain continuity div(u)
  - PINOSpectralLossEps   : Spectral derivatives + conservative continuity div(eps u)
  - PINOSpectralLossDiv   : Spectral derivatives + plain continuity div(u)

Design principles:
  - All variants wrap a consistent nn.Module interface
  - Derivative computation is transparent and reproducible
  - Air properties (μ) hardcoded to maintain consistency across runs
  - Compatible with mixed-precision training and W&B logging

This module does NOT:
  - Handle model architecture (see learning_models_*)
  - Orchestrate training (see learning_training_loop)
  - Manage gradient clipping or loss scaling (training loop responsibility)
  - Support arbitrary permeability schedules beyond training setup

Physics formulation (strong, stationary, heterogeneous porous media):
    -∇p + ∇·τ - μ K^{-1} u = 0        (Brinkman momentum)
    div(ε u) = 0   OR   div(u) = 0    (Conservative or plain continuity)
    τ = (μ/ε) * (∇u + ∇u^T - (2/3)∇·u I)  (Deviatoric stress, COMSOL convention)

Tensor conventions:
    inputs  x : (B, C_in, H, W)     [coordinates, permeability, BC]
    outputs y : (B, C_out, H, W)    [pressure, velocity u, velocity v]
    predictions pred : (B, C_out, H, W)
    Spatial derivatives along last two dimensions (H, W)

Field conventions:
    - kxx, kyy are stored as log10(Kxx), log10(Kyy) with K in m^2
    - kxy is stored as kxy_hat = Kxy / sqrt(Kxx*Kyy) (dimensionless)
    - K^{-1} constructed consistently as (1/K0) * Khat^{-1}, with K0 = sqrt(Kxx*Kyy)
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import wandb
from torch import nn

from src import domain

if TYPE_CHECKING:
    from collections.abc import Callable


MU_AIR = 1.8139e-5  # Pa*s


# =============================================================================
# Small helpers
# =============================================================================
def _interior(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Extract interior region by cropping 'pad' pixels from each boundary."""
    if pad <= 0:
        return x
    return x[..., pad:-pad, pad:-pad]


def _to_float(x: torch.Tensor) -> float:
    """Convert a scalar tensor to python float safely."""
    return float(x.detach().cpu().item())


def _infer_dx_dy(x_coord: torch.Tensor, y_coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Infer uniform grid spacing from coordinate fields.

    Parameters
    ----------
    x_coord, y_coord : torch.Tensor
        Coordinate fields of shape (B, H, W).

    Returns
    -------
    dx_t, dy_t : torch.Tensor
        Scalar tensors (on device) for dx and dy.

    """
    dx_t = (x_coord[:, 0, 1] - x_coord[:, 0, 0]).abs().mean().clamp_min(1e-12)
    dy_t = (y_coord[:, 1, 0] - y_coord[:, 0, 0]).abs().mean().clamp_min(1e-12)
    return dx_t, dy_t


# =============================================================================
# Derivative backends
# =============================================================================
class _DerivativeBackend:
    """Interface for spatial derivatives on uniform grids."""

    def grad(self, f: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def div(self, fx: torch.Tensor, fy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class _PhysicalDerivatives(_DerivativeBackend):
    """Physical-space derivatives via torch.gradient."""

    def grad(self, f: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # torch.gradient returns (df/dy, df/dx) if you request two dims
        dfdy = torch.gradient(f, dim=-2)[0] / dy
        dfdx = torch.gradient(f, dim=-1)[0] / dx
        return dfdx, dfdy

    def div(self, fx: torch.Tensor, fy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        dfxdx = torch.gradient(fx, dim=-1)[0] / dx
        dfydy = torch.gradient(fy, dim=-2)[0] / dy
        return dfxdx + dfydy


@dataclass(frozen=True)
class _SpectralDerivatives(_DerivativeBackend):
    """
    Spectral derivatives via FFT.

    grad_mode
        "fft"         : plain FFT (periodic assumption)
        "fft_reflect" : reflect-extension before FFT (reduces edge artefacts)
    """

    grad_mode: Literal["fft", "fft_reflect"] = "fft_reflect"

    @staticmethod
    def _fftfreq_2d(
        H: int,
        W: int,
        dx: float,
        dy: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ky = 2.0 * torch.pi * torch.fft.fftfreq(H, d=dy, device=device, dtype=dtype)  # (H,)
        kx = 2.0 * torch.pi * torch.fft.rfftfreq(W, d=dx, device=device, dtype=dtype)  # (W//2+1,)
        return ky, kx

    def _spectral_grad_fft(self, f: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
        _B, H, W = f.shape
        work_dtype = torch.float32 if f.dtype in (torch.float16, torch.bfloat16) else f.dtype
        ky, kx = self._fftfreq_2d(H, W, dx, dy, device=f.device, dtype=work_dtype)

        Ff = torch.fft.rfft2(f.to(work_dtype), dim=(-2, -1))
        dfdx = torch.fft.irfft2(1j * kx.view(1, 1, -1) * Ff, s=(H, W))
        dfdy = torch.fft.irfft2(1j * ky.view(1, -1, 1) * Ff, s=(H, W))
        return dfdx.to(f.dtype), dfdy.to(f.dtype)

    def _spectral_grad_fft_reflect(self, f: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
        _B, H, W = f.shape
        work_dtype = torch.float32 if f.dtype in (torch.float16, torch.bfloat16) else f.dtype

        # symmetrische Reflect-Extension:
        # pad order is (left, right, top, bottom) for last two dims (W, H)
        pad_w = W - 1
        pad_h = H - 1
        f_pad = torch.nn.functional.pad(f, (pad_w, pad_w, pad_h, pad_h), mode="reflect").to(work_dtype)
        Hp, Wp = f_pad.shape[-2], f_pad.shape[-1]

        ky, kx = self._fftfreq_2d(Hp, Wp, dx, dy, device=f.device, dtype=work_dtype)

        Ff = torch.fft.rfft2(f_pad, dim=(-2, -1))
        dfdx_pad = torch.fft.irfft2(1j * kx.view(1, 1, -1) * Ff, s=(Hp, Wp))
        dfdy_pad = torch.fft.irfft2(1j * ky.view(1, -1, 1) * Ff, s=(Hp, Wp))

        dfdx = dfdx_pad[..., pad_h : pad_h + H, pad_w : pad_w + W].to(f.dtype)
        dfdy = dfdy_pad[..., pad_h : pad_h + H, pad_w : pad_w + W].to(f.dtype)
        return dfdx, dfdy

    def grad(self, f: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx_f = _to_float(dx)
        dy_f = _to_float(dy)

        if self.grad_mode == "fft":
            return self._spectral_grad_fft(f, dx_f, dy_f)
        if self.grad_mode == "fft_reflect":
            return self._spectral_grad_fft_reflect(f, dx_f, dy_f)
        msg = f"Unknown grad_mode: {self.grad_mode}"
        raise ValueError(msg)

    def div(self, fx: torch.Tensor, fy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        dfxdx, _ = self.grad(fx, dx, dy)
        _, dfydy = self.grad(fy, dx, dy)
        return dfxdx + dfydy


# =============================================================================
# Shared internal implementation
# =============================================================================
_ContinuityMode = Literal["eps_u", "u"]


class _PINOBrinkmanLossBase(nn.Module):
    """
    Shared Brinkman PINO loss implementation.

    continuity_mode:
        "eps_u" : div(eps u) = 0 (conservative for heterogeneous porous media)
        "u"     : div(u) = 0     (plain incompressibility)
    """

    def __init__(
        self,
        *,
        backend: _DerivativeBackend,
        continuity_mode: _ContinuityMode,
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_phys: float,
        lambda_p: float,
        in_normalizer: Any | None = None,
        out_normalizer: Any | None = None,
        interior_pad: int = 0,
        log_every: int = 10,
        # numeric safeties
        eps_eps: float = 1e-6,
        eps_K0: float = 1e-30,
        kxy_hat_clip: float = 0.999,
        det_hat_min: float = 1e-4,
    ) -> None:
        super().__init__()

        self.backend = backend
        self.continuity_mode: _ContinuityMode = continuity_mode

        self.data_loss = data_loss
        self.lambda_phys = float(lambda_phys)
        self.lambda_p = float(lambda_p)

        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer

        self.interior_pad = int(interior_pad)
        self.log_every = int(log_every)
        self.step = 0

        self._eps_eps = float(eps_eps)
        self._eps_K0 = float(eps_K0)
        self._kxy_hat_clip = float(kxy_hat_clip)
        self._det_hat_min = float(det_hat_min)

        self.input_fields = domain.field_sets.DEFAULT_INPUTS_2D
        self.output_fields = domain.field_sets.DEFAULT_OUTPUTS_2D
        self.iidx = {n: i for i, n in enumerate(self.input_fields)}
        self.oidx = {n: i for i, n in enumerate(self.output_fields)}

    def set_normalizers(self, *, in_normalizer: Any, out_normalizer: Any) -> None:
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer

    @torch.no_grad()
    def compute_diagnostics(
        self,
        pred: torch.Tensor,
        *,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute physics diagnostics using the exact same implementation as the training loss.

        Important:
            pred and x must be in NORMALIZED space (same as forward()).

        Returns:
            Rx, Ry, Rc, div_u, div_eps_u as (B,1,H,W) and scalar MSE terms.

        """
        if self.in_normalizer is None or self.out_normalizer is None:
            msg = "PINO loss: normalizers not set. Call set_normalizers(...) or pass in/out normalizer to __init__."
            raise RuntimeError(msg)

        # Denormalize exactly as in forward()
        x_phys = self.in_normalizer.inverse_transform(x)
        pred_phys = self.out_normalizer.inverse_transform(pred)

        # Outputs
        p = pred_phys[:, self.oidx["p"]]  # (B,H,W)
        u = pred_phys[:, self.oidx["u"]]
        v = pred_phys[:, self.oidx["v"]]

        # Inputs
        eps = x_phys[:, self.iidx["phi"]].clamp_min(self._eps_eps)
        p_bc = x_phys[:, self.iidx["p_bc"]]

        x_coord = x_phys[:, self.iidx["x"]]
        y_coord = x_phys[:, self.iidx["y"]]
        dx_t, dy_t = _infer_dx_dy(x_coord, y_coord)

        Kxx = 10.0 ** x_phys[:, self.iidx["kxx"]]
        Kyy = 10.0 ** x_phys[:, self.iidx["kyy"]]
        kxy_hat = x_phys[:, self.iidx["kxy"]].clamp(-self._kxy_hat_clip, self._kxy_hat_clip)
        K0 = torch.sqrt((Kxx * Kyy).clamp_min(self._eps_K0))

        # Derivatives
        dpdx, dpdy = self.backend.grad(p, dx_t, dy_t)
        dudx, dudy = self.backend.grad(u, dx_t, dy_t)
        dvdx, dvdy = self.backend.grad(v, dx_t, dy_t)

        div_u = dudx + dvdy

        # Brinkman viscous stress (COMSOL-like deviatoric)
        coef = MU_AIR / eps
        tau_xx = coef * (2.0 * dudx - (2.0 / 3.0) * div_u)
        tau_yy = coef * (2.0 * dvdy - (2.0 / 3.0) * div_u)
        tau_xy = coef * (dudy + dvdx)

        div_tau_x = self.backend.div(tau_xx, tau_xy, dx_t, dy_t)
        div_tau_y = self.backend.div(tau_xy, tau_yy, dx_t, dy_t)

        # Darcy drag (consistent K^{-1} construction)
        kxx_hat = Kxx / K0
        kyy_hat = Kyy / K0

        det_hat = kxx_hat * kyy_hat - kxy_hat * kxy_hat
        det_hat_safe = det_hat.clamp_min(self._det_hat_min)

        invhat_xx = kyy_hat / det_hat_safe
        invhat_xy = -kxy_hat / det_hat_safe
        invhat_yy = kxx_hat / det_hat_safe

        inv_xx = invhat_xx / K0
        inv_xy = invhat_xy / K0
        inv_yy = invhat_yy / K0

        drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
        drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

        # Residuals
        Rx = -dpdx + div_tau_x - drag_x
        Ry = -dpdy + div_tau_y - drag_y

        div_eps_u = self.backend.div(eps * u, eps * v, dx_t, dy_t)
        Rc = div_eps_u if self.continuity_mode == "eps_u" else self.backend.div(u, v, dx_t, dy_t)

        # Interior-cropped (exactly like phys term in forward)
        pad = self.interior_pad
        Rx_i = _interior(Rx, pad)
        Ry_i = _interior(Ry, pad)
        Rc_i = _interior(Rc, pad)

        mom_mse_full = (Rx.pow(2) + Ry.pow(2)).mean()
        cont_mse_full = Rc.pow(2).mean()

        mom_mse = (Rx_i.pow(2) + Ry_i.pow(2)).mean()
        cont_mse = Rc_i.pow(2).mean()

        # Pressure BC loss (exactly like forward)
        dy = dy_t
        y_min = y_coord.amin(dim=(-2, -1), keepdim=True)
        y_max = y_coord.amax(dim=(-2, -1), keepdim=True)

        inlet = (y_coord - y_min).abs() <= 0.5 * dy
        outlet = (y_coord - y_max).abs() <= 0.5 * dy

        p_inlet_mse = (p[inlet] - p_bc[inlet]).pow(2).mean() if inlet.any() else torch.zeros((), device=p.device, dtype=p.dtype)
        p_outlet_mse = p[outlet].mean().pow(2) if outlet.any() else torch.zeros((), device=p.device, dtype=p.dtype)
        bc_mse = p_inlet_mse + p_outlet_mse

        # Return channel-first fields for drop-in compatibility
        return {
            "Rx": Rx.unsqueeze(1),
            "Ry": Ry.unsqueeze(1),
            "Rc": Rc.unsqueeze(1),
            "div_u": div_u.unsqueeze(1),
            "div_eps_u": div_eps_u.unsqueeze(1),
            "mom_mse": mom_mse,
            "cont_mse": cont_mse,
            "mom_mse_full": mom_mse_full,
            "cont_mse_full": cont_mse_full,
            "bc_mse": bc_mse,
            "p_inlet_mse": p_inlet_mse,
            "p_outlet_mse": p_outlet_mse,
        }

    def forward(
        self,
        pred: torch.Tensor,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        # ------------------------------------------------------------
        # 1) Data loss (normalized space)
        # ------------------------------------------------------------
        data = self.data_loss(pred, y)

        # ------------------------------------------------------------
        # 2) Denormalize to physical space
        # ------------------------------------------------------------
        if self.in_normalizer is None or self.out_normalizer is None:
            msg = "PINO loss: normalizers not set. Call set_normalizers(in_normalizer=..., out_normalizer=...)."
            raise RuntimeError(msg)

        x_phys = self.in_normalizer.inverse_transform(x)
        pred_phys = self.out_normalizer.inverse_transform(pred)

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

        x_coord = x_phys[:, self.iidx["x"]]
        y_coord = x_phys[:, self.iidx["y"]]
        dx_t, dy_t = _infer_dx_dy(x_coord, y_coord)

        Kxx = 10.0 ** x_phys[:, self.iidx["kxx"]]
        Kyy = 10.0 ** x_phys[:, self.iidx["kyy"]]

        kxy_hat = x_phys[:, self.iidx["kxy"]].clamp(-self._kxy_hat_clip, self._kxy_hat_clip)
        K0 = torch.sqrt((Kxx * Kyy).clamp_min(self._eps_K0))  # (B,H,W)

        # ------------------------------------------------------------
        # 5) Derivatives
        # ------------------------------------------------------------
        dpdx, dpdy = self.backend.grad(p, dx_t, dy_t)
        dudx, dudy = self.backend.grad(u, dx_t, dy_t)
        dvdx, dvdy = self.backend.grad(v, dx_t, dy_t)

        div_u = dudx + dvdy

        # ------------------------------------------------------------
        # 6) Brinkman viscous stress (COMSOL-like deviatoric)
        # ------------------------------------------------------------
        coef = MU_AIR / eps
        tau_xx = coef * (2.0 * dudx - (2.0 / 3.0) * div_u)
        tau_yy = coef * (2.0 * dvdy - (2.0 / 3.0) * div_u)
        tau_xy = coef * (dudy + dvdx)

        div_tau_x = self.backend.div(tau_xx, tau_xy, dx_t, dy_t)
        div_tau_y = self.backend.div(tau_xy, tau_yy, dx_t, dy_t)

        # ------------------------------------------------------------
        # 7) Darcy drag (consistent K^{-1} construction)
        # ------------------------------------------------------------
        kxx_hat = Kxx / K0
        kyy_hat = Kyy / K0

        det_hat = kxx_hat * kyy_hat - kxy_hat * kxy_hat  # equals 1 - kxy_hat^2
        det_hat_safe = det_hat.clamp_min(self._det_hat_min)

        invhat_xx = kyy_hat / det_hat_safe
        invhat_xy = -kxy_hat / det_hat_safe
        invhat_yy = kxx_hat / det_hat_safe

        inv_xx = invhat_xx / K0
        inv_xy = invhat_xy / K0
        inv_yy = invhat_yy / K0

        drag_x = MU_AIR * (inv_xx * u + inv_xy * v)
        drag_y = MU_AIR * (inv_xy * u + inv_yy * v)

        # ------------------------------------------------------------
        # 8) Residuals
        # ------------------------------------------------------------
        Rx = -dpdx + div_tau_x - drag_x
        Ry = -dpdy + div_tau_y - drag_y

        if self.continuity_mode == "eps_u":
            Rc = self.backend.div(eps * u, eps * v, dx_t, dy_t)  # div(eps u)
            rc_log_key = "physics/div_eps_u_l2"
        elif self.continuity_mode == "u":
            Rc = self.backend.div(u, v, dx_t, dy_t)  # div(u)
            rc_log_key = "physics/div_u_l2"
        else:
            msg = f"Unknown continuity_mode: {self.continuity_mode}"
            raise ValueError(msg)

        # ------------------------------------------------------------
        # 9) Physics loss (optionally interior-only)
        # ------------------------------------------------------------
        pad = self.interior_pad
        Rx_i = _interior(Rx, pad)
        Ry_i = _interior(Ry, pad)
        Rc_i = _interior(Rc, pad)

        phys = Rx_i.pow(2).mean() + Ry_i.pow(2).mean() + Rc_i.pow(2).mean()

        # ------------------------------------------------------------
        # 10) Pressure BC loss (inlet + outlet, y-based masks)
        # ------------------------------------------------------------
        dy = dy_t  # scalar tensor
        y_min = y_coord.amin(dim=(-2, -1), keepdim=True)
        y_max = y_coord.amax(dim=(-2, -1), keepdim=True)

        inlet = (y_coord - y_min).abs() <= 0.5 * dy
        outlet = (y_coord - y_max).abs() <= 0.5 * dy

        p_inlet_loss = (p[inlet] - p_bc[inlet]).pow(2).mean() if inlet.any() else torch.zeros((), device=p.device, dtype=p.dtype)
        p_outlet_loss = p[outlet].mean().pow(2) if outlet.any() else torch.zeros((), device=p.device, dtype=p.dtype)

        p_bc_loss = p_inlet_loss + p_outlet_loss

        # ------------------------------------------------------------
        # 11) Total loss
        # ------------------------------------------------------------
        total = data + self.lambda_phys * phys + self.lambda_p * p_bc_loss

        # ------------------------------------------------------------
        # 12) Logging
        # ------------------------------------------------------------
        if wandb is not None and getattr(wandb, "run", None) is not None and (self.step % self.log_every == 0):
            total_val = total.item()
            log_dict: dict[str, float] = {
                "loss/total": total_val,
                "loss/data": data.item(),
                "loss/phys": phys.item(),
                "loss/p_bc": p_bc_loss.item(),
                "loss/frac_loss_data": data.item() / max(total_val, 1e-30),
                "loss/frac_loss_phys": (self.lambda_phys * phys.item()) / max(total_val, 1e-30),
                "loss/frac_loss_p_bc": (self.lambda_p * p_bc_loss.item()) / max(total_val, 1e-30),
                "physics/Rx_l2": Rx.pow(2).mean().sqrt().item(),
                "physics/Ry_l2": Ry.pow(2).mean().sqrt().item(),
                rc_log_key: Rc.pow(2).mean().sqrt().item(),
            }
            wandb.log(log_dict, commit=False)

        self.step += 1
        return total


# =============================================================================
# Public wrappers (4 options)
# =============================================================================
class PINOPhysicalLossEps(_PINOBrinkmanLossBase):
    """Physical derivatives + conservative continuity div(eps u)."""

    def __init__(
        self,
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_phys: float,
        lambda_p: float,
        in_normalizer: Any | None = None,
        out_normalizer: Any | None = None,
        *,
        interior_pad: int = 0,
        log_every: int = 10,
    ) -> None:
        """
        Initialize the PINOPhysicalLossEps.

        Parameters
        ----------
        data_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The data loss function to use (e.g., H1Loss).
        lambda_phys : float
            Weight for the physics loss term.
        lambda_p : float
            Weight for the pressure boundary condition loss term.
        in_normalizer : Any | None, optional
            Normalizer for input data, by default None.
        out_normalizer : Any | None, optional
            Normalizer for output data, by default None.
        interior_pad : int, optional
            Padding size for interior loss calculation, by default 0.
        log_every : int, optional
            Frequency of logging, by default 10.

        """
        super().__init__(
            backend=_PhysicalDerivatives(),
            continuity_mode="eps_u",
            data_loss=data_loss,
            lambda_phys=lambda_phys,
            lambda_p=lambda_p,
            in_normalizer=in_normalizer,
            out_normalizer=out_normalizer,
            interior_pad=interior_pad,
            log_every=log_every,
        )


class PINOPhysicalLossDiv(_PINOBrinkmanLossBase):
    """Physical derivatives + plain continuity div(u)."""

    def __init__(
        self,
        data_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_phys: float,
        lambda_p: float,
        in_normalizer: Any | None = None,
        out_normalizer: Any | None = None,
        *,
        interior_pad: int = 0,
        log_every: int = 10,
    ) -> None:
        """
        Initialize the PINOPhysicalLossDiv.

        Parameters
        ----------
        data_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The data loss function to use (e.g., H1Loss).
        lambda_phys : float
            Weight for the physics loss term.
        lambda_p : float
            Weight for the pressure boundary condition loss term.
        in_normalizer : Any | None, optional
            Normalizer for input data, by default None.
        out_normalizer : Any | None, optional
            Normalizer for output data, by default None.
        interior_pad : int, optional
            Padding size for interior loss calculation, by default 0.
        log_every : int, optional
            Frequency of logging, by default 10.

        """
        super().__init__(
            backend=_PhysicalDerivatives(),
            continuity_mode="u",
            data_loss=data_loss,
            lambda_phys=lambda_phys,
            lambda_p=lambda_p,
            in_normalizer=in_normalizer,
            out_normalizer=out_normalizer,
            interior_pad=interior_pad,
            log_every=log_every,
        )


class PINOSpectralLossEps(_PINOBrinkmanLossBase):
    """Spectral derivatives + conservative continuity div(eps u)."""

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
        Initialize the PINOSpectralLossEps.

        Parameters
        ----------
        data_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The data loss function to use (e.g., H1Loss).
        lambda_phys : float
            Weight for the physics loss term.
        lambda_p : float
            Weight for the pressure boundary condition loss term.
        in_normalizer : Any | None, optional
            Normalizer for input data, by default None.
        out_normalizer : Any | None, optional
            Normalizer for output data, by default None.
        grad_mode : Literal["fft", "fft_reflect"], optional
            Gradient computation mode, by default "fft_reflect".
        interior_pad : int, optional
            Padding size for interior loss calculation, by default 2.
        log_every : int, optional
            Frequency of logging, by default 10.

        """
        super().__init__(
            backend=_SpectralDerivatives(grad_mode=grad_mode),
            continuity_mode="eps_u",
            data_loss=data_loss,
            lambda_phys=lambda_phys,
            lambda_p=lambda_p,
            in_normalizer=in_normalizer,
            out_normalizer=out_normalizer,
            interior_pad=interior_pad,
            log_every=log_every,
        )


class PINOSpectralLossDiv(_PINOBrinkmanLossBase):
    """Spectral derivatives + plain continuity div(u)."""

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
        Initialize the PINOSpectralLossDiv.

        Parameters
        ----------
        data_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The data loss function to use (e.g., H1Loss).
        lambda_phys : float
            Weight for the physics loss term.
        lambda_p : float
            Weight for the pressure boundary condition loss term.
        in_normalizer : Any | None, optional
            Normalizer for input data, by default None.
        out_normalizer : Any | None, optional
            Normalizer for output data, by default None.
        grad_mode : Literal["fft", "fft_reflect"], optional
            Gradient computation mode, by default "fft_reflect".
        interior_pad : int, optional
            Padding size for interior loss calculation, by default 2.
        log_every : int, optional
            Frequency of logging, by default 10.

        """
        super().__init__(
            backend=_SpectralDerivatives(grad_mode=grad_mode),
            continuity_mode="u",
            data_loss=data_loss,
            lambda_phys=lambda_phys,
            lambda_p=lambda_p,
            in_normalizer=in_normalizer,
            out_normalizer=out_normalizer,
            interior_pad=interior_pad,
            log_every=log_every,
        )
