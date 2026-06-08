"""
===============================================================================
learning_training_hooks.py
===============================================================================
Provide training diagnostics hooks for neural-operator models.

Responsibilities:
  - Register PyTorch forward hooks
  - Extract and aggregate spectral energy diagnostics
  - Support FNO, PI-FNO and UNO diagnostics

Design principles:
  - Hooks are lightweight and non-invasive
  - Model internals are observed rather than modified
  - Diagnostics stay independent of loss computation

Boundaries:
  - Training-loop execution belongs to learning.training.loop
  - Loss computation belongs to learning.losses
===============================================================================
"""

from collections import defaultdict

import torch


class SpectralEnergyHook:
    """
    Hook to compute average spectral energy distributions of spectral operator layer outputs.

    Usage:
        hook = SpectralEnergyHook()
        for layer in fno_model.fourier_layers:
            layer.register_forward_hook(hook.hook)
        # After training and evaluation:
        spectral_energy = hook.aggregate()

    Attributes
    ----------
    energy : defaultdict[int, list[torch.Tensor]]
        Stores spectral energy distributions for each hooked layer.

    Methods
    -------
    reset()
        Resets the stored energy distributions.
    hook(module, inputs, output)
        Forward hook to compute and store spectral energy of layer outputs.
    aggregate()
        Aggregates and returns the average spectral energy distributions.

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping from layer IDs to their average spectral energy distributions.

    """

    def __init__(self) -> None:
        """Initialize the SpectralEnergyHook."""
        self.reset()

    def reset(self) -> None:
        """Reset the stored energy distributions."""
        self.energy: defaultdict[int, list[torch.Tensor]] = defaultdict(list)

    def hook(self, module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """
        Forward hook to compute and store spectral energy of layer outputs.

        Parameters
        ----------
        module : torch.nn.Module
            The module to which the hook is attached.
        _inputs : tuple[torch.Tensor, ...]
            Input tensors to the module.
        output : torch.Tensor
            Output tensor from the module.

        """
        # output: [B, C, Nx, Ny]
        with torch.no_grad():
            spatial_dims = tuple(range(2, output.ndim))
            # Remove DC per-sample, per-channel (reduces domination of k~0 bin)
            x = output - output.mean(dim=spatial_dims, keepdim=True)
            fft = torch.fft.rfftn(x, dim=spatial_dims)
            power = fft.real**2 + fft.imag**2
            power = power.mean(dim=(0, 1))  # avg over batch + channels
            self.energy[id(module)].append(power.cpu())

    def aggregate(self) -> dict[int, torch.Tensor]:
        """
        Aggregate and return the average spectral energy distributions.

        Returns
        -------
        dict
            Mapping from layer IDs to their average spectral energy distributions.

        """
        return {layer_id: torch.stack(vals).mean(dim=0) for layer_id, vals in self.energy.items()}
