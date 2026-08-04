"""
===============================================================================
cli_device.py
===============================================================================
Define the shared semantic runtime-device option used by command modules.

Responsibilities:
  - Expose the exact auto, cuda, and cpu CLI vocabulary
  - Keep device help semantics consistent across training, Optuna, and artifacts
  - Avoid importing Torch while importing or constructing command parsers

Design principles:
  - One dependency-light helper owns the exact command-line vocabulary
  - Help text distinguishes fallback, strict CUDA, and CPU-only policies
  - Parsing records requested policy without resolving hardware

This module does NOT:
  - Validate or resolve runtime hardware. ``learning.device`` owns that boundary
  - Apply service-specific overrides. Each command entry point owns forwarding
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.learning.learning_device_policy import DEVICE_POLICIES

if TYPE_CHECKING:
    import argparse


def add_device_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None,
    help_prefix: str = "Runtime device policy",
) -> None:
    """
    Add the exact shared runtime-device option to one parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Command parser receiving the ``--device`` option.
    default : str | None
        Default policy. ``None`` preserves the service config value. Artifact
        generation supplies ``auto`` because it has no experiment YAML input.
    help_prefix : str, optional
        Service-specific prefix prepended to the shared policy semantics.

    """
    parser.add_argument(
        "--device",
        choices=DEVICE_POLICIES,
        default=default,
        help=(f"{help_prefix}: auto chooses CUDA when usable, otherwise CPU. cuda is strict and never falls back. cpu avoids CUDA use"),
    )
