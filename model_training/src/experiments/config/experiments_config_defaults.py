"""
===============================================================================
experiments_config_defaults.py
===============================================================================
Define task and component defaults for experiment configs.

Responsibilities:
  - Define task defaults for supported problem types
  - Map tasks to standard datasets, channels and field selections
  - Provide model, loss, optimizer, scheduler and training defaults

Design principles:
  - Defaults are declarative data
  - User YAML values override defaults
  - Resolved configs are complete enough for reproducible runs

Boundaries:
  - YAML parsing and saving belong to experiments.config.loader
  - Training execution belongs to learning.training.loop
===============================================================================
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Task defaults
# =============================================================================
#
# Each task defines:
# - input/output fields for model
# - default train and OOD datasets
# - task-specific metrics and preprocessing
#
TASK_DEFAULTS = {
    "steady_flow": {
        # Data handling
        "data": {
            "train_dataset": "lhs_var80_seed3001",
            "ood_datasets": ["lhs_var120_seed4001"],
            "train_ratio": 0.8,
            "ood_fraction": 0.2,
            "batch_size": 32,
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
        },
        # Model channels (7 inputs: x, y, kxx, kyy, kxy, phi, p_bc; 3 outputs: p, u, v)
        "in_channels": 7,
        "out_channels": 3,
        "input_fields": [
            "x",
            "y",
            "kxx",
            "kyy",
            "kxy",
            "phi",
            "p_bc",
        ],
        "output_fields": ["p", "u", "v"],
        # Training defaults
        "training": {
            "n_epochs": 600,
            "eval_interval": 5,
            "mixed_precision": False,
            "save_best_metric": "eval_overall_rmse",
        },
        # Evaluation defaults
        "evaluation": {
            "losses": {
                "h1": True,
                "l2": True,
                "overall_rmse": True,
                "rmse_p_pa": True,
                "rmse_u_ms": True,
                "rmse_v_ms": True,
            },
        },
    },
}


# =============================================================================
# Model architecture defaults
# =============================================================================
#
# FNO and UNO have sensible defaults. These are used if not specified in YAML.
#
MODEL_DEFAULTS = {
    "FNO": {
        "params": {
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "fno_skip": "linear",
            "channel_mlp_skip": "soft-gating",
            "implementation": "factorized",
        },
    },
    "PI-FNO": {
        "params": {
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
            "fno_skip": "linear",
            "channel_mlp_skip": "soft-gating",
            "implementation": "factorized",
        },
    },
    "UNO": {
        # UNO mode schedule and scalings are architecture-dependent
        # Set in code based on n_layers
    },
    "PI-UNO": {
        # Same as UNO
    },
}


# =============================================================================
# Loss defaults
# =============================================================================
#
LOSS_DEFAULTS = {
    "supervised": {
        "type": "supervised",
        "data_loss": "h1",
    },
    "pino": {
        "type": "pino",
        "data_loss": "h1",
    },
}


# =============================================================================
# Optimizer defaults
# =============================================================================
#
OPTIMIZER_DEFAULTS = {
    "adamw": {
        "type": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-6,
    },
}


# =============================================================================
# Scheduler defaults
# =============================================================================
#
SCHEDULER_DEFAULTS = {
    "reduce_on_plateau": {
        "type": "reduce_on_plateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 20,
        "min_lr": 1e-8,
    },
}


def get_task_defaults(task: str) -> dict[str, Any]:
    """
    Get all defaults for a specific task.

    Parameters
    ----------
    task : str
        Task name (e.g., "steady_flow")

    Returns
    -------
    dict[str, Any]
        Task defaults dictionary

    Raises
    ------
    KeyError
        If task is not recognized

    """
    if task not in TASK_DEFAULTS:
        msg = f"Unknown task: {task}. Available tasks: {list(TASK_DEFAULTS.keys())}"
        raise KeyError(msg)
    return TASK_DEFAULTS[task]
