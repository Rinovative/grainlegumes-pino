"""
===============================================================================
experiments_config_defaults.py
===============================================================================
Define generic runtime defaults around task-owned semantic contracts.

Responsibilities:
  - Provide generic run, data-loader, optimizer, scheduler, and training defaults
  - Project task dataset fallbacks, losses, and metric definitions into config defaults
  - Return serializable defaults for strict experiment resolution

Design principles:
  - Explicit experiment data selection overrides TaskSpec dataset fallbacks
  - Task-fixed scientific semantics come exclusively from the registered TaskSpec
  - Runtime defaults remain independent of concrete task field names
  - Semantic identifiers are distinct from Python implementation class names

This module does NOT:
  - Parse or strictly validate user config. ``experiments.config.loader`` owns admission
  - Construct models, losses, metrics, or physics. Their registries own construction
  - Define dataset storage schemas or lifecycle behavior
===============================================================================
"""

from __future__ import annotations

from typing import Any

from src import domain

RUN_DEFAULTS: dict[str, Any] = {
    "seed": 9,
    "deterministic": True,
    "device": "auto",
    "suffix": None,
}

DATA_RUNTIME_DEFAULTS: dict[str, Any] = {
    "train_ratio": 0.8,
    "ood_fraction": 0.2,
    "batch_size": 32,
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
}

PHYSICS_LOSS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "derivatives": {
        "kind": "spectral",
        "extension": "reflect",
    },
    "interior_crop": 2,
    "residual_weight": {
        "target": 1.0e-4,
        "warmup": {"kind": "linear", "epochs": 50},
    },
    "boundary_weight": {
        "target": 5.0e-4,
        "warmup": {"kind": "linear", "epochs": 50},
    },
}

OPTIMIZER_DEFAULTS: dict[str, dict[str, Any]] = {
    "adamw": {
        "kind": "adamw",
        "betas": [0.9, 0.999],
        "second_moment_floor": 1.0e-6,
    }
}

SCHEDULER_DEFAULTS: dict[str, dict[str, Any]] = {
    "reduce_on_plateau": {
        "kind": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 20,
        "min_lr": 1.0e-8,
    }
}

TRAINING_DEFAULTS: dict[str, Any] = {
    "epochs": 600,
    "evaluation_interval": 5,
    "ood_evaluation_interval": 5,
    "mixed_precision": False,
}

WANDB_ENTITY = "Rinovative-Hub"
WANDB_REPOSITORY_PROJECTS = {
    "airflow": "grainlegumes-pino-airflow",
    "drying": "grainlegumes-pino-drying",
}
WANDB_TASK_PROJECTS = {
    "steady_flow": WANDB_REPOSITORY_PROJECTS["airflow"],
}
WANDB_MAX_TAGS = 2
WANDB_WORKFLOWS = (
    "train",
    "optuna_trial",
    "gpu_smoke",
    "cpu_acceptance",
    "tracking_validation",
)

TRACKING_DEFAULTS: dict[str, Any] = {
    "wandb": {
        "mode": "online",
        "workflow": "train",
        "study": None,
        "project": WANDB_TASK_PROJECTS["steady_flow"],
        "entity": WANDB_ENTITY,
        "tags": [],
        "monitor": {
            "enabled": True,
            "interval": 5,
            "max_cases": 4,
        },
        "upload": {
            "evaluation_artifacts": False,
        },
    }
}


def get_task_defaults(task_id: str) -> dict[str, Any]:
    """
    Project one registered task into generic experiment defaults.

    Parameters
    ----------
    task_id : str
        Exact registered task identifier.

    Returns
    -------
    dict[str, Any]
        Serializable runtime defaults with task dataset fallbacks and semantics.

    Raises
    ------
    ValueError
        If `task_id` is not registered.

    """
    task = domain.tasks.registry.get_task(task_id)
    return {
        "run": RUN_DEFAULTS,
        "data": {
            "train_dataset": task.default_datasets.train,
            "ood_datasets": list(task.default_datasets.ood),
            **DATA_RUNTIME_DEFAULTS,
        },
        "loss": {
            "data": {
                "kind": task.data_losses[0],
                "space": "normalized",
                "weight": 1.0,
            },
            "physics": {
                **PHYSICS_LOSS_DEFAULTS,
                "continuity": task.physics.continuity,
            },
        },
        "evaluation": {
            "metrics": [metric.as_dict(all_fields=task.output_names) for metric in task.default_metrics],
        },
        "optimizer": OPTIMIZER_DEFAULTS["adamw"],
        "scheduler": SCHEDULER_DEFAULTS["reduce_on_plateau"],
        "training": TRAINING_DEFAULTS,
        "tracking": TRACKING_DEFAULTS,
    }
