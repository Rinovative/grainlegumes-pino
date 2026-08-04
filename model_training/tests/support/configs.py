"""Build compact artificial configuration requests owned by the tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def direct_config(
    *,
    model_kind: str = "fno",
    physics_enabled: bool = False,
    device: str = "cpu",
    wandb_mode: str = "disabled",
    workflow: str = "train",
    suffix: str | None = "fixture",
) -> dict[str, Any]:
    """Return a complete, artificial direct-training request owned by the tests."""
    if model_kind == "fno":
        model_params: dict[str, Any] = {
            "n_modes": [4, 5],
            "hidden_channels": 8,
            "n_layers": 2,
        }
    elif model_kind == "uno":
        model_params = {
            "modes_x": 4,
            "modes_y": 5,
            "hidden_channels": 8,
            "n_layers": 5,
            "mode_ratio": 0.5,
        }
    else:
        message = f"Unsupported synthetic model kind: {model_kind!r}"
        raise ValueError(message)

    physics: dict[str, Any] = {"enabled": False}
    if physics_enabled:
        physics = {
            "enabled": True,
            "derivatives": {"kind": "physical", "extension": "none"},
            "interior_crop": 1,
            "continuity": "div_velocity",
            "residual_weight": {
                "target": 2.0e-3,
                "warmup": {"kind": "linear", "epochs": 2},
            },
            "boundary_weight": {
                "target": 3.0e-3,
                "warmup": {"kind": "linear", "epochs": 2},
            },
        }

    return {
        "task": "steady_flow",
        "run": {
            "seed": 17,
            "deterministic": True,
            "device": device,
            "suffix": suffix,
        },
        "data": {
            "train_dataset": "synthetic_train",
            "ood_datasets": ["synthetic_ood"],
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {"kind": model_kind, "params": model_params},
        "loss": {
            "data": {"kind": "relative_h1", "space": "normalized", "weight": 1.0},
            "physics": physics,
        },
        "evaluation": {"objective": {"id": "normalized_group_macro_rmse"}},
        "optimizer": {"kind": "adamw", "lr": 1.0e-3, "weight_decay": 0.0},
        "scheduler": {"kind": "reduce_on_plateau", "factor": 0.5, "patience": 2},
        "training": {
            "epochs": 3,
            "evaluation_interval": 1,
            "ood_evaluation_interval": 1,
            "mixed_precision": False,
        },
        "tracking": {
            "wandb": {
                "mode": wandb_mode,
                "workflow": workflow,
                "monitor": {"interval": 1, "max_cases": 2},
                "upload": {"evaluation_artifacts": False},
            }
        },
    }


def optuna_config(
    *,
    model_kind: str = "fno",
    physics_enabled: bool = False,
    multivariate: bool = True,
    role: str | None = None,
) -> dict[str, Any]:
    """Return a compact artificial Optuna request with explicit search semantics."""
    study: dict[str, Any] = {
        "name": f"synthetic_{model_kind}_study",
        "seed": 29,
        "n_trials": 2,
        "sampler": {"kind": "tpe", "multivariate": multivariate},
        "pruner": {
            "kind": "median",
            "n_startup_trials": 1,
            "n_warmup_steps": 1,
            "interval_steps": 1,
        },
    }
    if role is not None:
        study["role"] = role

    experiment = direct_config(
        model_kind=model_kind,
        physics_enabled=physics_enabled,
        workflow="optuna_trial",
        suffix="optuna",
    )
    return {
        "study": study,
        "experiment": experiment,
        "search_space": {
            "model.params.hidden_channels": {
                "name": "hidden_channels",
                "kind": "categorical",
                "values": [8, 12],
            },
            "optimizer.lr": {
                "name": "learning_rate",
                "kind": "float",
                "low": 1.0e-4,
                "high": 1.0e-2,
                "log": True,
            },
        },
    }


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one test-owned YAML request and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path
