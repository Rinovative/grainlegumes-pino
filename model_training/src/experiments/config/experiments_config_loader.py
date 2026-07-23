"""
===============================================================================
experiments_config_loader.py
===============================================================================
Load, merge and resolve experiment configuration dictionaries.

Responsibilities:
  - Load YAML experiment files
  - Merge user configs with task and component defaults
  - Resolve path roots and generated run names
  - Build dataloaders from resolved data settings

Design principles:
  - Input YAMLs stay minimal
  - Effective configs are fully expanded
  - Required sections fail fast when missing

Boundaries:
  - Defaults belong to experiments.config.defaults
  - Model, loss and optimizer construction belong to learning factories
  - Training execution belongs to learning.training.loop
===============================================================================
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from src import common, datasets

from . import experiments_config_defaults as config_defaults

LOSS_DEFAULTS = config_defaults.LOSS_DEFAULTS
MODEL_DEFAULTS = config_defaults.MODEL_DEFAULTS
OPTIMIZER_DEFAULTS = config_defaults.OPTIMIZER_DEFAULTS
SCHEDULER_DEFAULTS = config_defaults.SCHEDULER_DEFAULTS
get_task_defaults = config_defaults.get_task_defaults


class _YamlModule(Protocol):
    """Minimal PyYAML surface used by this module."""

    def safe_load(self, stream: TextIO) -> Any:
        """Load YAML from a text stream."""

    def dump(
        self,
        data: Any,
        stream: TextIO,
        *,
        default_flow_style: bool,
        sort_keys: bool,
    ) -> Any:
        """Write YAML to a text stream."""


yaml = cast("_YamlModule", importlib.import_module("yaml"))
_FNO_MODE_DIMENSIONS = 2


def load_yaml(path: Path | str) -> dict[str, Any]:
    """
    Load a YAML file.

    Parameters
    ----------
    path : Path | str
        Path to the YAML file

    Returns
    -------
    dict[str, Any]
        Parsed YAML content

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    yaml.YAMLError
        If the YAML is malformed

    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(config: dict[str, Any], path: Path | str) -> None:
    """
    Save a config dictionary to a YAML file.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary
    path : Path | str
        Path where to save the YAML file

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge override into base dictionary.

    Override values take precedence. Recursive for nested dicts.

    Parameters
    ----------
    base : dict[str, Any]
        Base configuration
    override : dict[str, Any]
        Override values

    Returns
    -------
    dict[str, Any]
        Merged configuration

    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def generate_run_name(config: dict[str, Any]) -> str:
    """
    Generate a unique run name from config parameters.

    Pattern: <prefix>__<task>__<model>__<key_params>__s<seed>__<suffix>

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary

    Returns
    -------
    str
        Generated run name

    """
    task = config["task"]
    arch = config["model"]["architecture"]
    seed = config["run"]["seed"]
    prefix = config["run"].get("prefix")
    suffix = config["run"].get("suffix")

    # Model key parameters
    model_params = config["model"].get("params", {})
    if arch in ("FNO", "PI-FNO"):
        n_modes = model_params.get("n_modes")
        if not isinstance(n_modes, (list, tuple)) or len(n_modes) != _FNO_MODE_DIMENSIONS:
            msg = f"FNO run name requires model.params.n_modes with two entries, got: {n_modes!r}"
            raise ValueError(msg)
        h_ch = model_params.get("hidden_channels", 0)
        n_layers = model_params.get("n_layers", 0)
        model_key = f"{arch.lower()}_m{n_modes[0]}x{n_modes[1]}_h{h_ch}_l{n_layers}"
    elif arch in ("UNO", "PI-UNO"):
        h_ch = model_params.get("hidden_channels", 0)
        n_layers = model_params.get("n_layers", 0)
        model_key = f"{arch.lower()}_h{h_ch}_l{n_layers}"
    else:
        model_key = arch.lower()

    name_parts = [task, model_key, f"s{seed}"]
    if prefix:
        name_parts.insert(0, str(prefix))
    if suffix:
        name_parts.append(suffix)

    return "__".join(name_parts)


def resolve_config(user_config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve an experiment configuration dictionary with all defaults.

    Parameters
    ----------
    user_config : dict[str, Any]
        Experiment configuration dictionary

    Returns
    -------
    dict[str, Any]
        Fully expanded effective configuration

    Raises
    ------
    KeyError
        If required task or config sections missing

    """
    # Get task
    if "task" not in user_config:
        msg = "Missing required 'task' in config"
        raise KeyError(msg)
    task = user_config["task"]

    # Get task defaults
    task_defaults = get_task_defaults(task)

    # Start with task defaults and merge user config
    effective_config = deep_merge(task_defaults, user_config)

    # Merge model defaults if not present
    model_arch = effective_config["model"]["architecture"]
    if model_arch in MODEL_DEFAULTS:
        model_user = effective_config["model"].copy()
        model_defaults = MODEL_DEFAULTS[model_arch].copy()
        effective_config["model"] = deep_merge(model_defaults, model_user)

    # Ensure in/out channels are set
    if "in_channels" not in effective_config["model"]["params"]:
        effective_config["model"]["params"]["in_channels"] = effective_config["in_channels"]
    if "out_channels" not in effective_config["model"]["params"]:
        effective_config["model"]["params"]["out_channels"] = effective_config["out_channels"]

    # Merge loss defaults
    loss_type = effective_config["loss"].get("type", "supervised")
    if loss_type in LOSS_DEFAULTS:
        effective_config["loss"] = deep_merge(LOSS_DEFAULTS[loss_type], effective_config["loss"])

    # Merge optimizer defaults
    opt_type = effective_config["optimizer"].get("type", "adamw")
    if opt_type in OPTIMIZER_DEFAULTS:
        opt_user = effective_config["optimizer"].copy()
        opt_defaults = OPTIMIZER_DEFAULTS[opt_type].copy()
        effective_config["optimizer"] = deep_merge(opt_defaults, opt_user)

    # Merge scheduler defaults
    sched_type = effective_config["scheduler"].get("type", "reduce_on_plateau")
    if sched_type in SCHEDULER_DEFAULTS:
        sched_user = effective_config["scheduler"].copy()
        sched_defaults = SCHEDULER_DEFAULTS[sched_type].copy()
        effective_config["scheduler"] = deep_merge(sched_defaults, sched_user)

    # Add path roots to config
    effective_config["paths"] = {
        "project_root": str(common.paths.get_project_root()),
        "storage_root": str(common.paths.get_storage_root()),
        "data_root": str(common.paths.get_data_root()),
        "train_root": str(common.paths.get_train_root()),
    }

    # Generate run name if not present
    if "name" not in effective_config["run"]:
        effective_config["run"]["name"] = generate_run_name(effective_config)

    return effective_config


def load_and_resolve_config(yaml_path: Path | str) -> dict[str, Any]:
    """
    Load experiment YAML and resolve to effective config with all defaults.

    Parameters
    ----------
    yaml_path : Path | str
        Path to experiment YAML file

    Returns
    -------
    dict[str, Any]
        Fully expanded effective configuration

    Raises
    ------
    FileNotFoundError
        If YAML file not found
    KeyError
        If required task or config sections missing
    yaml.YAMLError
        If YAML is malformed

    """
    return resolve_config(load_yaml(yaml_path))


def create_dataloaders_from_config(
    config: dict[str, Any],
    *,
    split_indices: dict[str, Any] | None = None,
    data_processor: Any | None = None,
) -> dict[str, Any]:
    """
    Create dataloaders from config dictionary.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved configuration dictionary with data section
    split_indices : dict[str, Any] | None, optional
        Previously saved split membership to reuse. When provided, no new
        train/eval/OOD membership is generated.
    data_processor : Any | None, optional
        Previously restored data processor to reuse. When provided, the
        dataset layer must not fit replacement normalizers.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: train, eval, data_processor, split_indices

    """
    data_cfg = config.get("data", {})
    dataset_root = Path(config["paths"]["train_root"])

    # Resolve dataset paths
    train_dataset_name = str(data_cfg.get("train_dataset", "lhs_var80_seed3001"))
    ood_datasets = data_cfg.get("ood_datasets") or ["lhs_var120_seed4001"]
    if not isinstance(ood_datasets, list) or not ood_datasets:
        msg = f"data.ood_datasets must be a non-empty list, got: {ood_datasets!r}"
        raise ValueError(msg)
    ood_dataset_name = str(ood_datasets[0])

    path_train = dataset_root / train_dataset_name / f"{train_dataset_name}.pt"
    path_test_ood = dataset_root / ood_dataset_name / f"{ood_dataset_name}.pt"

    # Extract dataloader config
    dataloader_cfg = {
        "batch_size": data_cfg.get("batch_size", 32),
        "num_workers": data_cfg.get("num_workers", 8),
        "pin_memory": data_cfg.get("pin_memory", True),
        "persistent_workers": data_cfg.get("persistent_workers", True),
    }

    # Call real create_dataloaders
    train_loader, test_loaders, normalizer, split_indices = datasets.base.create_dataloaders(
        dataset_cls=datasets.simulation.PhysicsDataset,
        path_train=str(path_train),
        path_test_ood=str(path_test_ood),
        train_ratio=data_cfg.get("train_ratio", 0.8),
        ood_fraction=data_cfg.get("ood_fraction", 0.2),
        split_seed=config.get("run", {}).get("seed", 9),
        split_indices=split_indices,
        data_processor=data_processor,
        **dataloader_cfg,
    )

    # Return in expected format
    eval_loader = test_loaders.get("eval")
    if eval_loader is None and test_loaders:
        eval_loader = next(iter(test_loaders.values()))
    if eval_loader is None:
        msg = "No evaluation dataloader was created."
        raise ValueError(msg)

    return {
        "train": train_loader,
        "eval": eval_loader,
        "data_processor": normalizer,
        "split_indices": split_indices,
    }
