"""
===============================================================================
common_paths.py
===============================================================================
Resolve project, storage, dataset, run and artifact paths.

Responsibilities:
  - Read project and storage root environment variables
  - Provide stable dataset and run-directory resolvers
  - Resolve split-index, normalizer, checkpoint and artifact paths

Design principles:
  - Environment variables are the canonical path source
  - Defaults keep local notebook and Docker execution usable
  - Callers pass logical names instead of hardcoded storage paths

Boundaries:
  - Dataset membership belongs to datasets and training code
  - Experiment semantics belong to experiments.config

Notes:
  Fallback paths:
    - missing environment variables resolve relative to the repository and storage roots
===============================================================================

"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory from environment or default."""
    root = os.environ.get("PROJECT_ROOT")
    if root:
        return Path(root)
    # Fallback: navigate up from this file
    return Path(__file__).parent.parent.parent.parent


def get_storage_root() -> Path:
    """Get the storage root directory from environment or default."""
    root = os.environ.get("STORAGE_ROOT")
    if root:
        return Path(root)
    # Fallback: sibling directory to repository
    return get_project_root().parent / "storage"


def get_data_root() -> Path:
    """Get the data root directory from environment or default."""
    root = os.environ.get("DATA_ROOT")
    if root:
        return Path(root)
    return get_storage_root() / "data"


def get_train_root() -> Path:
    """Get the training data root directory from environment or default."""
    root = os.environ.get("TRAIN_ROOT")
    if root:
        return Path(root)
    return get_storage_root() / "data_training"


def get_gen_root() -> Path:
    """Get the data generation root directory from environment or default."""
    root = os.environ.get("GEN_ROOT")
    if root:
        return Path(root)
    return get_storage_root() / "data_generation"


def resolve_dataset_path(dataset_name: str, task: str | None = None) -> Path:
    """
    Resolve a dataset name to its physical path.

    For training datasets, the dataset is located under TRAIN_ROOT.
    For shared/raw datasets, the dataset is located under DATA_ROOT.

    Parameters
    ----------
    dataset_name : str
        Logical name of the dataset (e.g., "lhs_var80_seed3001")
    task : str | None
        Optional task name to help locate the dataset. If provided,
        the dataset is searched under TRAIN_ROOT / <task> / <dataset_name>.

    Returns
    -------
    Path
        Physical path to the dataset directory.

    """
    if task:
        return get_train_root() / task / dataset_name
    return get_train_root() / dataset_name


def resolve_run_output_dir(task: str, run_name: str) -> Path:
    """
    Resolve a run output directory path.

    Run outputs are stored under TRAIN_ROOT / <task> / runs / <run_name>
    or similar convention managed by the training orchestration layer.

    Parameters
    ----------
    task : str
        Task name (e.g., "steady_flow")
    run_name : str
        Run name (e.g., "fno_m128x160_h64_l3_s9")

    Returns
    -------
    Path
        Physical path to the run output directory.

    """
    return get_train_root() / task / "runs" / run_name


def resolve_split_indices_path(run_dir: Path | str) -> Path:
    """
    Resolve the split indices file path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to split_indices.pt file.

    """
    return Path(run_dir) / "split_indices.pt"


def resolve_normalizer_path(run_dir: Path | str) -> Path:
    """
    Resolve the normalizer state file path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to normalizer.pt file.

    """
    return Path(run_dir) / "normalizer.pt"
