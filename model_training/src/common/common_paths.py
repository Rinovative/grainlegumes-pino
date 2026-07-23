"""
===============================================================================
common_paths.py
===============================================================================
Resolve project, storage, dataset, run and artifact paths.

Responsibilities:
  - Read project and storage root environment variables
  - Provide stable dataset and run-directory resolvers
  - Resolve split-index, normalizer, checkpoint and artifact paths
  - Identify current saved-run directories by their required artifact files

Design principles:
  - Environment variables are the canonical path source
  - Defaults keep local notebook and Docker execution usable
  - Callers pass logical names instead of hardcoded storage paths
  - The current run contract is explicit and centralized

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

RUN_CONFIG_FILENAME = "config.yaml"
RUN_NORMALIZER_FILENAME = "normalizer.pt"
RUN_CHECKPOINT_FILENAME = "best_checkpoint.pt"
RUN_SPLIT_INDICES_FILENAME = "split_indices.pt"
RUN_SUMMARY_FILENAME = "summary.json"
CURRENT_RUN_REQUIRED_FILES = (
    RUN_CONFIG_FILENAME,
    RUN_CHECKPOINT_FILENAME,
    RUN_NORMALIZER_FILENAME,
    RUN_SPLIT_INDICES_FILENAME,
)


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


def resolve_runs_root(task: str) -> Path:
    """
    Resolve the directory containing saved runs for a task.

    Parameters
    ----------
    task : str
        Task name (e.g., "steady_flow").

    Returns
    -------
    Path
        Directory containing run output directories.

    """
    return get_train_root() / task / "runs"


def resolve_run_config_path(run_dir: Path | str) -> Path:
    """
    Resolve the current run configuration path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to config.yaml file.

    """
    return Path(run_dir) / RUN_CONFIG_FILENAME


def resolve_best_checkpoint_file(run_dir: Path | str) -> Path:
    """
    Resolve the current best checkpoint path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to best_checkpoint.pt file.

    """
    return Path(run_dir) / RUN_CHECKPOINT_FILENAME


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
    return Path(run_dir) / RUN_SPLIT_INDICES_FILENAME


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
    return Path(run_dir) / RUN_NORMALIZER_FILENAME


def resolve_run_summary_path(run_dir: Path | str) -> Path:
    """
    Resolve the current run summary path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to summary.json file.

    """
    return Path(run_dir) / RUN_SUMMARY_FILENAME


def resolve_current_run_required_paths(run_dir: Path | str) -> tuple[Path, ...]:
    """
    Resolve required file paths for the current saved-run contract.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Required run files: config, checkpoint, normalizer and split indices.

    """
    run_dir = Path(run_dir)
    return tuple(run_dir / filename for filename in CURRENT_RUN_REQUIRED_FILES)


def missing_current_run_files(run_dir: Path | str) -> tuple[Path, ...]:
    """
    Return required current-contract run files missing from a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Missing required file paths.

    """
    return tuple(path for path in resolve_current_run_required_paths(run_dir) if not path.is_file())


def is_current_run_dir(run_dir: Path | str) -> bool:
    """
    Return whether a directory satisfies the current saved-run contract.

    A current run directory must contain:
        - config.yaml
        - best_checkpoint.pt
        - normalizer.pt
        - split_indices.pt

    summary.json is part of the full run contract but is not required for
    artifact discovery, so interrupted-but-loadable runs can still be inspected.

    Parameters
    ----------
    run_dir : Path | str
        Candidate run output directory path.

    Returns
    -------
    bool
        True if the directory contains the required current run files.

    """
    run_dir = Path(run_dir)
    return run_dir.is_dir() and not missing_current_run_files(run_dir)


def resolve_analysis_root(run_dir: Path | str) -> Path:
    """
    Resolve the analysis artifact root for a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to the run's analysis artifact root.

    """
    return Path(run_dir) / "analysis"


def resolve_id_analysis_dir(run_dir: Path | str) -> Path:
    """
    Resolve the in-distribution artifact directory for a run.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to analysis/id.

    """
    return resolve_analysis_root(run_dir) / "id"


def resolve_ood_analysis_dir(run_dir: Path | str, dataset_name: str) -> Path:
    """
    Resolve the OOD artifact directory for a run and dataset.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.
    dataset_name : str
        Logical OOD dataset name.

    Returns
    -------
    Path
        Path to analysis/ood/<dataset_name>.

    """
    return resolve_analysis_root(run_dir) / "ood" / dataset_name
