"""
===============================================================================
common_paths.py
===============================================================================
Resolve project, storage, dataset, generated-data, run, and artifact paths.

Responsibilities:
  - Read independent dataset, generated-data, and output roots
  - Provide stable logical dataset and run-directory resolvers
  - Resolve split-index, normalizer, checkpoint and artifact paths
  - Identify current saved-run directories by their required artifact files

Design principles:
  - Environment variables are the canonical path source
  - Dataset inputs never derive from a run/output override
  - Defaults keep local notebook and Docker execution usable
  - Callers pass logical names instead of hardcoded storage paths
  - The current run contract is explicit and centralized

Boundaries:
  - Dataset membership belongs to datasets and training code
  - Experiment semantics belong to experiments.config

Notes:
  Default paths:
    - missing environment variables resolve relative to the repository and storage roots
===============================================================================

"""

import json
import os
from pathlib import Path

RUN_CONFIG_FILENAME = "config.yaml"
RUN_NORMALIZER_FILENAME = "normalizer.pt"
RUN_BEST_CHECKPOINT_FILENAME = "best_checkpoint.pt"
RUN_LAST_CHECKPOINT_FILENAME = "last_checkpoint.pt"
RUN_SPLIT_INDICES_FILENAME = "split_indices.pt"
RUN_SUMMARY_FILENAME = "summary.json"
CURRENT_RUN_REQUIRED_FILES = (
    RUN_CONFIG_FILENAME,
    RUN_SPLIT_INDICES_FILENAME,
    RUN_NORMALIZER_FILENAME,
    RUN_BEST_CHECKPOINT_FILENAME,
    RUN_LAST_CHECKPOINT_FILENAME,
    RUN_SUMMARY_FILENAME,
)
RESUME_RUN_REQUIRED_FILES = (
    RUN_CONFIG_FILENAME,
    RUN_SPLIT_INDICES_FILENAME,
    RUN_NORMALIZER_FILENAME,
    RUN_LAST_CHECKPOINT_FILENAME,
    RUN_SUMMARY_FILENAME,
)


def get_project_root() -> Path:
    """
    Return the project root.

    Returns
    -------
    Path
        Root selected by ``PROJECT_ROOT`` or inferred from this module.

    """
    root = os.environ.get("PROJECT_ROOT")
    if root:
        return Path(root)
    # Default: navigate up from this file
    return Path(__file__).parent.parent.parent.parent


def get_storage_root() -> Path:
    """
    Return the shared storage root.

    Returns
    -------
    Path
        Root selected by ``STORAGE_ROOT`` or the project sibling default.

    """
    root = os.environ.get("STORAGE_ROOT")
    if root:
        return Path(root)
    # Default: sibling directory to repository
    return get_project_root().parent / "storage"


def get_dataset_root() -> Path:
    """
    Return the root containing immutable task datasets.

    Returns
    -------
    Path
        Root selected by ``DATASET_ROOT`` or the repository storage default.

    """
    root = os.environ.get("DATASET_ROOT")
    if root:
        return Path(root).expanduser()
    return get_storage_root() / "data_training"


def get_generated_data_root() -> Path:
    """
    Return the root containing raw and processed generated data.

    Returns
    -------
    Path
        Root selected by ``GENERATED_DATA_ROOT`` or the storage default.

    """
    root = os.environ.get("GENERATED_DATA_ROOT")
    if root:
        return Path(root).expanduser()
    return get_storage_root() / "data_generation"


def get_output_root() -> Path:
    """
    Return the root containing training runs and tuning studies.

    Returns
    -------
    Path
        Root selected by ``OUTPUT_ROOT`` or the storage default.

    """
    root = os.environ.get("OUTPUT_ROOT")
    if root:
        return Path(root).expanduser()
    return get_storage_root() / "model_outputs"


def validate_logical_name(value: object, *, label: str) -> str:
    """Return one safe non-empty logical path component."""
    if not isinstance(value, str) or not value or value.strip() != value:
        msg = f"{label} must be a single non-empty path component, got {value!r}."
        raise ValueError(msg)
    if value in {".", ".."} or Path(value).is_absolute() or "/" in value or "\\" in value or "\x00" in value:
        msg = f"{label} must be a single non-empty path component, got {value!r}."
        raise ValueError(msg)
    return value


def resolve_dataset_dir(dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one logical dataset directory.

    Parameters
    ----------
    dataset_id : str
        Non-empty logical dataset identifier.
    dataset_root : Path | str | None, optional
        Explicit dataset root. Output roots are never consulted.

    Returns
    -------
    Path
        ``<dataset_root>/<dataset_id>``.

    """
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    root = Path(dataset_root).expanduser() if dataset_root is not None else get_dataset_root()
    return root / dataset_id


def resolve_dataset_path(dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one logical merged-dataset file.

    Parameters
    ----------
    dataset_id : str
        Non-empty logical dataset identifier.
    dataset_root : Path | str | None, optional
        Explicit dataset root. Output roots are never consulted.

    Returns
    -------
    Path
        ``<dataset_root>/<dataset_id>/<dataset_id>.pt``.

    """
    return resolve_dataset_dir(dataset_id, dataset_root=dataset_root) / f"{dataset_id}.pt"


def resolve_generated_batch_dir(
    dataset_id: str,
    *,
    stage: str,
    generated_data_root: Path | str | None = None,
) -> Path:
    """
    Resolve one generated-data batch directory.

    Parameters
    ----------
    dataset_id : str
        Non-empty logical dataset identifier.
    stage : str
        Exact generated-data stage: ``raw`` or ``processed``.
    generated_data_root : Path | str | None, optional
        Explicit generated-data root.

    Returns
    -------
    Path
        ``<generated_data_root>/<stage>/<dataset_id>``.

    """
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    if stage not in {"raw", "processed"}:
        msg = f"stage must be 'raw' or 'processed', got {stage!r}."
        raise ValueError(msg)
    root = Path(generated_data_root).expanduser() if generated_data_root is not None else get_generated_data_root()
    return root / stage / dataset_id


def resolve_run_output_dir(
    task: str,
    run_name: str,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """
    Resolve a run output directory independently of dataset inputs.

    Parameters
    ----------
    task : str
        Registered task identifier.
    run_name : str
        Run name.
    output_root : Path | str | None, optional
        Explicit run/output root.

    Returns
    -------
    Path
        ``<output_root>/<task>/runs/<run_name>``.

    """
    task = validate_logical_name(task, label="task")
    run_name = validate_logical_name(run_name, label="run_name")
    root = Path(output_root).expanduser() if output_root is not None else get_output_root()
    return root / task / "runs" / run_name


def resolve_optuna_trial_dir(
    task: str,
    study_name: str,
    trial_number: int,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """
    Resolve one study-qualified Optuna trial directory.

    Parameters
    ----------
    task : str
        Registered task identifier.
    study_name : str
        Non-empty study identity.
    trial_number : int
        Non-negative Optuna trial number.
    output_root : Path | str | None, optional
        Explicit output root.

    Returns
    -------
    Path
        ``<output_root>/<task>/optuna/<study>/trials/trial_<number>``.

    """
    task = validate_logical_name(task, label="task")
    study_name = validate_logical_name(study_name, label="study_name")
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        msg = f"trial_number must be a non-negative integer, got {trial_number!r}."
        raise ValueError(msg)
    root = Path(output_root).expanduser() if output_root is not None else get_output_root()
    return root / task / "optuna" / study_name / "trials" / f"trial_{trial_number:06d}"


def resolve_runs_root(task: str, *, output_root: Path | str | None = None) -> Path:
    """
    Resolve the directory containing saved runs for a task.

    Parameters
    ----------
    task : str
        Registered task identifier.
    output_root : Path | str | None, optional
        Explicit run/output root.

    Returns
    -------
    Path
        Directory containing run output directories.

    """
    task = validate_logical_name(task, label="task")
    root = Path(output_root).expanduser() if output_root is not None else get_output_root()
    return root / task / "runs"


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
    return Path(run_dir) / RUN_BEST_CHECKPOINT_FILENAME


def resolve_last_checkpoint_file(run_dir: Path | str) -> Path:
    """
    Resolve the exact-resume checkpoint path within a run directory.

    Parameters
    ----------
    run_dir : Path | str
        Run output directory path.

    Returns
    -------
    Path
        Path to last_checkpoint.pt.

    """
    return Path(run_dir) / RUN_LAST_CHECKPOINT_FILENAME


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
        Required config, split, normalizer, best/last checkpoint, and summary paths.

    """
    run_dir = Path(run_dir)
    return tuple(run_dir / filename for filename in CURRENT_RUN_REQUIRED_FILES)


def missing_resume_run_files(run_dir: Path | str) -> tuple[Path, ...]:
    """
    Return files required before an explicit resume can inspect a run.

    A best checkpoint is deliberately not required because an interrupted run
    may have completed epochs before its first evaluation.
    """
    run_dir = Path(run_dir)
    return tuple(run_dir / filename for filename in RESUME_RUN_REQUIRED_FILES if not (run_dir / filename).is_file())


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

    This is a shallow discovery predicate. It requires every current run
    artifact plus a valid summary whose status is ``completed``.
    Consumers still call the full lifecycle validator before loading content.

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
    if not run_dir.is_dir() or missing_current_run_files(run_dir):
        return False
    summary_path = resolve_run_summary_path(run_dir)
    try:
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(summary, dict) and summary.get("schema_version") == 1 and summary.get("status") == "completed"


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
    dataset_name = validate_logical_name(dataset_name, label="dataset_name")
    return resolve_analysis_root(run_dir) / "ood" / dataset_name
