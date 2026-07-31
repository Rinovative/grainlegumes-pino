"""
===============================================================================
common_paths.py
===============================================================================
Resolve the two public data domains and their owned runtime paths.

Responsibilities:
  - Resolve generation and model-training roots from their two public variables
  - Derive each domain's ``meta``, ``raw``, and ``processed`` lifecycle stages
  - Resolve final datasets, metadata snapshots, runs, studies, and artifacts
  - Identify current saved-run directories by their required artifact files

Design principles:
  - Application paths never derive scientific data locations from host storage
  - Repository-local domain paths are the portable defaults in every runtime
  - Training datasets and outputs share one model-training ownership boundary
  - Logical names are validated as single path components before composition
  - The current saved-run file contract is explicit and centralized

This module does NOT:
  - Create datasets, runs, checkpoints, summaries, or analysis artifacts
  - Decide dataset membership, experiment semantics, or resume eligibility
  - Validate artifact contents beyond the shallow completed-run predicate
===============================================================================
"""

import hashlib
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


def get_generated_data_root() -> Path:
    """
    Return the authoritative generation-domain root.

    ``GENERATED_DATA_ROOT`` is the only public override. The portable default
    is the repository-local ``data_generation/data`` mount target.
    """
    root = os.environ.get("GENERATED_DATA_ROOT")
    if root:
        return Path(root).expanduser()
    return get_project_root() / "data_generation" / "data"


def get_model_training_data_root() -> Path:
    """
    Return the self-contained model-training data-domain root.

    ``MODEL_TRAINING_DATA_ROOT`` is the only public override. The portable
    default is the repository-local ``model_training/data`` mount target.
    """
    root = os.environ.get("MODEL_TRAINING_DATA_ROOT")
    if root:
        return Path(root).expanduser()
    return get_project_root() / "model_training" / "data"


def get_generation_meta_root() -> Path:
    """Return the authoritative generation metadata stage."""
    return get_generated_data_root() / "meta"


def get_generation_raw_root() -> Path:
    """Return the authoritative generated raw-input stage."""
    return get_generated_data_root() / "raw"


def get_generation_processed_root() -> Path:
    """Return the authoritative generated COMSOL-output stage."""
    return get_generated_data_root() / "processed"


def get_training_meta_root() -> Path:
    """Return the validated model-training metadata-snapshot stage."""
    return get_model_training_data_root() / "meta"


def get_training_raw_root() -> Path:
    """Return the immutable final-dataset stage used by training."""
    return get_model_training_data_root() / "raw"


def get_training_processed_root() -> Path:
    """Return the stage owning runs, studies, logs, and acceptance outputs."""
    return get_model_training_data_root() / "processed"


def get_training_state_root(*, model_training_data_root: Path | str | None = None) -> Path:
    """Return the hidden root for transient model-training coordination state."""
    root = Path(model_training_data_root).expanduser() if model_training_data_root is not None else get_model_training_data_root()
    return root / ".state"


def get_dataset_build_locks_root(*, model_training_data_root: Path | str | None = None) -> Path:
    """Return the persistent OS-lock-anchor root for dataset publication."""
    return get_training_state_root(model_training_data_root=model_training_data_root) / "dataset-builds" / "locks"


def get_dataset_build_transactions_root(*, model_training_data_root: Path | str | None = None) -> Path:
    """Return the recoverable dataset-publication transaction registry."""
    return get_training_state_root(model_training_data_root=model_training_data_root) / "dataset-builds" / "transactions"


def get_run_locks_root(*, model_training_data_root: Path | str | None = None) -> Path:
    """Return the persistent OS-lock-anchor root for saved-run writers."""
    return get_training_state_root(model_training_data_root=model_training_data_root) / "runs" / "locks"


def resolve_dataset_build_lock_path(
    dataset_id: str,
    *,
    model_training_data_root: Path | str | None = None,
) -> Path:
    """Resolve one dataset builder's persistent advisory-lock anchor."""
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    return get_dataset_build_locks_root(model_training_data_root=model_training_data_root) / f"dataset-{dataset_id}.lock"


def resolve_dataset_build_transaction_path(
    dataset_id: str,
    *,
    model_training_data_root: Path | str | None = None,
) -> Path:
    """Resolve one dataset builder's durable recovery marker."""
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    return get_dataset_build_transactions_root(model_training_data_root=model_training_data_root) / f"dataset-{dataset_id}.json"


def resolve_run_lock_path(
    run_dir: Path | str,
    *,
    model_training_data_root: Path | str | None = None,
) -> Path:
    """Resolve one path-qualified saved-run writer lock below hidden state."""
    canonical_run = Path(run_dir).expanduser().resolve(strict=False)
    digest = hashlib.sha256(os.fsencode(canonical_run)).hexdigest()
    return get_run_locks_root(model_training_data_root=model_training_data_root) / f"run-{digest}.lock"


def resolve_artifact_lock_path(
    artifact_dir: Path | str,
    *,
    model_training_data_root: Path | str | None = None,
) -> Path:
    """Resolve one path-qualified analysis-artifact lock below hidden run state."""
    canonical_artifact = Path(artifact_dir).expanduser().resolve(strict=False)
    digest = hashlib.sha256(os.fsencode(canonical_artifact)).hexdigest()
    return get_run_locks_root(model_training_data_root=model_training_data_root) / f"artifact-{digest}.lock"


def get_dataset_root() -> Path:
    """
    Return the derived final-dataset root.

    This derived convenience reads no independent dataset-root variable; callers needing
    a bounded alternate location pass an explicit resolver or CLI override.
    """
    return get_training_raw_root()


def get_output_root() -> Path:
    """
    Return the derived training and evaluation output root.

    This derived convenience reads no independent output-root variable; callers needing a
    bounded alternate location pass an explicit resolver or CLI override.
    """
    return get_training_processed_root()


def validate_logical_name(value: object, *, label: str) -> str:
    """
    Validate one logical identifier for safe use as a path component.

    Parameters
    ----------
    value : object
        Candidate identifier. It must be a non-empty, already-trimmed string
        containing no separator, NUL, absolute path, ``.`` or ``..`` value.
    label : str
        Contract name included in validation errors.

    Returns
    -------
    str
        The unchanged validated component.

    Raises
    ------
    ValueError
        If ``value`` is not exactly one safe logical component.

    """
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

    Raises
    ------
    ValueError
        If ``dataset_id`` is not one safe logical path component.

    """
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    root = Path(dataset_root).expanduser() if dataset_root is not None else get_dataset_root()
    return root / dataset_id


def resolve_dataset_metadata_dir(
    dataset_id: str,
    *,
    metadata_root: Path | str | None = None,
) -> Path:
    """
    Resolve one dataset's validated metadata snapshot directory.

    The directory is distinct from the authoritative ``.pt`` payload and
    contains only small, builder-validated training/evaluation provenance.
    """
    dataset_id = validate_logical_name(dataset_id, label="dataset_id")
    root = Path(metadata_root).expanduser() if metadata_root is not None else get_training_meta_root()
    return root / dataset_id


def resolve_dataset_path(dataset_id: str, *, dataset_root: Path | str | None = None) -> Path:
    """
    Resolve one logical final training-dataset file.

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

    Raises
    ------
    ValueError
        If ``dataset_id`` is not one safe logical path component.

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

    Raises
    ------
    ValueError
        If ``dataset_id`` is unsafe or ``stage`` is not exactly ``"raw"`` or
        ``"processed"``.

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

    Raises
    ------
    ValueError
        If ``task`` or ``run_name`` is not one safe logical path component.

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
        ``<output_root>/<task>/studies/<study>/trials/trial_<number>``.

    Raises
    ------
    ValueError
        If task/study identifiers are unsafe or ``trial_number`` is a boolean,
        non-integer, or negative value.

    """
    task = validate_logical_name(task, label="task")
    study_name = validate_logical_name(study_name, label="study_name")
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        msg = f"trial_number must be a non-negative integer, got {trial_number!r}."
        raise ValueError(msg)
    return resolve_study_dir(task, study_name, output_root=output_root) / "trials" / f"trial_{trial_number:06d}"


def resolve_study_dir(
    task: str,
    study_name: str,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve one Optuna study below the task-owned ``studies`` subtree."""
    task = validate_logical_name(task, label="task")
    study_name = validate_logical_name(study_name, label="study_name")
    root = Path(output_root).expanduser() if output_root is not None else get_output_root()
    return root / task / "studies" / study_name


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
        ``<output_root>/<task>/runs``.

    Raises
    ------
    ValueError
        If ``task`` is not one safe logical path component.

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

    Parameters
    ----------
    run_dir : Path | str
        Existing or prospective run output directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Missing config, split, normalizer, last-checkpoint, and summary paths in
        contract order. A best checkpoint is deliberately not required because
        an interrupted run may precede its first evaluation.

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
        ``True`` only when every required current-run file exists and the JSON
        summary has schema version 1 with status ``"completed"``.

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
    if not isinstance(summary, dict):
        return False
    schema_version = summary.get("schema_version")
    return isinstance(schema_version, int) and not isinstance(schema_version, bool) and schema_version == 1 and summary.get("status") == "completed"


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
        Path to ``analysis/ood/<dataset_name>``.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not one safe logical path component.

    """
    dataset_name = validate_logical_name(dataset_name, label="dataset_name")
    return resolve_analysis_root(run_dir) / "ood" / dataset_name
